from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from insight.settings import get_settings

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_BLOCKED_KEYWORDS = (
    "ATTACH", "DETACH", "COPY", "INSTALL", "LOAD", "PRAGMA", "SET",
    "CALL", "EXPORT", "IMPORT", "CREATE", "INSERT", "UPDATE", "DELETE",
    "DROP", "ALTER", "VACUUM", "CHECKPOINT", "SUMMARIZE", "DESCRIBE",
    "SHOW", "GRANT", "REVOKE", "USE", "BEGIN", "COMMIT", "ROLLBACK",
)

_ALLOWED_LEADING = ("SELECT", "WITH")


def sanitize_table_name(name: str) -> str:
    stem = Path(name).stem.lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if not cleaned or not _IDENTIFIER.match(cleaned):
        cleaned = f"t_{cleaned}" if cleaned else "t"
    return cleaned


def unique_table_names(filenames: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for filename in filenames:
        base = sanitize_table_name(filename)
        candidate, suffix = base, 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        mapping[filename] = candidate
    return mapping


class DuckSession:
    """Embedded read-only analytics session over registered DataFrames.

    External file/http access is disabled after registration; queries are
    restricted to single SELECT/WITH statements and bounded by an interrupt
    watchdog plus a hard row cap.
    """

    def __init__(self, datasets: dict[str, pd.DataFrame]):
        settings = get_settings()
        self.table_names = unique_table_names(list(datasets))
        self.con = duckdb.connect(":memory:")
        for filename, table in self.table_names.items():
            self.con.register(table, datasets[filename])
        try:
            self.con.execute("SET enable_external_access=false")
            self.con.execute("SET autoinstall_known_extensions=false")
            self.con.execute("SET autoload_known_extensions=false")
            self.con.execute("SET lock_configuration=true")
        except duckdb.Error:
            pass
        self.timeout_s = max(1.0, float(getattr(settings, "sql_timeout_s", 10)))
        self.max_rows = max(100, int(getattr(settings, "sql_max_output_rows", 10_000)))
        self.last_truncated = False

    def close(self) -> None:
        try:
            self.con.close()
        except Exception:
            pass

    def __enter__(self) -> "DuckSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _strip_comments(sql: str) -> str:
        sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
        sql = re.sub(r"--[^\n]*", " ", sql)
        return sql.strip()

    def validate_statement(self, query: str) -> str:
        sql = self._strip_comments(query)
        parts = [p.strip() for p in sql.split(";") if p.strip()]
        if len(parts) != 1:
            raise ValueError(
                "exactly one SQL statement is allowed (no chained statements)"
            )
        statement = parts[0]
        head = statement.split(None, 1)[0].upper() if statement.split() else ""
        if head not in _ALLOWED_LEADING:
            raise ValueError(
                f"only SELECT/WITH queries are allowed (got '{head or 'empty'}')"
            )
        for keyword in _BLOCKED_KEYWORDS:
            if re.search(rf"\b{keyword}\b", statement, flags=re.I):
                raise ValueError(f"'{keyword}' statements are not permitted")
        return statement.rstrip(";").strip()

    def execute(self, query: str) -> pd.DataFrame:
        statement = self.validate_statement(query)
        guarded = f"SELECT * FROM ({statement}) AS __insight_q LIMIT {self.max_rows + 1}"

        timer: Optional[threading.Timer] = threading.Timer(
            self.timeout_s, self._interrupt
        )
        timer.daemon = True
        timer.start()
        try:
            frame = self.con.execute(guarded).fetchdf()
        except duckdb.Error as exc:
            message = str(exc)
            if "interrupt" in message.lower():
                raise ValueError(
                    f"query exceeded the {self.timeout_s:.0f}s limit and was interrupted"
                ) from exc
            raise ValueError(message[:400]) from exc
        finally:
            timer.cancel()

        self.last_truncated = len(frame) > self.max_rows
        if self.last_truncated:
            frame = frame.head(self.max_rows)
        return frame

    def describe(self) -> str:
        """Compact schema text for the planner prompt (exact table + column names)."""
        lines: list[str] = []
        try:
            info = self.con.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
            ).fetchall()
        except duckdb.Error:
            return "(schema unavailable)"
        counts: dict[str, int] = {}
        for table in dict.fromkeys(self.table_names.values()):
            try:
                counts[table] = int(
                    self.con.execute(f"SELECT count(*) FROM \"{table}\"").fetchone()[0]
                )
            except duckdb.Error:
                counts[table] = 0
        current: Optional[str] = None
        for table, column, dtype in info:
            if table != current:
                lines.append(f"- {table} ({counts.get(table, 0):,} rows): ")
                current = table
            lines[-1] += f"{column} {dtype}, "
        return "\n".join(line.rstrip(", ") for line in lines)

    def _interrupt(self) -> None:
        try:
            self.con.interrupt()
        except Exception:
            pass
