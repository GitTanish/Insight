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
    def _mask_sql(sql: str) -> tuple[str, str]:
        """Return two same-length masked copies of `sql`.

        `masked` has every character inside a single-quoted string,
        dollar-quoted string, double-quoted identifier, line comment or block
        comment replaced by a space — statement splitting and keyword scans
        then never see text that is not SQL. `executable` blanks only the
        comment spans, preserving literals, because inline comments would
        otherwise swallow the wrapper query's closing parenthesis.
        """
        masked = list(sql)
        executable = list(sql)
        n = len(sql)
        i = 0
        while i < n:
            ch = sql[i]
            nxt = sql[i + 1] if i + 1 < n else ""
            end: Optional[int] = None
            comment = False
            if ch in ("'", '"'):
                j = i + 1
                while j < n:
                    if sql[j] == ch:
                        if j + 1 < n and sql[j + 1] == ch:
                            j += 2
                            continue
                        break
                    j += 1
                end = min(j + 1, n)
            elif ch == "-" and nxt == "-":
                newline = sql.find("\n", i + 2)
                end = n if newline == -1 else newline
                comment = True
            elif ch == "/" and nxt == "*":
                close = sql.find("*/", i + 2)
                end = n if close == -1 else close + 2
                comment = True
            elif ch == "$":
                tag_match = re.match(r"\$[A-Za-z_0-9]*\$", sql[i:])
                if tag_match:
                    tag = tag_match.group(0)
                    close = sql.find(tag, i + len(tag))
                    end = n if close == -1 else close + len(tag)
            if end is None:
                i += 1
                continue
            for k in range(i, end):
                masked[k] = " "
                if comment:
                    executable[k] = " "
            i = end
        return "".join(masked), "".join(executable)

    def validate_statement(self, query: str) -> str:
        original = query.strip()
        masked, executable = self._mask_sql(original)
        spans: list[tuple[int, int]] = []
        start = 0
        for idx, char in enumerate(masked):
            if char == ";":
                spans.append((start, idx))
                start = idx + 1
        spans.append((start, len(masked)))
        non_empty = [(s, e) for s, e in spans if original[s:e].strip()]
        if len(non_empty) != 1:
            raise ValueError(
                "exactly one SQL statement is allowed (no chained statements)"
            )
        lo, hi = non_empty[0]
        statement = executable[lo:hi].strip()
        masked_statement = masked[lo:hi].strip()
        head = masked_statement.split(None, 1)[0].upper() if masked_statement.split() else ""
        if head not in _ALLOWED_LEADING:
            raise ValueError(
                f"only SELECT/WITH queries are allowed (got '{head or 'empty'}')"
            )
        for keyword in _BLOCKED_KEYWORDS:
            if re.search(rf"\b{keyword}\b", masked_statement, flags=re.I):
                raise ValueError(f"'{keyword}' statements are not permitted")
        return statement

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
