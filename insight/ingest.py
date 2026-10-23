"""Shared CSV/Excel ingestion used by the webapp and the toolkit/MCP server."""

from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

import pandas as pd

from insight.analytics.sql_engine import sanitize_table_name
from insight.domain.dataset import DatasetProfile
from insight.profiling.profiler import profile_dataframe
from insight.settings import get_settings

ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
DELIMITERS = [",", ";", "\t"]
EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
SNIFF_BYTES = 64 * 1024
MAX_EXCEL_SHEETS = 64


def _sniff_delimiter(text: str) -> str | None:
    sample = "\n".join(line for line in text.splitlines()[:50] if line.strip())
    if not sample:
        return None
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(DELIMITERS))
    except csv.Error:
        return None
    return dialect.delimiter if dialect.delimiter in DELIMITERS else None


def parse_csv_content(content: bytes) -> tuple[pd.DataFrame | None, str | None]:
    settings = get_settings()
    last_error = "empty file"
    row_limit = max(1, settings.max_upload_rows)
    for encoding in ENCODINGS:
        delimiters = list(DELIMITERS)
        try:
            sample = content[:SNIFF_BYTES].decode(encoding, errors="ignore")
            detected = _sniff_delimiter(sample)
        except (UnicodeDecodeError, LookupError) as exc:
            last_error = f"{encoding}: {exc}"
            continue
        if detected:
            delimiters = [detected] + [d for d in delimiters if d != detected]
        for delimiter in delimiters:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding=encoding,
                    delimiter=delimiter,
                    nrows=row_limit + 1,
                )
                if not df.empty and len(df.columns) > 0:
                    if len(df) > row_limit:
                        return (
                            None,
                            f"File has more than {row_limit:,} rows "
                            f"(max {row_limit:,})",
                        )
                    return df, None
                last_error = f"parsed with {encoding}/{delimiter!r} but empty"
            except Exception as exc:
                last_error = f"{encoding}/{delimiter!r}: {exc}"
    return None, f"Could not parse CSV. Last attempt — {last_error}"


def parse_tables(
    filename: str, content: bytes
) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Parse an upload into named tables.

    CSVs yield a single table; Excel workbooks yield one table per non-empty
    sheet so the DuckDB layer can join across sheets. Returns
    ({table_name: dataframe}, None) or (None, error).
    """
    settings = get_settings()
    suffix = Path(filename or "").suffix.lower()

    frames: dict[str, pd.DataFrame] = {}
    if suffix in EXCEL_EXTENSIONS:
        try:
            workbook = pd.ExcelFile(io.BytesIO(content))
        except Exception as exc:
            return None, f"Could not parse Excel file: {exc}"
        try:
            if len(workbook.sheet_names) > MAX_EXCEL_SHEETS:
                return None, (
                    f"Workbook has {len(workbook.sheet_names):,} sheets "
                    f"(max {MAX_EXCEL_SHEETS})"
                )
            stem = sanitize_table_name(Path(filename).stem)
            seen: set[str] = set()
            for sheet_name in workbook.sheet_names:
                try:
                    df = workbook.parse(
                        sheet_name, nrows=max(1, settings.max_upload_rows) + 1
                    )
                except Exception:
                    continue
                if df is None or df.empty or len(df.columns) == 0:
                    continue
                if len(df) > settings.max_upload_rows:
                    return None, (
                        f"Table '{sheet_name}' has {len(df):,} rows "
                        f"(max {settings.max_upload_rows:,})"
                    )
                base = sanitize_table_name(f"{stem}_{sheet_name}")
                candidate, n = base, 2
                while candidate in seen:
                    candidate = f"{base}_{n}"
                    n += 1
                seen.add(candidate)
                frames[candidate] = df
        finally:
            workbook.close()
        if not frames:
            return None, "Workbook contains no readable sheets."
    else:
        df, err = parse_csv_content(content)
        if df is None:
            return None, err
        frames[sanitize_table_name(Path(filename).stem)] = df

    for name, df in frames.items():
        if len(df) > settings.max_upload_rows:
            return None, (
                f"Table '{name}' has {len(df):,} rows "
                f"(max {settings.max_upload_rows:,})"
            )
    return frames, None


def profile_from_content(content: bytes, name: str) -> DatasetProfile | None:
    df, error = parse_csv_content(content)
    if df is None or error:
        return None
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    return profile_dataframe(df, name, content_hash)
