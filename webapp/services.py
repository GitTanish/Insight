from __future__ import annotations

import hashlib
import io
import uuid
from pathlib import Path

import pandas as pd

from insight.analytics.sql_engine import sanitize_table_name
from insight.domain.dataset import DatasetProfile
from insight.profiling.profiler import profile_dataframe
from insight.settings import get_settings

ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
DELIMITERS = [",", ";", "\t"]


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


def parse_csv_content(content: bytes) -> tuple[pd.DataFrame | None, str | None]:
    settings = get_settings()
    last_error = "empty file"
    for encoding in ENCODINGS:
        for delimiter in DELIMITERS:
            try:
                df = pd.read_csv(
                    io.BytesIO(content), encoding=encoding, delimiter=delimiter
                )
                if not df.empty and len(df.columns) > 0:
                    if len(df) > settings.max_upload_rows:
                        return (
                            None,
                            f"File has {len(df):,} rows (max {settings.max_upload_rows:,})",
                        )
                    return df, None
                last_error = f"parsed with {encoding}/{delimiter!r} but empty"
            except Exception as exc:
                last_error = f"{encoding}/{delimiter!r}: {exc}"
    return None, f"Could not parse CSV. Last attempt — {last_error}"


EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def parse_tables(
    filename: str, content: bytes
) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    """Parse an upload into named tables.

    CSVs yield a single table; Excel workbooks yield one table per non-empty
    sheet so the DuckDB layer can join across sheets. Returns
    ({table_name: dataframe}, None) or (None, error).
    """
    from pathlib import Path as _Path

    settings = get_settings()
    suffix = _Path(filename or "").suffix.lower()

    frames: dict[str, pd.DataFrame] = {}
    if suffix in EXCEL_EXTENSIONS:
        try:
            workbook = pd.read_excel(io.BytesIO(content), sheet_name=None)
        except Exception as exc:
            return None, f"Could not parse Excel file: {exc}"
        stem = sanitize_table_name(_Path(filename).stem)
        seen: set[str] = set()
        for sheet_name, df in workbook.items():
            if df is None or df.empty or len(df.columns) == 0:
                continue
            base = sanitize_table_name(f"{stem}_{sheet_name}")
            candidate, n = base, 2
            while candidate in seen:
                candidate = f"{base}_{n}"
                n += 1
            seen.add(candidate)
            frames[candidate] = df
        if not frames:
            return None, "Workbook contains no readable sheets."
    else:
        df, err = parse_csv_content(content)
        if df is None:
            return None, err
        frames[sanitize_table_name(_Path(filename).stem)] = df

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


def session_artifacts_dir(session_id: str) -> Path:
    root = get_settings().artifacts_dir / "web" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def clear_session_artifacts(session_id: str) -> None:
    import shutil

    root = get_settings().artifacts_dir / "web" / session_id
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


def build_report_docx_bytes(
    turns: list[dict], dataset_name: str, date_str: str,
    findings: list[dict] | None = None,
) -> bytes:
    from insight.reports.docx import build_report_docx as _build

    return _build(turns, dataset_name, date_str, findings=findings)
