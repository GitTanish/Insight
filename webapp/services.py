from __future__ import annotations

import uuid
from pathlib import Path

from insight.ingest import (  # noqa: F401  re-exported for existing callers
    DELIMITERS,
    ENCODINGS,
    EXCEL_EXTENSIONS,
    parse_csv_content,
    parse_tables,
    profile_from_content,
)
from insight.settings import get_settings


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


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
