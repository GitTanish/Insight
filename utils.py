from __future__ import annotations

import io
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from insight.domain.dataset import DatasetProfile
from insight.domain.visualization import AnalysisResponse
from insight.profiling.profiler import profile_dataframe
from insight.settings import get_settings

ENCODINGS = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
DELIMITERS = [",", ";", "\t"]


@st.cache_data(show_spinner=False)
def parse_csv(content: bytes) -> tuple[pd.DataFrame | None, str | None]:
    settings = get_settings()
    last_error = "empty file"
    try:
        import io as _io

        import openpyxl  # noqa: F401

        workbook = pd.read_excel(_io.BytesIO(content), sheet_name=None)
        for sheet_df in workbook.values():
            if sheet_df is not None and not sheet_df.empty and len(sheet_df.columns):
                if len(sheet_df) > settings.max_upload_rows:
                    return (
                        None,
                        f"File has {len(sheet_df):,} rows "
                        f"(max {settings.max_upload_rows:,})",
                    )
                return sheet_df, None
    except Exception:
        pass
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
                            f"File has {len(df):,} rows "
                            f"(max {settings.max_upload_rows:,})",
                        )
                    return df, None
                last_error = f"parsed with {encoding}/{delimiter!r} but empty"
            except Exception as exc:
                last_error = f"{encoding}/{delimiter!r}: {exc}"
    return None, f"Could not parse file. Last attempt — {last_error}"


@st.cache_data(show_spinner=False)
def compute_profile(content: bytes, name: str) -> DatasetProfile | None:
    import hashlib

    df, error = parse_csv(content)
    if df is None or error:
        return None
    content_hash = hashlib.sha256(content).hexdigest()[:16]
    return profile_dataframe(df, name, content_hash)


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


@st.cache_data(ttl=900, show_spinner=False)
def discover_models_cached(_force_token: int = 0) -> list[dict]:
    import asyncio

    from insight.llm.registry import discover_models as _discover

    try:
        infos = asyncio.run(_discover())
    except Exception:
        infos = []
    return [i.model_dump() for i in infos]


def ensure_discovery_registered(force: bool = False) -> int:
    from insight.llm import registry as llm_registry

    if force:
        discover_models_cached.clear()

    payloads = discover_models_cached(int(time.time()) if force else 0)
    registered = 0
    for payload in payloads:
        model_id = payload["model_id"]
        if model_id not in llm_registry.DISCOVERED:
            llm_registry.DISCOVERED[model_id] = (
                llm_registry.ModelInfo(**payload)
            )
            registered += 1
    return registered


def session_artifacts_dir(session_id: str) -> Path:
    root = get_settings().artifacts_dir / session_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def clear_session(session_id: str | None) -> None:
    st.session_state.turns = []
    st.session_state.quick_query = None
    st.session_state.analysis_state = None
    st.session_state.session_id = new_session_id()
    if session_id:
        root = get_settings().artifacts_dir / session_id
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)


def build_report_docx(
    turns: list[dict], dataset_name: str, date_str: str,
    findings: list[dict] | None = None,
) -> bytes:
    from insight.reports.docx import build_report_docx as _build

    return _build(turns, dataset_name, date_str, findings=findings)
