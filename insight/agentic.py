"""Deterministic next-best-action suggestions.

Zero LLM cost and zero latency: candidate follow-ups are derived from the
dataset profile, what has already been analyzed, and (for multi-table
sessions) columns shared across tables. The UI renders them as one-click
follow-up chips; the planner's briefing path can reuse them too.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from insight.domain.dataset import ColumnRole, DatasetProfile


def _analyzed_text(state) -> str:
    if state is None:
        return ""
    parts = [*(state.metrics or []), *(state.dims or [])]
    for digest in getattr(state, "recent_results", []) or []:
        parts.append(digest.question)
        parts.extend(digest.operations)
    return " ".join(parts).lower()


def suggest_followups(
    profile: DatasetProfile,
    state=None,
    tables: Optional[dict] = None,
    max_items: int = 3,
) -> list[str]:
    """Return up to `max_items` high-value follow-up questions.

    Ranking: joins across tables > untested association > trend of an unused
    metric > outlier sweep of an unused numeric > slice comparison.
    Already-analyzed columns are skipped so suggestions stay novel.
    """
    seen_text = _analyzed_text(state)
    candidates: list[str] = []

    def add(text: str, *keywords: str) -> None:
        if any(k in seen_text for k in keywords if k):
            return
        if any(text.lower() == c.lower() for c in candidates):
            return
        candidates.append(text)

    numerics = [
        c for c in profile.columns
        if c.role in (ColumnRole.numeric, ColumnRole.datetime) and c.missing_pct < 0.5
    ]
    categoricals = [
        c for c in profile.columns
        if c.role in (ColumnRole.categorical, ColumnRole.boolean)
        and 1 < c.unique_count <= 25
    ]
    dates = [
        c for c in profile.columns
        if c.role == ColumnRole.datetime
        and not re.search(r"birth|born|dob|expir", c.name, re.IGNORECASE)
    ]
    metrics = _meaningful_metrics(numerics)

    if not metrics:
        return []

    shared = _shared_columns(tables or {})
    if shared and "join" not in seen_text:
        add(
            f"Join {shared[0][0]} with {shared[1]} on {shared[0][1]} and compare "
            f"{metrics[0].name} across the joined groups.",
            "join", shared[0][1],
        )

    if categoricals and metrics and "statistical_test" not in seen_text:
        cat, met = categoricals[0].name, metrics[0].name
        add(
            f"Is {met} associated with {cat}? Test it statistically.",
            "statistical_test", cat,
        )

    if dates and metrics and "trend" not in seen_text:
        add(
            f"How does {metrics[0].name} trend over time?",
            "trend", dates[0].name,
        )

    unused_metric = next(
        (m.name for m in metrics if m.name.lower() not in seen_text), None
    )
    if unused_metric and "detect_outliers" not in seen_text:
        add(
            f"Are there outliers in {unused_metric}? Which rows are unusual?",
            "detect_outliers", unused_metric,
        )

    if len(categoricals) >= 2 and metrics:
        dim = next((c.name for c in categoricals if c.name.lower() not in seen_text), None)
        if dim:
            add(
                f"Compare {metrics[0].name} across {dim} — which group leads?",
                "group_aggregate", dim,
            )

    if categoricals and metrics and "calibration" not in seen_text:
        met = next(
            (
                m.name for m in metrics
                if 0 < m.min is not None and m.max is not None and m.min >= 0 and m.max <= 1
            ),
            None,
        )
        binary = next(
            (
                c.name for c in numerics
                if c.unique_count == 2 and c.name not in (met or "")
            ),
            None,
        )
        if met and binary:
            add(
                f"Audit whether {met} is calibrated against {binary}.",
                "isotonic_calibration", met, binary,
            )

    return candidates[:max_items]


def _meaningful_metrics(numerics) -> list:
    """Numeric columns worth analysing: not ids, not codes, not constants."""
    out = []
    for column in numerics:
        if column.role != ColumnRole.numeric:
            continue
        low = column.name.lower()
        if low.endswith("_id") or low in {"zip_code", "index", "year", "month"}:
            continue
        if low.startswith(("id_", "unnamed")):
            continue
        if column.unique_count <= 1:
            continue
        out.append(column)
    return out


def _shared_columns(tables: dict) -> list[tuple[str, str]]:
    """Find a plausible join key shared by two tables.

    Only identifier-like columns qualify (name ends in _id / equals id, or the
    column is numeric and high-cardinality). Joining on free-text columns like
    first names is never suggested.
    """
    if len(tables) < 2:
        return []
    shared: dict[str, list[str]] = {}
    for table_name, frame in tables.items():
        for column in getattr(frame, "columns", []):
            shared.setdefault(str(column), []).append(table_name)

    def key_score(column: str, table_list: list[str]) -> float:
        low = column.lower()
        frame = next(
            f for name, f in tables.items() if name == table_list[0]
        )
        series = frame[column]
        if low == "id" or low.endswith("_id"):
            return 0.0
        if pd.api.types.is_numeric_dtype(series):
            uniqueness = series.nunique(dropna=True) / max(1, len(series))
            if uniqueness > 0.8:
                return 1.0
        return 99.0

    ranked = []
    for column, table_list in shared.items():
        if len(table_list) < 2 or column.lower().startswith("unnamed"):
            continue
        score = key_score(column, table_list)
        if score < 90:
            ranked.append((score, -len(table_list), column, table_list))
    ranked.sort()
    if not ranked:
        return []
    _, _, column, table_list = ranked[0]
    return [(table_list[0], column), (table_list[1])]
