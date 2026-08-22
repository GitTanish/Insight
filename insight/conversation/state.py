from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import BaseModel, Field

from insight.domain.analysis import AnalysisPlan


class ActiveFilter(BaseModel):
    column: str
    op: str
    value: Any = None

    def describe(self) -> str:
        rendered = "null" if self.value is None else json.dumps(self.value, default=str)
        return f"{self.column} {self.op} {rendered}"


class AnalysisSessionState(BaseModel):
    dataset_id: str = ""
    active_filters: list[ActiveFilter] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    dims: list[str] = Field(default_factory=list)
    turns_analyzed: int = 0

    def is_empty(self) -> bool:
        return not (self.active_filters or self.metrics or self.dims)


def _filters_from_step(step) -> list[ActiveFilter]:
    params = step.params if isinstance(step.params, dict) else {}
    if step.operation == "filter_rows":
        column, op = params.get("column"), params.get("op")
        if isinstance(column, str) and isinstance(op, str):
            return [ActiveFilter(column=column, op=op, value=params.get("value"))]
        return []
    found: list[ActiveFilter] = []
    for key in ("filters", "subset_a", "subset_b"):
        entries = params.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if (
                isinstance(entry, dict)
                and isinstance(entry.get("column"), str)
                and isinstance(entry.get("op"), str)
            ):
                found.append(
                    ActiveFilter(
                        column=entry["column"], op=entry["op"], value=entry.get("value")
                    )
                )
    return found


def _add_unique(target: list[str], values: Any) -> None:
    if not isinstance(values, list):
        values = [values]
    for value in values:
        if isinstance(value, str):
            canonical = value.strip()
            if canonical and not any(
                existing.lower() == canonical.lower() for existing in target
            ):
                target.append(canonical)


def extract_state_from_plan(
    plan: AnalysisPlan,
    base: Optional[AnalysisSessionState] = None,
) -> AnalysisSessionState:
    state = base.model_copy(deep=True) if base is not None else AnalysisSessionState()
    for step in plan.steps:
        for new_filter in _filters_from_step(step):
            state.active_filters = [
                existing
                for existing in state.active_filters
                if not (
                    existing.column.lower() == new_filter.column.lower()
                    and existing.op == new_filter.op
                )
            ]
            state.active_filters.append(new_filter)

        params = step.params if isinstance(step.params, dict) else {}
        if step.operation == "group_aggregate":
            _add_unique(state.dims, params.get("group_by"))
            metrics = params.get("metrics")
            if isinstance(metrics, list):
                for metric in metrics:
                    if isinstance(metric, dict):
                        _add_unique(
                            state.metrics,
                            metric.get("alias") or metric.get("column"),
                        )
        elif step.operation == "trend":
            _add_unique(state.dims, params.get("date_column"))
            _add_unique(state.metrics, params.get("metric_column"))
        elif step.operation == "regression":
            _add_unique(state.metrics, params.get("target_column"))
            _add_unique(state.metrics, params.get("feature_columns"))

    state.turns_analyzed += 1
    return state


def serialize_state_for_planner(state: Optional[AnalysisSessionState]) -> str:
    if state is None or state.is_empty():
        return "(no active filters or tracked fields)"
    lines: list[str] = []
    if state.active_filters:
        rendered = "; ".join(f.describe() for f in state.active_filters)
        lines.append(f"active row filters (already applied to prior answers): {rendered}")
    if state.dims:
        lines.append(f"dimensions analyzed so far: {', '.join(state.dims)}")
    if state.metrics:
        lines.append(f"metrics analyzed so far: {', '.join(state.metrics)}")
    return "\n".join(lines)
