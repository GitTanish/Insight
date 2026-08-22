from __future__ import annotations

import numpy as np
import pandas as pd

from insight.analytics.operations import (
    _fmt_edge,
    _mask_for_filter,
    _numeric,
    apply_filters,
)
from insight.domain.analysis import AnalysisPlan, Filter
from insight.domain.execution import ExecutionResult
from insight.domain.visualization import ValidationIssue

_TOLERANCE = 1e-6


def _values_close(a: float, b: float) -> bool:
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= _TOLERANCE


def _effective_frames(df, plan: AnalysisPlan) -> dict[int, pd.DataFrame]:
    """Replay implicit-chain filter_rows mutations so verifiers see the same
    row-set the executor used (steps with explicit input_step stay exempt)."""
    effective: dict[int, pd.DataFrame] = {}
    working = df
    for step in sorted(plan.steps, key=lambda s: s.step_id):
        if step.input_step is not None:
            continue
        effective[step.step_id] = working
        if step.operation == "filter_rows":
            params = step.params if isinstance(step.params, dict) else {}
            try:
                condition = Filter(
                    column=params["column"], op=params["op"], value=params.get("value")
                )
                working = working[_mask_for_filter(working, condition)]
            except Exception:
                pass
    return effective


def _verify_group_aggregate(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = dict(step.params)
    group_by = params.get("group_by")
    if not isinstance(group_by, str) or group_by not in df.columns:
        return issues
    work = apply_filters(
        df,
        [Filter(**f) for f in params.get("filters", [])],
    )
    for metric in params.get("metrics", []):
        agg = metric.get("agg", "count")
        alias = metric.get("alias") or f"{agg}_{metric.get('column')}" if metric.get("column") else None
        alias = alias or "row_count"
        if alias not in table.columns:
            continue
        if agg == "count":
            expected = work.groupby(group_by, observed=True).size()
        else:
            col = metric.get("column")
            if col not in df.columns:
                continue
            expected = (
                _numeric(work[col])
                .groupby(work[group_by])
                .agg(agg)
            )
        got = pd.Series(
            [row[table.columns.index(alias)] for row in table.rows],
            index=[str(r[0]) for r in table.rows],
        )
        expected.index = expected.index.astype(str)
        for key, g_val in expected.items():
            if key not in got.index:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"independent recomputation mismatch ({alias}): "
                            f"group '{key}' missing from result"
                        ),
                        step_id=step.step_id,
                    )
                )
                continue
            r_val = got[key]
            try:
                ok = _values_close(float(r_val), float(g_val))
            except (TypeError, ValueError):
                ok = str(r_val) == str(g_val)
            if not ok:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"independent recomputation mismatch ({alias}): "
                            f"group '{key}' reported {r_val} but engine recomputes {g_val}"
                        ),
                        step_id=step.step_id,
                    )
                )
    return issues


def _verify_top_n(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    sort_col = step.params.get("sort_by")
    n = int(step.params.get("n", 10))
    if not table.rows or sort_col not in df.columns:
        return issues
    idx = table.columns.index(sort_col)
    vals = [_numeric(pd.Series([r[idx]]))[0] for r in table.rows]
    ascending = bool(step.params.get("ascending", False))
    pairs = [
        (vals[i], vals[i + 1]) for i in range(len(vals) - 1)
        if pd.notna(vals[i]) and pd.notna(vals[i + 1])
    ]
    violations = [(a, b) for a, b in pairs if (a > b if ascending else a < b)]
    if violations:
        issues.append(
            ValidationIssue(
                severity="error",
                message=(
                    f"independent check failed (top_n): ordering violates "
                    f"ascending={ascending} at {violations[0]}"
                ),
                step_id=step.step_id,
            )
        )
    if len(table.rows) > n:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"independent check failed (top_n): returned {len(table.rows)} rows, requested {n}",
                step_id=step.step_id,
            )
        )
    return issues


def _verify_value_counts(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    col = step.params.get("column")
    if col not in df.columns or not table.rows:
        return issues
    count_idx = table.columns.index("count")
    counted_total = sum(int(r[count_idx]) for r in table.rows)
    distinct_total = int(df[col].nunique(dropna=True))
    top_expected = df[col].value_counts()
    if counted_total > len(df):
        issues.append(
            ValidationIssue(
                severity="error",
                message=(
                    f"independent check failed (value_counts): counts sum to "
                    f"{counted_total:,} but dataset has only {len(df):,} rows"
                ),
                step_id=step.step_id,
            )
        )
    first_label = str(table.rows[0][0])
    if distinct_total and str(top_expected.index[0]) != first_label:
        issues.append(
            ValidationIssue(
                severity="error",
                message=(
                    f"independent recomputation mismatch (value_counts): most frequent "
                    f"value is '{top_expected.index[0]}', table reports '{first_label}'"
                ),
                step_id=step.step_id,
            )
        )
    return issues


def _verify_correlation(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not table.columns or table.columns[0] != "column":
        return issues
    matrix = {
        row[0]: {c: v for c, v in zip(table.columns[1:], row[1:])}
        for row in table.rows
    }
    for name, row_vals in matrix.items():
        diag = row_vals.get(name)
        try:
            if diag is not None and abs(float(diag) - 1.0) > 1e-6:
                raise ValueError
        except (TypeError, ValueError):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=f"independent check failed (correlation): diagonal for '{name}' is {diag}",
                    step_id=step.step_id,
                )
            )
    checked = 0
    for i, (name_a, row_vals) in enumerate(matrix.items()):
        for name_b, val in row_vals.items():
            mirrored = matrix.get(name_b, {}).get(name_a)
            if mirrored is None:
                continue
            try:
                if abs(float(val) - float(mirrored)) > 1e-6:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"independent check failed (correlation): matrix not "
                                f"symmetric at [{name_a}][{name_b}]"
                            ),
                            step_id=step.step_id,
                        )
                    )
                checked += 1
            except (TypeError, ValueError):
                continue
            if checked > 40:
                break
    return issues


def independently_verify(df, plan: AnalysisPlan, execution: ExecutionResult) -> list[ValidationIssue]:
    tables = execution.tables_by_step()
    effective = _effective_frames(df, plan)
    issues: list[ValidationIssue] = []
    for step in plan.steps:
        if step.input_step is not None:
            continue
        table = tables.get(step.step_id)
        if table is None:
            continue
        step_df = effective.get(step.step_id, df)
        if step.operation == "group_aggregate":
            issues.extend(_verify_group_aggregate(step_df, step, table))
        elif step.operation == "top_n":
            issues.extend(_verify_top_n(step_df, step, table))
        elif step.operation == "value_counts":
            issues.extend(_verify_value_counts(step_df, step, table))
        elif step.operation == "correlation":
            issues.extend(_verify_correlation(step_df, step, table))
    return issues
