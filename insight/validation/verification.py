from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from insight.analytics.operations import (
    _fmt_edge,
    _mask_for_filter,
    _numeric,
    apply_filters,
    fold_other_levels,
)
from insight.domain.analysis import AnalysisPlan, Filter
from insight.domain.execution import ExecutionResult
from insight.domain.visualization import ValidationIssue

_TOLERANCE = 1e-6


def _is_na(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _values_close(a, b) -> bool:
    a_na, b_na = _is_na(a), _is_na(b)
    if a_na or b_na:
        return a_na and b_na
    try:
        a_f, b_f = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if b_f == 0:
        return abs(a_f) < 1e-9
    return abs(a_f - b_f) / abs(b_f) <= _TOLERANCE


def _table_value_close(got, expected) -> bool:
    """Compare a table cell (already _scalar-rounded) with a recomputed value."""
    if _is_na(got) or _is_na(expected):
        return _is_na(got) and _is_na(expected)
    if isinstance(got, float) or isinstance(expected, float):
        try:
            if got == round(float(expected), 6):
                return True
        except (TypeError, ValueError):
            pass
    return _values_close(got, expected)


def _key_token(value) -> str:
    return "<NA>" if _is_na(value) else str(value)


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _metric_dict(metric) -> dict:
    if isinstance(metric, dict):
        return dict(metric)
    if hasattr(metric, "model_dump"):
        try:
            return metric.model_dump()
        except Exception:
            return {}
    return {}


def _coerce_filters(raw) -> list[Filter]:
    filters: list[Filter] = []
    for item in raw or []:
        if isinstance(item, Filter):
            filters.append(item)
        elif isinstance(item, dict):
            filters.append(Filter(**item))
    return filters


def _recompute_mismatch(step, detail: str) -> ValidationIssue:
    return ValidationIssue(
        severity="error",
        message=f"independent recomputation mismatch ({detail})",
        step_id=step.step_id,
    )


def _rounded_close(got, expected, digits: int = 6) -> bool:
    """Compare a rounded calculation against its full-precision recomputation."""
    if _is_na(got) or _is_na(expected):
        return _is_na(got) and _is_na(expected)
    try:
        return float(got) == round(float(expected), digits)
    except (TypeError, ValueError):
        return _values_close(got, expected)


def _calculations_by_step(execution: ExecutionResult, step_id: int) -> dict:
    for result in execution.steps:
        if (
            result.step_id == step_id
            and result.success
            and result.output is not None
        ):
            return {calc.label: calc.value for calc in result.output.calculations}
    return {}


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
    params = step.params if isinstance(step.params, dict) else {}
    group_by = params.get("group_by")
    if isinstance(group_by, str):
        group_cols = [group_by]
    elif isinstance(group_by, (list, tuple)) and len(group_by) > 0:
        group_cols = [str(c) for c in group_by]
    else:
        return issues
    if any(col not in df.columns for col in group_cols):
        return issues
    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    if work.empty:
        return issues
    other_after = _as_int(params.get("other_after"), 0) or None
    if len(group_cols) == 1:
        work, _folded = fold_other_levels(work, group_cols[0], other_after)
    grouped = work.groupby(group_cols, dropna=False, observed=True)
    expected_metrics: dict[str, pd.Series] = {}
    for metric in params.get("metrics", []):
        spec = _metric_dict(metric)
        agg = getattr(spec.get("agg", "count"), "value", spec.get("agg", "count"))
        agg = str(agg)
        col = spec.get("column")
        alias = spec.get("alias") or (f"{agg}_{col}" if col else "row_count")
        if agg == "count":
            expected_metrics[alias] = grouped.size()
        else:
            if col not in work.columns:
                continue
            series = work[col] if agg == "nunique" else _numeric(work[col])
            expected_metrics[alias] = series.groupby(
                [work[c] for c in group_cols], dropna=False, observed=True
            ).agg(agg)
    for name in [*group_cols, *expected_metrics]:
        if name not in table.columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"group_aggregate output column '{name}' missing from result",
                )
            )
    if not expected_metrics:
        return issues
    expected = pd.DataFrame(expected_metrics).reset_index()
    sort_by = params.get("sort_by")
    if sort_by and sort_by in expected.columns:
        expected = expected.sort_values(
            str(sort_by),
            ascending=_as_bool(params.get("ascending", False)),
            na_position="last",
        )
    top_n = params.get("top_n")
    if top_n:
        expected = expected.head(max(1, _as_int(top_n, len(expected))))
    if len(table.rows) > len(expected) or (
        not table.truncated and len(table.rows) < len(expected)
    ):
        issues.append(
            _recompute_mismatch(
                step,
                f"group_aggregate returned {len(table.rows)} rows, expected "
                f"{len(expected)}",
            )
        )
    metric_aliases = list(expected_metrics)
    column_values = {c: list(expected[c]) for c in group_cols}
    expected_entries = [
        (
            tuple(_key_token(column_values[c][i]) for c in group_cols),
            {a: expected[a].iloc[i] for a in metric_aliases},
        )
        for i in range(len(expected))
    ]
    got: dict[tuple[str, ...], dict[str, object]] = {}
    for row in table.rows:
        key = tuple(_key_token(row[table.columns.index(c)]) for c in group_cols)
        got[key] = {
            a: row[table.columns.index(a)] for a in metric_aliases if a in table.columns
        }
    expected_keys = {key for key, _ in expected_entries}
    for key, values in expected_entries:
        label = " / ".join(key)
        if key not in got:
            if not table.truncated:
                for alias in metric_aliases:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"independent recomputation mismatch ({alias}): "
                                f"group '{label}' missing from result"
                            ),
                            step_id=step.step_id,
                        )
                    )
            continue
        for alias in metric_aliases:
            r_val = got[key].get(alias)
            g_val = values[alias]
            if not _values_close(r_val, g_val):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"independent recomputation mismatch ({alias}): "
                            f"group '{label}' reported {r_val} but engine recomputes {g_val}"
                        ),
                        step_id=step.step_id,
                    )
                )
    for key in got:
        if key not in expected_keys:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"independent recomputation mismatch: group "
                        f"'{' / '.join(key)}' is not in the recomputed result"
                    ),
                    step_id=step.step_id,
                )
            )
    return issues


def _verify_top_n(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    sort_col = params.get("sort_by")
    n = max(1, _as_int(params.get("n", 10), 10))
    if not table.rows or sort_col not in df.columns:
        return issues
    if len(table.rows) > n:
        issues.append(
            ValidationIssue(
                severity="error",
                message=f"independent check failed (top_n): returned {len(table.rows)} rows, requested {n}",
                step_id=step.step_id,
            )
        )
    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    if work.empty:
        return issues
    work = work.reset_index(drop=True)
    ascending = _as_bool(params.get("ascending", False))
    sort_key = _numeric(work[sort_col])
    ordered = work.iloc[
        sort_key.sort_values(ascending=ascending, na_position="last", kind="stable").index
    ].head(n)
    expected_len = len(ordered)
    projected = params.get("columns")
    expected_columns = (
        [str(name) for name in projected]
        if isinstance(projected, (list, tuple)) and projected
        else list(work.columns)
    )
    for name in expected_columns:
        if name not in table.columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"top_n output column '{name}' missing from result",
                )
            )
    shared_cols = [c for c in table.columns if c in ordered.columns]
    if shared_cols:
        for position in range(min(len(table.rows), expected_len)):
            row = table.rows[position]
            for col_name in shared_cols:
                got_val = row[table.columns.index(col_name)]
                expected_val = ordered.iloc[position][col_name]
                if not _table_value_close(got_val, expected_val):
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            message=(
                                f"independent check failed (top_n): row {position} column "
                                f"'{col_name}' reports {got_val} but recomputed top-{n} "
                                f"expects {expected_val}"
                            ),
                            step_id=step.step_id,
                        )
                    )
                    return issues
    if len(table.rows) < expected_len and not table.truncated:
        issues.append(
            ValidationIssue(
                severity="error",
                message=(
                    f"independent check failed (top_n): returned {len(table.rows)} rows "
                    f"but {expected_len} rows are selectable"
                ),
                step_id=step.step_id,
            )
        )
    return issues


def _verify_value_counts(df, step, table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    col = params.get("column")
    if col not in df.columns or not table.rows:
        return issues
    expected_columns = [str(col), "count"]
    if _as_bool(params.get("normalize", False)):
        expected_columns.append("proportion")
    for name in expected_columns:
        if name not in table.columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"value_counts output column '{name}' missing from result",
                )
            )
    if "count" not in table.columns:
        return issues
    if table.columns[0] != str(col):
        issues.append(
            _recompute_mismatch(
                step,
                f"value_counts label column '{col}' is missing or misplaced",
            )
        )
        return issues
    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    raw = work[col]
    valid_total = int(raw.notna().sum())
    counts = raw.value_counts()
    other_after = _as_int(params.get("other_after"), 0) or None
    if other_after is not None and len(counts) > other_after:
        kept = counts.iloc[: other_after - 1]
        folded = int(counts.iloc[other_after - 1:].sum())
        counts = pd.concat([kept, pd.Series({f"Other ({folded:,} rows)": folded})])
    top_n = max(1, _as_int(params.get("top_n", 15), 15))
    counts = counts.head(top_n)
    expected = {str(label): int(value) for label, value in counts.items()}
    if len(table.rows) > len(expected) or (
        not table.truncated and len(table.rows) < len(expected)
    ):
        issues.append(
            _recompute_mismatch(
                step,
                f"value_counts returned {len(table.rows)} rows, expected {len(expected)}",
            )
        )
    count_idx = table.columns.index("count")
    got = {str(r[0]): r[count_idx] for r in table.rows}
    for label, value in got.items():
        if label not in expected:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"independent recomputation mismatch (value_counts): "
                        f"'{label}' is not among the recomputed top values"
                    ),
                    step_id=step.step_id,
                )
            )
            continue
        if not _values_close(value, expected[label]):
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"independent recomputation mismatch (value_counts): "
                        f"'{label}' reported {value} but engine recomputes {expected[label]}"
                    ),
                    step_id=step.step_id,
                )
            )
    for label, value in expected.items():
        if label not in got:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"independent recomputation mismatch (value_counts): "
                        f"'{label}' missing from result (expected {value})"
                    ),
                    step_id=step.step_id,
                )
            )
    counted_total = sum(
        int(r[count_idx]) for r in table.rows if not _is_na(r[count_idx])
    )
    if counted_total > valid_total:
        issues.append(
            ValidationIssue(
                severity="error",
                message=(
                    f"independent check failed (value_counts): counts sum to "
                    f"{counted_total:,} but the filtered dataset has only "
                    f"{valid_total:,} non-null rows"
                ),
                step_id=step.step_id,
            )
        )
    if "proportion" in table.columns:
        pidx = table.columns.index("proportion")
        for row in table.rows:
            label = str(row[0])
            if label not in expected:
                continue
            expected_p = expected[label] / valid_total if valid_total else 0.0
            try:
                got_p = float(row[pidx])
            except (TypeError, ValueError):
                continue
            if _is_na(got_p) or abs(got_p - expected_p) > _TOLERANCE + 5e-5:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"independent recomputation mismatch (value_counts): "
                            f"'{label}' proportion reported {got_p} but engine "
                            f"recomputes {expected_p:.4f}"
                        ),
                        step_id=step.step_id,
                    )
                )
    first_label = str(table.rows[0][0]) if table.rows else ""
    if expected:
        expected_top = next(iter(expected))
        if first_label != expected_top:
            issues.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"independent recomputation mismatch (value_counts): most frequent "
                        f"value is '{expected_top}', table reports '{first_label}'"
                    ),
                    step_id=step.step_id,
                )
            )
    return issues


def _verify_correlation(df, step, table, calcs: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    if not table.columns or table.columns[0] != "column":
        return [
            _recompute_mismatch(
                step,
                "correlation output is missing the 'column' index column",
            )
        ]
    if not table.rows:
        return [_recompute_mismatch(step, "correlation output has no rows")]

    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    if work.empty:
        return issues

    requested = params.get("columns")
    if isinstance(requested, (list, tuple)) and requested:
        candidates = [str(name) for name in requested]
        if any(name not in df.columns for name in candidates):
            return issues
    else:
        candidates = list(work.select_dtypes(include=["number"]).columns)
    if not candidates:
        return issues

    numeric = work[candidates].apply(_numeric)
    binary_flags = [
        name for name in numeric.columns if numeric[name].dropna().nunique() <= 2
    ]
    continuous = [
        name
        for name in numeric.columns
        if name not in binary_flags and numeric[name].nunique(dropna=True) > 1
    ]
    if not continuous:
        return issues

    method = str(params.get("method", "pearson"))
    corr = numeric[continuous].corr(method=method).round(4)
    expected_labels = [str(name) for name in corr.columns]
    expected = {
        str(row_label): {
            str(column_label): corr.iloc[row_index, column_index]
            for column_index, column_label in enumerate(corr.columns)
        }
        for row_index, row_label in enumerate(corr.index)
    }

    expected_columns = ["column", *expected_labels]
    for name in expected_columns:
        if name not in table.columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation output column '{name}' missing from result",
                )
            )
    for name in table.columns:
        if name not in expected_columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation output column '{name}' is not in the "
                    "recomputed matrix",
                )
            )

    row_labels = [str(row[0]) for row in table.rows]
    seen: set[str] = set()
    for label in row_labels:
        if label in seen:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation row '{label}' appears more than once",
                )
            )
        seen.add(label)
    expected_set = set(expected_labels)
    for label in expected_labels:
        if label not in seen and not table.truncated:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation row '{label}' missing from result",
                )
            )
    for label in seen:
        if label not in expected_set:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation row '{label}' is not in the recomputed matrix",
                )
            )
    if len(row_labels) > len(expected_labels):
        issues.append(
            _recompute_mismatch(
                step,
                f"correlation returned {len(row_labels)} rows, expected "
                f"{len(expected_labels)}",
            )
        )
    elif len(row_labels) < len(expected_labels) and not table.truncated:
        issues.append(
            _recompute_mismatch(
                step,
                f"correlation returned {len(row_labels)} rows, expected "
                f"{len(expected_labels)}",
            )
        )

    for row in table.rows:
        row_label = str(row[0])
        if row_label not in expected:
            continue
        for column_label, expected_value in expected[row_label].items():
            if column_label not in table.columns:
                continue
            got_value = row[table.columns.index(column_label)]
            if not _table_value_close(got_value, expected_value):
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"correlation [{row_label}][{column_label}] reported "
                        f"{got_value} but engine recomputes {expected_value}",
                    )
                )

    if "method" in calcs and str(calcs["method"]) != method:
        issues.append(
            _recompute_mismatch(
                step,
                f"correlation method reported {calcs['method']} but engine "
                f"recomputes {method}",
            )
        )
    if "excluded_binary_columns" in calcs and not _rounded_close(
        calcs["excluded_binary_columns"], len(binary_flags), 0
    ):
        issues.append(
            _recompute_mismatch(
                step,
                "correlation excluded_binary_columns reported "
                f"{calcs['excluded_binary_columns']} but engine recomputes "
                f"{len(binary_flags)}",
            )
        )

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if pd.notna(value):
                pairs.append(
                    (abs(float(value)), corr.columns[i], corr.columns[j], float(value))
                )
    pairs.sort(reverse=True)
    if pairs:
        strongest = round(pairs[0][3], 6)
        if "strongest_correlation" in calcs and not _rounded_close(
            calcs["strongest_correlation"], strongest, 6
        ):
            issues.append(
                _recompute_mismatch(
                    step,
                    "correlation strongest_correlation reported "
                    f"{calcs['strongest_correlation']} but engine recomputes "
                    f"{strongest}",
                )
            )
    for strength, name_a, name_b, value in pairs[:3]:
        if strength < 0.5:
            continue
        label = f"{name_a} x {name_b}"
        if label in calcs and not _rounded_close(calcs[label], value, 6):
            issues.append(
                _recompute_mismatch(
                    step,
                    f"correlation {label} reported {calcs[label]} but engine "
                    f"recomputes {round(value, 6)}",
                )
            )
    return issues


def _verify_compare_subsets(df, step, table, calcs: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    metric = _metric_dict(params.get("metric") or {})
    agg = str(getattr(metric.get("agg", "count"), "value", metric.get("agg", "count")))
    column = metric.get("column")
    label_a = str(params.get("label_a", "subset_a"))
    label_b = str(params.get("label_b", "subset_b"))
    if label_a == label_b:
        return issues

    try:
        frame_a = apply_filters(df, _coerce_filters(params.get("subset_a", [])))
        frame_b = apply_filters(df, _coerce_filters(params.get("subset_b", [])))
    except Exception:
        return issues
    if frame_a.empty or frame_b.empty:
        return issues

    expected: list[tuple[str, pd.DataFrame, float]] = []
    for label, frame in ((label_a, frame_a), (label_b, frame_b)):
        if agg == "count":
            value = float(len(frame))
        else:
            if not column or column not in frame.columns:
                return issues
            values = _numeric(frame[column]).dropna()
            if values.empty:
                return issues
            if agg == "nunique":
                value = float(values.nunique())
            else:
                try:
                    value = float(getattr(values, agg)())
                except (AttributeError, TypeError, ValueError):
                    return issues
        expected.append((label, frame, value))

    required = ("subset", "rows", "value")
    missing = [name for name in required if name not in table.columns]
    for name in missing:
        issues.append(
            _recompute_mismatch(
                step,
                f"compare_subsets output column '{name}' missing from result",
            )
        )
    if missing:
        return issues
    label_idx = table.columns.index("subset")
    rows_idx = table.columns.index("rows")
    value_idx = table.columns.index("value")
    if len(table.rows) != 2:
        issues.append(
            _recompute_mismatch(
                step,
                f"compare_subsets returned {len(table.rows)} rows, expected 2",
            )
        )
        return issues

    for (label, frame, value), row in zip(expected, table.rows):
        if _key_token(row[label_idx]) != label:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"compare_subsets row label {row[label_idx]!r} does not match {label!r}",
                )
            )
        if not _table_value_close(row[rows_idx], len(frame)):
            issues.append(
                _recompute_mismatch(
                    step,
                    f"compare_subsets {label}.rows reported {row[rows_idx]} but engine "
                    f"recomputes {len(frame)}",
                )
            )
        if not _table_value_close(row[value_idx], value):
            issues.append(
                _recompute_mismatch(
                    step,
                    f"compare_subsets {label}.value reported {row[value_idx]} but engine "
                    f"recomputes {value}",
                )
            )

    delta = expected[1][2] - expected[0][2]
    if "absolute_delta" in calcs and not _rounded_close(calcs["absolute_delta"], delta):
        issues.append(
            _recompute_mismatch(
                step,
                f"compare_subsets absolute_delta reported {calcs['absolute_delta']} but "
                f"engine recomputes {round(delta, 6)}",
            )
        )

    if agg == "mean" and column:
        for label, frame, value in expected:
            values = _numeric(frame[column]).dropna()
            n = len(values)
            if n < 30:
                continue
            half = 1.96 * float(values.std(ddof=1)) / float(np.sqrt(n))
            for calc_label, expected_value in (
                (f"ci95_low_{label}", value - half),
                (f"ci95_high_{label}", value + half),
            ):
                if calc_label in calcs and not _rounded_close(
                    calcs[calc_label], expected_value
                ):
                    issues.append(
                        _recompute_mismatch(
                            step,
                            f"compare_subsets {calc_label} reported {calcs[calc_label]} "
                            f"but engine recomputes {round(expected_value, 6)}",
                        )
                    )
    return issues


def _verify_anova(df, step, table, calcs: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    group_column = params.get("group_column")
    value_column = params.get("value_column")
    if not isinstance(group_column, str) or not isinstance(value_column, str):
        return issues
    if group_column not in df.columns or value_column not in df.columns:
        return issues

    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    if work.empty:
        return issues

    raw = work[[group_column, value_column]].copy()
    raw[value_column] = _numeric(raw[value_column])
    raw = raw.dropna()
    if raw.empty:
        return issues
    grouped = {
        str(level): group[value_column].to_numpy(dtype=float)
        for level, group in raw.groupby(group_column)
    }
    if len(grouped) < 3 or min(len(values) for values in grouped.values()) < 2:
        return issues

    try:
        statistic, p_value = sps.f_oneway(*grouped.values())
    except Exception:
        return issues
    all_values = np.concatenate(list(grouped.values()))
    grand_mean = float(all_values.mean())
    ss_between = sum(
        float(len(values)) * float(values.mean() - grand_mean) ** 2
        for values in grouped.values()
    )
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

    required = (group_column, "n", "mean", "std")
    missing = [name for name in required if name not in table.columns]
    for name in missing:
        issues.append(
            _recompute_mismatch(
                step,
                f"anova_test output column '{name}' missing from result",
            )
        )
    if missing:
        return issues
    group_idx = table.columns.index(group_column)
    n_idx = table.columns.index("n")
    mean_idx = table.columns.index("mean")
    std_idx = table.columns.index("std")
    got = {
        _key_token(row[group_idx]): (row[n_idx], row[mean_idx], row[std_idx])
        for row in table.rows
    }

    summary = raw.copy()
    summary[group_column] = summary[group_column].astype(str)
    agg = summary.groupby(group_column)[value_column].agg(
        n="count", mean="mean", std="std"
    )
    expected = {
        str(level): (int(row["n"]), float(row["mean"]), float(row["std"]))
        for level, row in agg.iterrows()
    }
    if len(table.rows) > len(expected) or (
        not table.truncated and len(table.rows) < len(expected)
    ):
        issues.append(
            _recompute_mismatch(
                step,
                f"anova_test returned {len(table.rows)} rows, expected {len(expected)}",
            )
        )
    for level, (n_expected, mean_expected, std_expected) in expected.items():
        if level not in got:
            if not table.truncated:
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"anova_test group '{level}' missing from result",
                    )
                )
            continue
        n_got, mean_got, std_got = got[level]
        for field, got_value, expected_value in (
            ("n", n_got, n_expected),
            ("mean", mean_got, mean_expected),
            ("std", std_got, std_expected),
        ):
            if not _table_value_close(got_value, expected_value):
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"anova_test group '{level}'.{field} reported {got_value} but "
                        f"engine recomputes {expected_value}",
                    )
                )
    for level in got:
        if level not in expected:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"anova_test group '{level}' is not in the recomputed result",
                )
            )

    for label, expected_value, digits in (
        ("statistic", float(statistic), 6),
        ("p_value", float(p_value), 6),
        ("effect_size", eta_squared, 4),
    ):
        if label in calcs and not _rounded_close(calcs[label], expected_value, digits):
            issues.append(
                _recompute_mismatch(
                    step,
                    f"anova_test {label} reported {calcs[label]} but engine recomputes "
                    f"{round(expected_value, digits)}",
                )
            )
    return issues


def _verify_regression(df, step, table, calcs: dict) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    params = step.params if isinstance(step.params, dict) else {}
    target = params.get("target_column")
    raw_features = params.get("feature_columns")
    if not isinstance(target, str) or not isinstance(raw_features, (list, tuple)):
        return issues
    features = [str(name) for name in raw_features]
    if not features or len(set(features)) != len(features):
        return issues
    columns = [target, *features]
    if any(name not in df.columns for name in columns):
        return issues

    try:
        work = apply_filters(df, _coerce_filters(params.get("filters", [])))
    except Exception:
        return issues
    if work.empty:
        return issues

    total_rows = len(work)
    data = work[columns].apply(_numeric).dropna()
    n = len(data)
    k = len(features)
    if n < k + 2:
        return issues

    y = data[target].to_numpy(dtype=float)
    raw_x = data[features].to_numpy(dtype=float)
    design = np.column_stack([np.ones(n), raw_x])
    xtx = design.T @ design
    try:
        condition = float(np.linalg.cond(xtx))
        if not np.isfinite(condition) or condition > 1e12:
            return issues
        betas = np.linalg.solve(xtx, design.T @ y)
        cov_factor = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        return issues
    if not np.all(np.isfinite(betas)):
        return issues

    residuals = y - design @ betas
    rss = float(residuals @ residuals)
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - rss / tss if tss > 0 else 0.0
    dof = n - k - 1
    adjusted = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else r_squared
    sigma2 = rss / dof if dof > 0 else 0.0
    std_errors = np.sqrt(np.clip(np.diag(sigma2 * cov_factor), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(std_errors > 0, betas / std_errors, np.nan)
    p_values = [
        float(2.0 * sps.t.sf(abs(t_stat), dof)) if np.isfinite(t_stat) else None
        for t_stat in t_stats
    ]

    y_std = float(y.std(ddof=1))
    standardized = []
    for j in range(k):
        x_std = float(raw_x[:, j].std(ddof=1))
        standardized.append(
            float(betas[j + 1] * x_std / y_std)
            if y_std > 0 and x_std > 0
            else None
        )

    required = (
        "term",
        "coefficient",
        "std_error",
        "t_stat",
        "p_value",
        "standardized_beta",
    )
    missing = [name for name in required if name not in table.columns]
    for name in missing:
        issues.append(
            _recompute_mismatch(
                step,
                f"regression output column '{name}' missing from result",
            )
        )
    if missing:
        return issues
    term_idx = table.columns.index("term")
    got = {str(row[term_idx]): row for row in table.rows}
    expected = [
        ("intercept", betas[0], std_errors[0], t_stats[0], p_values[0], None)
    ]
    for j, feature in enumerate(features):
        expected.append(
            (
                feature,
                betas[j + 1],
                std_errors[j + 1],
                t_stats[j + 1],
                p_values[j + 1],
                standardized[j],
            )
        )
    if len(table.rows) > len(expected) or (
        not table.truncated and len(table.rows) < len(expected)
    ):
        issues.append(
            _recompute_mismatch(
                step,
                f"regression returned {len(table.rows)} rows, expected {len(expected)}",
            )
        )

    for term, coef, std_error, t_stat, p_value, std_beta in expected:
        row = got.get(term)
        if row is None:
            if not table.truncated:
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"regression term '{term}' missing from result",
                    )
                )
            continue
        for column_name, expected_value in (
            ("coefficient", coef),
            ("std_error", std_error),
            ("t_stat", None if not np.isfinite(t_stat) else t_stat),
            ("p_value", p_value),
            ("standardized_beta", std_beta),
        ):
            if column_name not in table.columns:
                continue
            got_value = row[table.columns.index(column_name)]
            if not _table_value_close(got_value, expected_value):
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"regression {term}.{column_name} reported {got_value} but "
                        f"engine recomputes {expected_value}",
                    )
                )
    expected_terms = {row[0] for row in expected}
    for term in got:
        if term not in expected_terms:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"regression term '{term}' is not in the recomputed model",
                )
            )

    for label, expected_value, digits in (
        ("r_squared", r_squared, 6),
        ("adj_r_squared", adjusted, 6),
        ("n_observations", n, 0),
        ("rows_excluded_incomplete", total_rows - n, 0),
    ):
        if label in calcs and not _rounded_close(calcs[label], expected_value, digits):
            issues.append(
                _recompute_mismatch(
                    step,
                    f"regression {label} reported {calcs[label]} but engine recomputes "
                    f"{round(expected_value, digits)}",
                )
            )
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
        try:
            if step.operation == "group_aggregate":
                issues.extend(_verify_group_aggregate(step_df, step, table))
            elif step.operation == "top_n":
                issues.extend(_verify_top_n(step_df, step, table))
            elif step.operation == "value_counts":
                issues.extend(_verify_value_counts(step_df, step, table))
            elif step.operation == "correlation":
                issues.extend(
                    _verify_correlation(
                        step_df, step, table, _calculations_by_step(execution, step.step_id)
                    )
                )
            elif step.operation == "compare_subsets":
                issues.extend(
                    _verify_compare_subsets(
                        step_df, step, table, _calculations_by_step(execution, step.step_id)
                    )
                )
            elif step.operation == "anova_test":
                issues.extend(
                    _verify_anova(
                        step_df, step, table, _calculations_by_step(execution, step.step_id)
                    )
                )
            elif step.operation == "regression":
                issues.extend(
                    _verify_regression(
                        step_df, step, table, _calculations_by_step(execution, step.step_id)
                    )
                )
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"independent verification skipped for step {step.step_id} "
                        f"({step.operation}): {type(exc).__name__}: {exc}"
                    ),
                    step_id=step.step_id,
                )
            )
    return issues
