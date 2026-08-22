from __future__ import annotations

import inspect
import math
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union, get_args, get_origin

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as _sps_stats

from insight.domain.analysis import (
    AggFunc,
    AnalysisPlan,
    Filter,
    FilterOperator,
    MetricSpec,
)
from insight.domain.dataset import DatasetProfile
from insight.domain.execution import Calculation, DataTable, OperationOutput
from insight.settings import get_settings

_SCALAR_AGG_FUNCS = {
    AggFunc.sum, AggFunc.mean, AggFunc.median,
    AggFunc.min, AggFunc.max, AggFunc.std, AggFunc.var,
}


class OperationError(ValueError):
    pass


def _numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _coerce_value(series: pd.Series, value: Any) -> Any:
    if isinstance(value, list):
        return [_coerce_value(series, v) for v in value]
    if value is None:
        return None
    if pd.api.types.is_datetime64_any_dtype(series):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise OperationError(f"cannot interpret '{value}' as a date")
        return ts
    if pd.api.types.is_bool_dtype(series):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "1", "y"}
    if pd.api.types.is_numeric_dtype(series):
        try:
            num = float(value)
            return int(num) if float(num).is_integer() and isinstance(value, int) else num
        except (TypeError, ValueError) as exc:
            raise OperationError(
                f"'{value}' is not numeric for column '{series.name}'"
            ) from exc
    return str(value)


def _comparison_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return _numeric(series)


def _mask_for_filter(df: pd.DataFrame, f: Filter) -> pd.Series:
    if f.column not in df.columns:
        raise OperationError(f"unknown column '{f.column}'")

    series = df[f.column]

    if f.op == FilterOperator.isna:
        return series.isna()
    if f.op == FilterOperator.notna:
        return series.notna()

    value = _coerce_value(series, f.value)

    if f.op == FilterOperator.eq:
        mask = series == value
        if not mask.any() and not pd.api.types.is_numeric_dtype(series):
            mask = series.astype(str).str.lower() == str(value).lower()
        return mask.fillna(False)
    if f.op == FilterOperator.ne:
        return (series != value).fillna(True)

    comp = _comparison_series(series)

    if f.op == FilterOperator.gt:
        return (comp > value).fillna(False)
    if f.op == FilterOperator.gte:
        return (comp >= value).fillna(False)
    if f.op == FilterOperator.lt:
        return (comp < value).fillna(False)
    if f.op == FilterOperator.lte:
        return (comp <= value).fillna(False)

    if f.op == FilterOperator.in_:
        values = value if isinstance(value, list) else [value]
        lowered = {str(v).lower() for v in values}
        base = series.isin(values)
        alt = series.astype(str).str.lower().isin(lowered)
        return (base | alt).fillna(False)
    if f.op == FilterOperator.not_in:
        values = value if isinstance(value, list) else [value]
        lowered = {str(v).lower() for v in values}
        base = ~series.isin(values)
        alt = ~series.astype(str).str.lower().isin(lowered)
        return (base | alt).fillna(True)

    if f.op == FilterOperator.contains:
        return (
            series.astype(str).str.contains(str(value), case=False, na=False)
        )

    if f.op == FilterOperator.between:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise OperationError("'between' requires a [low, high] pair")
        low = _coerce_value(series, value[0])
        high = _coerce_value(series, value[1])
        return ((comp >= low) & (comp <= high)).fillna(False)

    raise OperationError(f"unsupported operator '{f.op}'")


def apply_filters(df: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    result = df
    for f in filters or []:
        result = result[_mask_for_filter(result, f)]
    if result is df:
        return df.copy()
    return result


def _fmt_edge(v: Any) -> str:
    f = _scalar(v)
    if f is None:
        return "?"
    num = float(f)
    if abs(num) >= 10_000 or float(num).is_integer():
        return f"{int(round(num)):,}"
    sig = f"{num:.4g}"
    return sig


def _scalar(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(v, pd.Timestamp):
        if v == v.normalize():
            return v.date().isoformat()
        return v.isoformat(sep=" ")
    if isinstance(v, pd.Interval):
        left = _fmt_edge(v.left)
        right = _fmt_edge(v.right)
        return f"({left}, {right}]" if v.closed_right else f"[{left}, {right})"
    return str(v)


def _adaptive_bins(values: pd.Series, requested: Optional[int]) -> int:
    n = len(values)
    if requested is not None:
        return max(2, min(int(requested), 100))
    sturges = int(np.ceil(np.log2(n) + 1)) if n > 1 else 5
    iqr = float(values.quantile(0.75) - values.quantile(0.25))
    rng = float(values.max() - values.min())
    if iqr > 0 and rng > 0:
        fd_width = 2 * iqr / (n ** (1 / 3))
        k_fd = int(np.ceil(rng / fd_width))
    else:
        k_fd = sturges
    return max(5, min(30, (sturges + k_fd) // 2))


def _bin_labels(edges: list[float], closed_right: bool = True) -> list[str]:
    labels = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        low_s, high_s = _fmt_edge(low), _fmt_edge(high)
        if i == 0 and not closed_right:
            labels.append(f"[{low_s}, {high_s})")
        elif i == len(edges) - 2:
            labels.append(f"({low_s}, {high_s}]")
        else:
            labels.append(f"[{low_s}, {high_s})")
    return labels


def _merge_sparse(
    counts: list[int], edges: list[float], floor: int
) -> tuple[list[str], list[int], int]:
    merged_away = 0
    while len(counts) > 3 and min(counts) < floor:
        idx = counts.index(min(counts))
        if idx == 0:
            j = 1
        elif idx == len(counts) - 1:
            j = idx - 1
        else:
            j = idx - 1 if counts[idx - 1] <= counts[idx + 1] else idx + 1
        keep, drop = sorted((idx, j))
        counts[keep] += counts[drop]
        del edges[keep + 1]
        del counts[drop]
        merged_away += 1
    return _bin_labels(edges), counts, merged_away


def make_table(step_id: int, name: str, df: pd.DataFrame) -> DataTable:
    settings = get_settings()
    capped = df.head(settings.max_table_rows)
    rows = [[_scalar(v) for v in row] for row in capped.itertuples(index=False)]
    return DataTable(
        step_id=step_id,
        name=name,
        columns=[str(c) for c in df.columns],
        rows=rows,
        total_rows=len(df),
        truncated=len(df) > settings.max_table_rows,
    )


def _fmt_num(v: Any) -> str:
    f = _scalar(v)
    if f is None:
        return "?"
    if isinstance(f, int):
        return f"{f:,}"
    return f"{f:,.4g}"


@dataclass
class OperationSpec:
    name: str
    description: str
    params_model: type[BaseModel]
    run: Callable[[pd.DataFrame, BaseModel, int], OperationOutput]
    mutates: bool = False
    mutate: Optional[Callable[[pd.DataFrame, BaseModel], pd.DataFrame]] = None


OPERATIONS: dict[str, OperationSpec] = {}


def register_operation(name: str, description: str, mutates: bool = False):
    def decorator(cls: type[BaseModel]):
        accepts_ctx = len(inspect.signature(cls._run).parameters) >= 4

        def _runner(df, params, step_id, ctx=None):
            if accepts_ctx:
                return cls._run(df, params, step_id, ctx)
            return cls._run(df, params, step_id)

        OPERATIONS[name] = OperationSpec(
            name=name,
            description=description,
            params_model=cls,
            run=_runner,
            mutates=mutates,
            mutate=getattr(cls, "_mutate", None),
        )
        return cls

    return decorator


class GroupAggregateParams(BaseModel):
    group_by: str | list[str] = Field(description="column name(s) to group by")
    metrics: list[MetricSpec] = Field(
        min_length=1,
        description=(
            "aggregations to compute, each {'column': str, 'agg': "
            "'count|nunique|sum|mean|median|min|max|std|var', 'alias': optional}"
        ),
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")
    sort_by: Optional[str] = Field(default=None, description="output column to sort by")
    ascending: bool = Field(default=False, description="sort ascending")
    top_n: Optional[int] = Field(default=None, ge=1, le=100, description="keep first N rows")
    other_after: Optional[int] = Field(
        default=None,
        ge=3,
        le=50,
        description=(
            "when grouping by ONE column with more distinct values than this, "
            "fold low-frequency levels into an 'Other' row (keeps top N-1 by row count)"
        ),
    )


@register_operation("group_aggregate", "Group rows by column(s) and aggregate metrics")
class GroupAggregateOp(GroupAggregateParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "GroupAggregateOp", step_id: int) -> OperationOutput:
        group_cols = [params.group_by] if isinstance(params.group_by, str) else list(params.group_by)
        missing = [c for c in group_cols if c not in df.columns]
        if missing:
            raise OperationError(f"unknown group_by column(s): {missing}")

        names = [m.output_name() for m in params.metrics]
        if len(names) != len(set(names)):
            raise OperationError("duplicate metric aliases")

        work = apply_filters(df, params.filters)
        if work.empty:
            raise OperationError("no rows after applying filters")

        if (
            params.other_after is not None
            and isinstance(params.group_by, str)
            and work[params.group_by].nunique() > params.other_after
        ):
            gcol = params.group_by
            level_counts = work[gcol].value_counts()
            keep_levels = set(level_counts.index[: params.other_after - 1])
            folded = int(level_counts.iloc[params.other_after - 1:].sum())
            work = work.copy()
            work[gcol] = work[gcol].where(work[gcol].isin(keep_levels), "Other")
            note_extra = f"folded {folded:,} rows from rare levels into 'Other'"
        else:
            note_extra = None

        for metric in params.metrics:
            if metric.column and metric.agg in _SCALAR_AGG_FUNCS:
                if metric.column not in work.columns:
                    raise OperationError(f"unknown metric column '{metric.column}'")
                work[metric.column] = _numeric(work[metric.column])

        grouped = work.groupby(group_cols, dropna=False, observed=True)
        parts: dict[str, pd.Series] = {}
        for metric in params.metrics:
            key = metric.output_name()
            if metric.agg == AggFunc.count:
                parts[key] = grouped.size()
            elif metric.agg == AggFunc.nunique:
                parts[key] = grouped[metric.column].nunique()
            else:
                parts[key] = getattr(grouped[metric.column], metric.agg.value)()

        result = pd.concat(parts, axis=1)
        result.columns = names
        result = result.reset_index()

        total_groups = len(result)
        calcs: list[Calculation] = [
            Calculation(
                label="groups",
                value=total_groups,
                detail=f"aggregated {len(work):,} source rows into {total_groups} groups"
                + (
                    f"; returning top {len(result)} by {params.sort_by}"
                    if params.top_n and len(result) < total_groups
                    else ""
                )
                + (f"; {note_extra}" if note_extra else ""),
            )
        ]
        seen_metric_cols: set[str] = set()
        for metric in params.metrics:
            if (
                metric.agg in _SCALAR_AGG_FUNCS
                and metric.column
                and metric.column not in seen_metric_cols
            ):
                seen_metric_cols.add(metric.column)
                valid = int(grouped[metric.column].count().sum())
                gaps = int(len(work) - valid)
                if gaps:
                    calcs.append(
                        Calculation(
                            label=f"excluded_missing_{metric.output_name()}",
                            value=gaps,
                            detail=(
                                f"{gaps:,} row(s) have missing '{metric.column}': counted "
                                f"in row_count but excluded from {metric.output_name()}"
                            ),
                        )
                    )

        if params.sort_by:
            if params.sort_by not in result.columns:
                raise OperationError(
                    f"sort_by '{params.sort_by}' is not an output column; "
                    f"available: {[str(c) for c in result.columns]}"
                )
            result = result.sort_values(
                params.sort_by, ascending=params.ascending, na_position="last"
            )
        if params.top_n:
            result = result.head(params.top_n)

        return OperationOutput(
            table=make_table(step_id, "grouped", result),
            calculations=calcs,
        )


class SummarizeParams(BaseModel):
    columns: Optional[list[str]] = Field(
        default=None, description="numeric columns to summarize; defaults to all numeric"
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("summarize", "Descriptive statistics (count/mean/std/min/max) per column")
class SummarizeOp(SummarizeParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "SummarizeOp", step_id: int) -> OperationOutput:
        work = apply_filters(df, params.filters)
        if params.columns:
            missing = [c for c in params.columns if c not in df.columns]
            if missing:
                raise OperationError(f"unknown column(s): {missing}")
            candidates = params.columns
        else:
            candidates = [
                c for c in work.select_dtypes(include=["number"]).columns
            ]
        if not candidates:
            raise OperationError("no numeric columns available to summarize")

        records: list[dict[str, Any]] = []
        for col in candidates:
            values = _numeric(work[col]).dropna()
            if values.empty:
                continue
            for agg_name, value in (
                ("count", len(values)),
                ("mean", values.mean()),
                ("std", values.std()),
                ("min", values.min()),
                ("25%", values.quantile(0.25)),
                ("50%", values.median()),
                ("75%", values.quantile(0.75)),
                ("max", values.max()),
            ):
                records.append({"column": col, "metric": agg_name, "value": _scalar(value)})

        table = pd.DataFrame(records, columns=["column", "metric", "value"])
        return OperationOutput(table=make_table(step_id, "summary", table))


class ValueCountsParams(BaseModel):
    column: str = Field(description="categorical column to count")
    top_n: int = Field(default=15, ge=1, le=100, description="limit to N most frequent")
    normalize: bool = Field(default=False, description="include proportion column")
    other_after: Optional[int] = Field(
        default=None,
        ge=3,
        le=50,
        description="when more distinct values than this, fold the tail into 'Other'",
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("value_counts", "Frequency table of unique values in one column")
class ValueCountsOp(ValueCountsParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "ValueCountsOp", step_id: int) -> OperationOutput:
        if params.column not in df.columns:
            raise OperationError(f"unknown column '{params.column}'")
        work = apply_filters(df, params.filters)
        counts = work[params.column].value_counts()

        other_note = None
        if params.other_after is not None and len(counts) > params.other_after:
            kept = counts.iloc[: params.other_after - 1]
            folded = int(counts.iloc[params.other_after - 1:].sum())
            counts = pd.concat([kept, pd.Series({f"Other ({folded:,} rows)": folded})])
            other_note = f"tail of {len(work[params.column].value_counts()) - params.other_after + 1} rare levels folded into 'Other'"

        counts = counts.head(params.top_n)
        table = pd.DataFrame(
            {params.column: [_scalar(v) for v in counts.index],
             "count": [int(c) for c in counts.values]}
        )
        if params.normalize:
            total = int(len(work))
            table["proportion"] = [round(c / total, 4) if total else 0.0 for c in counts.values]
        calc = Calculation(
            label="distinct values",
            value=int(work[params.column].nunique()),
            detail=other_note or f"showing top {len(table)} of {work[params.column].nunique()}",
        )
        return OperationOutput(table=make_table(step_id, "counts", table), calculations=[calc])


class TopNParams(BaseModel):
    sort_by: str = Field(description="column to order rows by")
    n: int = Field(default=10, ge=1, le=100, description="number of rows to keep")
    ascending: bool = Field(default=False, description="ascending order")
    columns: Optional[list[str]] = Field(
        default=None, description="subset of columns to return"
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("top_n", "Top-N rows ordered by a column (ranking)")
class TopNOp(TopNParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "TopNOp", step_id: int) -> OperationOutput:
        if params.sort_by not in df.columns:
            raise OperationError(f"unknown column '{params.sort_by}'")
        work = apply_filters(df, params.filters).reset_index(drop=True)

        sort_key = _numeric(work[params.sort_by])
        positions = sort_key.sort_values(
            ascending=params.ascending, na_position="last", kind="stable"
        ).index
        ordered = work.iloc[positions].head(params.n)

        if params.columns:
            missing = [c for c in params.columns if c not in df.columns]
            if missing:
                raise OperationError(f"unknown column(s): {missing}")
            ordered = ordered[params.columns]

        calc = Calculation(
            label=f"top {len(ordered)} by {params.sort_by}",
            value=_fmt_num(ordered.iloc[0][params.sort_by]) if len(ordered) else "-",
            detail=f"selected from {len(work):,} rows",
        )
        return OperationOutput(table=make_table(step_id, "top_n", ordered), calculations=[calc])


class CorrelationParams(BaseModel):
    columns: Optional[list[str]] = Field(
        default=None, description="numeric columns; defaults to all numeric columns"
    )
    method: Literal["pearson", "spearman"] = Field(default="pearson")
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("correlation", "Pairwise correlation matrix between continuous numeric columns")
class CorrelationOp(CorrelationParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "CorrelationOp", step_id: int) -> OperationOutput:
        work = apply_filters(df, params.filters)
        if params.columns:
            missing = [c for c in params.columns if c not in df.columns]
            if missing:
                raise OperationError(f"unknown column(s): {missing}")
            candidates = params.columns
        else:
            candidates = list(work.select_dtypes(include=["number"]).columns)
        numeric = work[candidates].apply(_numeric)

        binary_flags = [c for c in numeric.columns if numeric[c].dropna().nunique() <= 2]
        continuous = [
            c for c in numeric.columns
            if c not in binary_flags and numeric[c].nunique(dropna=True) > 1
        ]
        if not continuous:
            raise OperationError(
                "no continuous numeric columns to correlate. For binary/categorical "
                "pairs use the 'statistical_test' operation instead (auto-routes to "
                "group comparison or chi-square)."
            )

        corr = numeric[continuous].corr(method=params.method).round(4)
        matrix = corr.reset_index().rename(columns={"index": "column"})
        calcs: list[Calculation] = [
            Calculation(
                label="method",
                value=params.method,
                detail=f"computed over {len(numeric):,} rows per pair",
            )
        ]
        if binary_flags:
            calcs.append(
                Calculation(
                    label="excluded_binary_columns",
                    value=len(binary_flags),
                    detail=(
                        f"{binary_flags} are binary/constant; point-biserial coefficients "
                        "were withheld as they mislead — use statistical_test for group "
                        "mean comparisons instead"
                    ),
                )
            )
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                v = corr.iloc[i, j]
                if pd.notna(v):
                    pairs.append((abs(float(v)), cols[i], cols[j], float(v)))
        pairs.sort(reverse=True)
        if pairs:
            strength, a, b, v = pairs[0]
            direction = "positive" if v > 0 else "negative"
            calcs.append(
                Calculation(
                    label="strongest_correlation",
                    value=round(v, 6),
                    detail=f"{a} x {b}: {direction} (|r|={strength:.3f})",
                )
            )
        for strength, a, b, v in pairs[:3]:
            if strength >= 0.5:
                direction = "positive" if v > 0 else "negative"
                calcs.append(
                    Calculation(
                        label=f"{a} x {b}",
                        value=v,
                        detail=f"{direction} correlation ({strength:.2f} magnitude)",
                    )
                )
        return OperationOutput(table=make_table(step_id, "correlations", matrix), calculations=calcs)


class StatisticalTestParams(BaseModel):
    column_a: str = Field(description="first column")
    column_b: str = Field(description="second column")
    force: Optional[Literal["correlation", "correlation_spearman", "chi_square"]] = Field(
        default=None,
        description="override automatic test routing",
    )


@register_operation(
    "statistical_test",
    "Auto-routed significance test between two columns "
    "(binary x numeric -> Welch t-test / Mann-Whitney with group means; "
    "categorical x categorical -> chi-square + Cramer's V; "
    "continuous x continuous -> Pearson/Spearman)",
)
class StatisticalTestOp(StatisticalTestParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "StatisticalTestOp", step_id: int) -> OperationOutput:
        from insight.analytics.statistics import route_and_run

        for col in (params.column_a, params.column_b):
            if col not in df.columns:
                raise OperationError(f"unknown column '{col}'")

        work = df[[params.column_a, params.column_b]].copy()
        try:
            route, outcome = route_and_run(work, params.column_a, params.column_b, force=params.force)
        except ValueError as exc:
            raise OperationError(str(exc)) from exc

        calcs = [
            Calculation(label="test", value=outcome.test),
            Calculation(label="statistic", value=round(outcome.statistic, 6)),
            Calculation(label="p_value", value=(round(outcome.p_value, 6) if outcome.p_value is not None else None)),
            Calculation(
                label="effect_size",
                value=outcome.effect_size,
                detail=outcome.effect_label,
            ),
            Calculation(label="interpretation", value=outcome.interpretation),
        ]
        return OperationOutput(
            calculations=calcs,
            notes=(
                f"routed as '{route}'. {outcome.sample_note}. "
                f"assumptions: {', '.join(outcome.assumptions)}."
            ),
        )


class AnovaTestParams(BaseModel):
    group_column: str = Field(description="categorical column with 3+ groups")
    value_column: str = Field(description="numeric column to compare across groups")
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation(
    "anova_test",
    "One-way ANOVA comparing a numeric column across 3+ groups "
    "(F-test + eta-squared effect size + per-group summary table)",
)
class AnovaTestOp(AnovaTestParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "AnovaTestOp", step_id: int) -> OperationOutput:
        from insight.analytics.statistics import run_anova

        for col in (params.group_column, params.value_column):
            if col not in df.columns:
                raise OperationError(f"unknown column '{col}'")

        work = apply_filters(df, params.filters)
        try:
            outcome = run_anova(work, params.group_column, params.value_column)
        except ValueError as exc:
            raise OperationError(str(exc)) from exc

        summary = (
            work[[params.group_column, params.value_column]]
            .copy()
        )
        summary[params.value_column] = _numeric(summary[params.value_column])
        summary = summary.dropna()
        summary[params.group_column] = summary[params.group_column].astype(str)
        agg = (
            summary.groupby(params.group_column)[params.value_column]
            .agg(n="count", mean="mean", std="std")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        table = pd.DataFrame(
            {
                params.group_column: agg[params.group_column],
                "n": [int(v) for v in agg["n"]],
                "mean": [_scalar(v) for v in agg["mean"]],
                "std": [_scalar(v) for v in agg["std"]],
            }
        )
        calcs = [
            Calculation(label="test", value=outcome.test),
            Calculation(label="statistic", value=round(outcome.statistic, 6)),
            Calculation(label="p_value", value=(round(outcome.p_value, 6) if outcome.p_value is not None else None)),
            Calculation(
                label="effect_size",
                value=outcome.effect_size,
                detail=outcome.effect_label,
            ),
            Calculation(label="interpretation", value=outcome.interpretation),
        ]
        return OperationOutput(
            table=make_table(step_id, "group_summary", table),
            calculations=calcs,
            notes=(
                f"{outcome.sample_note}. assumptions: {', '.join(outcome.assumptions)}."
            ),
        )


class RegressionParams(BaseModel):
    target_column: str = Field(description="numeric outcome column to predict")
    feature_columns: list[str] = Field(
        min_length=1,
        description="numeric predictor columns (one or more)",
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation(
    "regression",
    "OLS linear regression of a numeric target on one or more numeric features "
    "(coefficients, standard errors, t-test p-values, R-squared, standardized betas)",
)
class RegressionOp(RegressionParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "RegressionOp", step_id: int) -> OperationOutput:
        for col in [params.target_column, *params.feature_columns]:
            if col not in df.columns:
                raise OperationError(f"unknown column '{col}'")

        work = apply_filters(df, params.filters)
        cols = [params.target_column, *params.feature_columns]
        total_rows = len(work)
        data = work[cols].apply(_numeric).dropna()
        n = len(data)
        k = len(params.feature_columns)
        if n < k + 2:
            raise OperationError(
                f"need at least {k + 2} complete rows for {k} feature(s); got {n}"
            )
        dropped = total_rows - n

        y = data[params.target_column].to_numpy(dtype=float)
        raw_x = data[params.feature_columns].to_numpy(dtype=float)
        design = np.column_stack([np.ones(n), raw_x])
        betas, _, _, _ = np.linalg.lstsq(design, y, rcond=None)

        fitted = design @ betas
        residuals = y - fitted
        rss = float((residuals ** 2).sum())
        tss = float(((y - y.mean()) ** 2).sum())
        r_squared = 1.0 - rss / tss if tss > 0 else 0.0
        dof = n - k - 1
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n - 1) / dof if dof > 0 else r_squared

        sigma2 = rss / dof if dof > 0 else 0.0
        xtx_inv = np.linalg.pinv(design.T @ design)
        cov = sigma2 * xtx_inv
        std_errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.where(std_errors > 0, betas / std_errors, np.nan)
        p_values = np.array([
            float(2.0 * _sps_stats.t.sf(abs(t), dof)) if np.isfinite(t) else None
            for t in t_stats
        ])

        y_std = float(y.std(ddof=1))
        std_betas = []
        for j in range(k):
            x_std = float(raw_x[:, j].std(ddof=1))
            std_betas.append(
                float(betas[j + 1] * x_std / y_std) if y_std > 0 and x_std > 0 else None
            )

        rows = [{"term": "intercept", "coefficient": _scalar(betas[0]),
                 "std_error": _scalar(std_errors[0]), "t_stat": _scalar(t_stats[0]),
                 "p_value": (_scalar(p_values[0]) if p_values[0] is not None else None),
                 "standardized_beta": None}]
        for j, feat in enumerate(params.feature_columns):
            rows.append({
                "term": feat,
                "coefficient": _scalar(betas[j + 1]),
                "std_error": _scalar(std_errors[j + 1]),
                "t_stat": _scalar(t_stats[j + 1]),
                "p_value": (_scalar(p_values[j + 1]) if p_values[j + 1] is not None else None),
                "standardized_beta": _scalar(std_betas[j]),
            })
        table = pd.DataFrame(rows, columns=[
            "term", "coefficient", "std_error", "t_stat", "p_value", "standardized_beta"
        ])

        strongest = max(
            range(k),
            key=lambda j: abs(std_betas[j]) if std_betas[j] is not None else -1.0,
            default=None,
        )
        calcs = [
            Calculation(label="r_squared", value=round(r_squared, 6),
                        detail=f"{params.target_column} ~ {' + '.join(params.feature_columns)}"),
            Calculation(label="adj_r_squared", value=round(adj_r_squared, 6)),
            Calculation(label="n_observations", value=int(n)),
            Calculation(
                label="rows_excluded_incomplete",
                value=int(dropped),
                detail=(
                    f"{dropped / total_rows:.1%} of rows dropped listwise for missing "
                    "values; estimates use complete cases only"
                ) if dropped else None,
            ),
        ]
        if strongest is not None and std_betas[strongest] is not None:
            calcs.append(Calculation(
                label="strongest_predictor",
                value=params.feature_columns[strongest],
                detail=f"standardized beta {_scalar(std_betas[strongest])}",
            ))
        significant = [
            params.feature_columns[j]
            for j in range(k)
            if p_values[j + 1] is not None and p_values[j + 1] < 0.05
        ]
        if significant:
            calcs.append(Calculation(
                label="significant_predictors_p_lt_0_05",
                value=", ".join(significant),
            ))
        return OperationOutput(
            table=make_table(step_id, "coefficients", table),
            calculations=calcs,
            notes=(
                f"OLS via least squares on {n:,} complete rows; assumes linearity "
                "and independent errors."
            ),
        )


class IsotonicCalibrationParams(BaseModel):
    score_column: str = Field(
        description="numeric score/probability column to audit"
    )
    outcome_column: str = Field(
        description="binary 0/1 outcome column the score should predict"
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation(
    "isotonic_calibration",
    "Audit how well a numeric score/probability tracks a binary outcome "
    "(isotonic regression via pool-adjacent-violators; reliability table, "
    "Brier score before/after recalibration, expected calibration error)",
)
class IsotonicCalibrationOp(IsotonicCalibrationParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "IsotonicCalibrationOp", step_id: int) -> OperationOutput:
        from insight.analytics.statistics import run_isotonic_calibration

        for col in (params.score_column, params.outcome_column):
            if col not in df.columns:
                raise OperationError(f"unknown column '{col}'")

        work = apply_filters(df, params.filters)
        try:
            report = run_isotonic_calibration(
                work, params.score_column, params.outcome_column
            )
        except ValueError as exc:
            raise OperationError(str(exc)) from exc

        calcs = [
            Calculation(label="test", value="isotonic calibration"),
            Calculation(
                label="brier_raw",
                value=round(report.brier_raw, 6),
                detail=f"mean squared error of raw {params.score_column}",
            ),
            Calculation(label="brier_calibrated", value=round(report.brier_calibrated, 6)),
            Calculation(label="ece", value=report.ece, detail="expected calibration error (10-bin)"),
            Calculation(label="base_rate", value=round(report.base_rate, 6)),
            Calculation(label="interpretation", value=report.interpretation),
        ]
        return OperationOutput(
            table=make_table(step_id, "reliability", report.reliability_table),
            calculations=calcs,
            notes=report.notes,
        )


class SqlQueryParams(BaseModel):
    query: str = Field(
        min_length=1,
        description=(
            "exactly one read-only DuckDB SQL statement (SELECT or WITH) over the "
            "registered tables — use for joins, window functions, pivots, text and "
            "date parsing that the fixed operations cannot express"
        ),
    )


@register_operation(
    "sql_query",
    "Run one read-only DuckDB SQL query over the registered table(s): multi-table "
    "JOINs, window functions, QUALIFY, PIVOT/UNPIVOT, regex/text and date functions. "
    "Only SELECT/WITH; no writes or file access.",
)
class SqlQueryOp(SqlQueryParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "SqlQueryOp", step_id: int, ctx=None) -> OperationOutput:
        if ctx is None:
            raise OperationError(
                "no SQL session is attached to this analysis (sql_query unavailable)"
            )
        try:
            frame = ctx.execute(params.query)
        except ValueError as exc:
            raise OperationError(str(exc)) from exc

        truncated = bool(getattr(ctx, "last_truncated", False))
        frame.columns = [str(c) for c in frame.columns]

        calcs = [
            Calculation(
                label="rows_returned",
                value=int(len(frame)),
                detail=f"capped at {ctx.max_rows:,} rows" + ("; TRUNCATED" if truncated else ""),
            ),
            Calculation(
                label="tables_available",
                value=", ".join(sorted(ctx.table_names.values())),
            ),
        ]
        return OperationOutput(
            table=make_table(step_id, "sql_result", frame),
            calculations=calcs,
            notes=(
                "executed by the embedded DuckDB engine in read-only mode"
                + ("; result set truncated to the row cap" if truncated else "")
            ),
        )


class DistributionParams(BaseModel):
    column: str = Field(description="numeric column")
    bins: Optional[int] = Field(
        default=None,
        ge=2,
        le=100,
        description="omit for adaptive bin count (Sturges + Freedman-Diaconis)",
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("distribution", "Histogram bin counts for a numeric column (adaptive bins)")
class DistributionOp(DistributionParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "DistributionOp", step_id: int) -> OperationOutput:
        if params.column not in df.columns:
            raise OperationError(f"unknown column '{params.column}'")
        work = apply_filters(df, params.filters)
        values = _numeric(work[params.column]).dropna()
        if values.empty:
            raise OperationError(f"no numeric values in '{params.column}'")

        k = _adaptive_bins(values, params.bins)
        counts_arr, edges_arr = np.histogram(values.values, bins=k)
        counts = [int(c) for c in counts_arr]
        edges = [float(e) for e in edges_arr]

        floor = max(5, int(np.ceil(len(values) * 0.05)))
        labels, counts, merged_away = _merge_sparse(counts, edges, floor)

        table = pd.DataFrame({"bin": labels, "count": counts})
        sparse_left = sum(1 for c in counts if c < 10)
        notes_parts = [f"{len(counts)} bins (adaptive from {k})"]
        if merged_away:
            notes_parts.append(
                f"{merged_away} sparse bins aggregated to keep every bucket >= {floor} rows"
            )
        calcs = [
            Calculation(label="mean", value=_scalar(values.mean())),
            Calculation(label="median", value=_scalar(values.median())),
            Calculation(label="min", value=_scalar(values.min())),
            Calculation(label="max", value=_scalar(values.max())),
            Calculation(label="excluded_missing", value=int(len(work) - len(values))),
        ]
        return OperationOutput(
            table=make_table(step_id, "distribution", table),
            calculations=calcs,
            notes="; ".join(notes_parts)
            + (f". {sparse_left} bins still have n<10." if sparse_left else ""),
        )


class OutliersParams(BaseModel):
    column: str = Field(description="numeric column to test")
    method: Literal["iqr", "zscore"] = Field(default="iqr")
    threshold: Optional[float] = Field(
        default=None,
        description="iqr multiplier (default 1.5) or z-score cutoff (default 3.0)",
    )
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("detect_outliers", "Flag outlier rows using IQR or z-score rule")
class OutliersOp(OutliersParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "OutliersOp", step_id: int) -> OperationOutput:
        if params.column not in df.columns:
            raise OperationError(f"unknown column '{params.column}'")
        work = apply_filters(df, params.filters)
        values = _numeric(work[params.column])
        valid = values.dropna()
        if valid.empty:
            raise OperationError(f"no numeric values in '{params.column}'")
        if params.method == "zscore":
            std = float(valid.std())
            if std == 0:
                raise OperationError("zero variance; z-score undefined")
            cutoff = params.threshold if params.threshold is not None else 3.0
            zscores = ((values - valid.mean()) / std).abs()
            mask = (zscores > cutoff).fillna(False)
            bounds_detail = f"mean={_fmt_num(valid.mean())}, std={_fmt_num(std)}, cutoff={cutoff}"
        else:
            multiplier = params.threshold if params.threshold is not None else 1.5
            q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
            mask = ((values < lower) | (values > upper)).fillna(False)
            bounds_detail = (
                f"IQR={_fmt_num(iqr)}, acceptable range [{_fmt_num(lower)}, {_fmt_num(upper)}]"
            )

        outliers = work[mask]
        preview_cols = list(outliers.columns)[:6]
        calcs = [
            Calculation(label="outlier_rows", value=len(outliers)),
            Calculation(
                label="outlier_share",
                value=round(len(outliers) / max(1, len(valid)), 4),
                detail=bounds_detail,
            ),
        ]
        if len(outliers):
            return OperationOutput(
                table=make_table(step_id, "outlier_examples", outliers[preview_cols]),
                calculations=calcs,
                notes=f"{len(outliers)} of {len(valid):,} rows flagged using {params.method}.",
            )
        return OperationOutput(calculations=calcs, notes="No outliers detected.")


_FREQ_CODES: dict[str, tuple[str, Callable[[pd.Period], str]]] = {
    "day": ("D", lambda p: p.strftime("%Y-%m-%d")),
    "week": ("W", lambda p: f"{p.year}-W{p.week:02d}" if hasattr(p, "week") else str(p)),
    "month": ("M", lambda p: p.strftime("%Y-%m")),
    "quarter": ("Q", lambda p: f"{p.year}-Q{p.quarter}"),
    "year": ("Y", lambda p: str(p.year)),
}


class TrendParams(BaseModel):
    date_column: str = Field(description="date/time column to bucket by")
    metric_column: Optional[str] = Field(
        default=None, description="numeric column to aggregate; defaults to row count"
    )
    freq: Literal["day", "week", "month", "quarter", "year"] = Field(default="month")
    agg: Literal["sum", "mean", "median", "count", "min", "max"] = Field(default="sum")
    filters: list[Filter] = Field(default_factory=list, description="optional row filters")


@register_operation("trend", "Aggregate a metric over time buckets")
class TrendOp(TrendParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "TrendOp", step_id: int) -> OperationOutput:
        if params.date_column not in df.columns:
            raise OperationError(f"unknown column '{params.date_column}'")
        work = apply_filters(df, params.filters)
        dates = pd.to_datetime(work[params.date_column], errors="coerce")
        valid_mask = dates.notna()
        work, dates = work[valid_mask], dates[valid_mask]
        if work.empty:
            raise OperationError(f"no parseable dates in '{params.date_column}'")

        code, formatter = _FREQ_CODES[params.freq]
        periods = dates.dt.to_period(code)

        if params.agg == "count" or params.metric_column is None:
            series = periods.groupby(periods).size()
            label = "row_count"
        else:
            if params.metric_column not in work.columns:
                raise OperationError(f"unknown column '{params.metric_column}'")
            values = _numeric(work[params.metric_column])
            series = values.groupby(periods.values).agg(params.agg)
            label = f"{params.agg}_{params.metric_column}"

        table = pd.DataFrame(
            {"period": [formatter(p) for p in series.index], label: [_scalar(v) for v in series.values]}
        )

        calcs: list[Calculation] = []
        if len(series) >= 2:
            first, last = float(series.iloc[0]), float(series.iloc[-1])
            change = None if first == 0 else round((last - first) / abs(first), 4)
            calcs.append(
                Calculation(
                    label="first_to_last_change",
                    value=change,
                    detail=(
                        f"{formatter(series.index[0])}={_fmt_num(first)} -> "
                        f"{formatter(series.index[-1])}={_fmt_num(last)}"
                    ),
                )
            )
        calcs.insert(
            0,
            Calculation(
                label="buckets",
                value=len(series),
                detail=f"{params.freq} buckets between {formatter(series.index[0])} and {formatter(series.index[-1])}",
            ),
        )
        return OperationOutput(table=make_table(step_id, "trend", table), calculations=calcs)


class CompareSubsetsParams(BaseModel):
    metric: MetricSpec = Field(description="single metric to compare, e.g. {'column':'revenue','agg':'sum'}")
    subset_a: list[Filter] = Field(description="filter defining subset A")
    subset_b: list[Filter] = Field(description="filter defining subset B")
    label_a: str = Field(default="subset_a")
    label_b: str = Field(default="subset_b")


@register_operation("compare_subsets", "Compute one metric over two disjoint subsets and compare")
class CompareSubsetsOp(CompareSubsetsParams):

    @staticmethod
    def _metric_scalar(frame: pd.DataFrame, metric: MetricSpec) -> float:
        if metric.agg == AggFunc.count:
            return float(len(frame))
        if not metric.column:
            raise OperationError("metric requires a column")
        if metric.column not in frame.columns:
            raise OperationError(f"unknown column '{metric.column}'")
        values = _numeric(frame[metric.column]).dropna()
        if values.empty:
            raise OperationError(f"no numeric values in '{metric.column}' within a subset")
        if metric.agg == AggFunc.nunique:
            return float(values.nunique())
        return float(getattr(values, metric.agg.value)())

    @staticmethod
    def _mean_ci(frame: pd.DataFrame, metric: MetricSpec) -> Optional[tuple[int, float, float, float]]:
        if metric.agg != AggFunc.mean or not metric.column or metric.column not in frame.columns:
            return None
        values = _numeric(frame[metric.column]).dropna()
        n = len(values)
        if n < 30:
            return None
        mean = float(values.mean())
        half = 1.96 * float(values.std(ddof=1)) / math.sqrt(n)
        return n, mean - half, mean + half, half

    @staticmethod
    def _run(df: pd.DataFrame, params: "CompareSubsetsOp", step_id: int) -> OperationOutput:
        a_frame = apply_filters(df, params.subset_a)
        b_frame = apply_filters(df, params.subset_b)
        if a_frame.empty or b_frame.empty:
            raise OperationError(
                "one of the subsets is empty; adjust the filters "
                f"({params.label_a}: {len(a_frame)} rows, {params.label_b}: {len(b_frame)} rows)"
            )

        value_a = CompareSubsetsOp._metric_scalar(a_frame, params.metric)
        value_b = CompareSubsetsOp._metric_scalar(b_frame, params.metric)
        delta = value_b - value_a
        change = None if value_a == 0 else round(delta / abs(value_a), 4)

        table = pd.DataFrame(
            {
                "subset": [params.label_a, params.label_b],
                "rows": [len(a_frame), len(b_frame)],
                "value": [_scalar(value_a), _scalar(value_b)],
            }
        )
        calcs = [
            Calculation(label=params.label_a, value=_scalar(value_a), detail=f"{len(a_frame):,} rows"),
            Calculation(label=params.label_b, value=_scalar(value_b), detail=f"{len(b_frame):,} rows"),
            Calculation(
                label="absolute_delta",
                value=_scalar(round(delta, 6)),
                detail=f"B relative to A: {change:+.1%}" if change is not None else None,
            ),
        ]

        ci_notes: list[str] = []
        for label, frame_subset in ((params.label_a, a_frame), (params.label_b, b_frame)):
            ci = CompareSubsetsOp._mean_ci(frame_subset, params.metric)
            if ci is None:
                continue
            n, low, high, half = ci
            calcs.append(Calculation(
                label=f"ci95_low_{label}",
                value=_scalar(round(low, 6)),
                detail=f"mean ± 1.96·SE (n={n:,})",
            ))
            calcs.append(Calculation(
                label=f"ci95_high_{label}",
                value=_scalar(round(high, 6)),
                detail=f"margin ±{_fmt_num(half)}",
            ))
            ci_notes.append(f"{label} 95% CI [{_fmt_num(low)}, {_fmt_num(high)}]")

        notes = None
        if ci_notes:
            notes = (
                "95% CIs use the normal approximation for means with n>=30 per subset: "
                + "; ".join(ci_notes)
                + "."
            )
        elif params.metric.agg == AggFunc.mean:
            notes = (
                "95% CIs omitted: normal approximation requires n>=30 per subset "
                f"({params.label_a}: {len(a_frame):,}, {params.label_b}: {len(b_frame):,})."
            )
        return OperationOutput(table=make_table(step_id, "comparison", table), calculations=calcs, notes=notes)


class FilterRowsParams(BaseModel):
    column: str = Field(description="column to filter on")
    op: FilterOperator = Field(description="comparison operator")
    value: Any = Field(default=None, description="value(s) to compare against; list for in/between")


@register_operation("filter_rows", "Keep only rows matching a condition", mutates=True)
class FilterRowsOp(FilterRowsParams):

    @staticmethod
    def _run(df: pd.DataFrame, params: "FilterRowsOp", step_id: int) -> OperationOutput:
        f = Filter(column=params.column, op=params.op, value=params.value)
        mask = _mask_for_filter(df, f)
        kept = int(mask.sum())
        calc = Calculation(
            label="rows_matching",
            value=kept,
            detail=f"removed {len(df) - kept:,} of {len(df):,} rows",
        )
        return OperationOutput(calculations=[calc], notes=None)

    @staticmethod
    def _mutate(df: pd.DataFrame, params: "FilterRowsOp") -> pd.DataFrame:
        f = Filter(column=params.column, op=params.op, value=params.value)
        return df[_mask_for_filter(df, f)]


_FILTER_SHAPE = "{column, op, value}[]"
_METRIC_SHAPE = "{'column': str, 'agg': 'count|nunique|sum|mean|median|min|max|std|var', 'alias'?: str}"

_NESTED_HINTS: dict[str, str] = {
    "MetricSpec": _METRIC_SHAPE,
    "Filter": _FILTER_SHAPE,
}


def _annotation_label(annotation: Any) -> str:
    if annotation is type(None):
        return "null"
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or origin is types.UnionType:
        return "|".join(_annotation_label(a) for a in args)
    if origin is Literal:
        return "|".join(str(a) for a in args)
    if origin is list:
        inner = _annotation_label(args[0]) if args else "any"
        return f"{inner}[]"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _NESTED_HINTS.get(annotation.__name__, annotation.__name__)
    if isinstance(annotation, type) and annotation in (str, int, float, bool):
        return annotation.__name__
    if annotation is Any:
        return "any"
    return getattr(annotation, "__name__", str(annotation))


def operation_catalog() -> str:
    lines: list[str] = ["Available operations:"]
    for spec in OPERATIONS.values():
        lines.append(f"- {spec.name}: {spec.description}")
        for fname, fld in spec.params_model.model_fields.items():
            label = _annotation_label(fld.annotation)
            if fld.is_required():
                required = "required"
            elif fld.default_factory is not None:
                required = f"default={fld.default_factory()!r}"
            else:
                required = f"default={fld.default!r}"
            desc = f" -- {fld.description}" if fld.description else ""
            lines.append(f"    * {fname} ({label}, {required}){desc}")
    return "\n".join(lines)


_COLUMN_REF_FIELDS: dict[str, set[str]] = {
    "filter_rows": {"column"},
    "group_aggregate": {"group_by"},
    "summarize": {"columns"},
    "value_counts": {"column"},
    "top_n": {"sort_by", "columns"},
    "correlation": {"columns"},
    "distribution": {"column"},
    "detect_outliers": {"column"},
    "trend": {"date_column", "metric_column"},
    "statistical_test": {"column_a", "column_b"},
    "anova_test": {"group_column", "value_column"},
    "regression": {"target_column", "feature_columns"},
    "isotonic_calibration": {"score_column", "outcome_column"},
    "sql_query": set(),
    "compare_subsets": set(),
}

TABLE_PRODUCING_OPS = frozenset(
    {
        "group_aggregate",
        "summarize",
        "value_counts",
        "top_n",
        "correlation",
        "distribution",
        "detect_outliers",
        "trend",
        "compare_subsets",
        "anova_test",
        "regression",
        "isotonic_calibration",
        "sql_query",
    }
)


def _canonical(profile: DatasetProfile, name: Any) -> tuple[str | None, str | None]:
    if not isinstance(name, str):
        return None, f"column reference must be a string, got: {name!r}"
    canonical = profile.canonical_column(name.strip())
    if canonical is None:
        return None, (
            f"unknown column '{name}'. available columns: "
            f"{profile.column_names()}"
        )
    return canonical, None


def validate_plan_columns(
    plan: AnalysisPlan, profile: DatasetProfile
) -> tuple[AnalysisPlan, list[str]]:
    issues: list[str] = []

    def canonical_or_issue(field_label: str, value: Any) -> str | None:
        if not isinstance(value, str):
            issues.append(
                f"{field_label}: column reference must be a string, got {value!r}"
            )
            return None
        canonical = profile.canonical_column(value.strip())
        if canonical is None:
            issues.append(
                f"{field_label}: unknown column '{value}'. "
                f"available columns: {profile.column_names()}"
            )
            return None
        return canonical

    for step in plan.steps:
        params = dict(step.params)
        ref_fields = _COLUMN_REF_FIELDS.get(step.operation)

        if ref_fields is None:
            issues.append(
                f"step {step.step_id}: unknown operation '{step.operation}'. "
                f"available: {sorted(OPERATIONS)}"
            )
            continue

        if step.input_step is not None:
            if step.input_step == step.step_id:
                issues.append(
                    f"step {step.step_id}: input_step cannot reference itself"
                )
            elif step.input_step not in {s.step_id for s in plan.steps}:
                issues.append(
                    f"step {step.step_id}: input_step references unknown "
                    f"step {step.input_step}"
                )
            else:
                target = next(
                    (s for s in plan.steps if s.step_id == step.input_step), None
                )
                if target is not None and plan.steps.index(target) > plan.steps.index(step):
                    issues.append(
                        f"step {step.step_id}: input_step {step.input_step} comes "
                        f"later in execution order (DAG must be acyclic)"
                    )
                if target is not None and target.operation not in TABLE_PRODUCING_OPS:
                    issues.append(
                        f"step {step.step_id}: input_step target ({target.operation}) "
                        f"produces no table; choose one of {sorted(TABLE_PRODUCING_OPS)}"
                    )

        def check_filters(filters: Any) -> None:
            if not isinstance(filters, list):
                return
            for entry in filters:
                if not isinstance(entry, dict) or "column" not in entry:
                    continue
                canonical = canonical_or_issue(
                    f"step {step.step_id} filter", entry["column"]
                )
                if canonical:
                    entry["column"] = canonical

        for field_name in ref_fields:
            if field_name not in params or params[field_name] is None:
                continue
            raw = params[field_name]
            label = f"step {step.step_id} '{field_name}'"
            if isinstance(raw, list):
                fixed = []
                for i, item in enumerate(raw):
                    canonical = canonical_or_issue(f"{label}[{i}]", item)
                    fixed.append(canonical if canonical else item)
                params[field_name] = fixed
            else:
                canonical = canonical_or_issue(label, raw)
                if canonical:
                    params[field_name] = canonical

        for key, value in params.items():
            if key == "filters" or key.startswith("subset_"):
                check_filters(value)

        if step.operation == "group_aggregate":
            metrics = params.get("metrics")
            if isinstance(metrics, list):
                for i, metric in enumerate(metrics):
                    if not isinstance(metric, dict):
                        continue
                    agg = str(metric.get("agg", "count"))
                    column = metric.get("column")
                    if agg != "count":
                        canonical = canonical_or_issue(
                            f"step {step.step_id} metrics[{i}]", column
                        )
                        if canonical:
                            metric["column"] = canonical

        if step.operation == "compare_subsets":
            metric = params.get("metric")
            if isinstance(metric, dict) and str(metric.get("agg", "")) != "count":
                canonical = canonical_or_issue(
                    f"step {step.step_id} metric", metric.get("column")
                )
                if canonical:
                    metric["column"] = canonical

        step.params = params

    valid_charts = {s.step_id for s in plan.steps}
    for chart in plan.charts:
        if chart.source_step not in valid_charts:
            issues.append(
                f"chart '{chart.title}' references source_step {chart.source_step} "
                f"which does not exist (valid steps: {sorted(valid_charts)})"
            )

    return plan, issues
