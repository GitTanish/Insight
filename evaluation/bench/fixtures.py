"""Deterministic, self-contained fixtures for the adversarial verifier matrix.

The fixtures intentionally cover both simple cases and the contexts the verifier
claims to support: filters, missing values, multi-column grouping, top-N
ordering, ``other_after`` folding, projected columns, and statistical results.
No fixture reads evaluation data from disk, so the matrix is reproducible from
a clean checkout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from insight.domain.analysis import AnalysisPlan, PlanStep

DEFAULT_SEED = 20261101
DEFAULT_ROWS = 720


@dataclass(frozen=True)
class NumericTarget:
    column: str
    bounds: tuple[float, float] | None = None
    expectation: str = "detect"


@dataclass(frozen=True)
class CountTarget:
    column: str
    expectation: str = "detect"


@dataclass(frozen=True)
class CalculationTarget:
    label: str
    bounds: tuple[float, float] | None = None
    expectation: str = "detect"


@dataclass(frozen=True)
class CorrelationTarget:
    row: str
    column: str
    symmetric_pair: bool = False
    expectation: str = "detect"


@dataclass(frozen=True)
class Fixture:
    fixture_id: str
    operation: str
    description: str
    frame: pd.DataFrame
    plan: AnalysisPlan
    tags: tuple[str, ...]
    numeric_targets: tuple[NumericTarget, ...] = ()
    count_targets: tuple[CountTarget, ...] = ()
    calculation_targets: tuple[CalculationTarget, ...] = ()
    correlation_targets: tuple[CorrelationTarget, ...] = ()

    def targets_for(self, kind: str) -> list:
        return list(getattr(self, f"{kind}_targets"))


def make_frame(seed: int = DEFAULT_SEED, n: int = DEFAULT_ROWS) -> pd.DataFrame:
    """Synthetic business frame with deterministic structure and missingness."""
    rng = np.random.default_rng(seed)

    region = rng.choice(
        np.array(["North", "South", "East", "West"]),
        size=n,
        p=[0.31, 0.27, 0.23, 0.19],
    )
    category = rng.choice(
        np.array(["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot"]),
        size=n,
        p=[0.26, 0.22, 0.18, 0.14, 0.11, 0.09],
    )
    channel = rng.choice(
        np.array(["web", "mobile", "partner", "store"]),
        size=n,
        p=[0.34, 0.27, 0.22, 0.17],
    )
    flag = rng.integers(0, 2, size=n)
    revenue = np.round(rng.lognormal(mean=6.05, sigma=0.62, size=n), 2)
    units = rng.integers(1, 31, size=n)
    discount = np.round(rng.beta(2.0, 9.0, size=n), 4)
    score = np.round(100.0 + 0.025 * revenue + rng.normal(0.0, 5.5, size=n), 2)
    x1 = rng.normal(0.0, 1.3, size=n)
    x2 = rng.normal(0.0, 0.9, size=n)
    y = 4.0 + 1.7 * x1 - 0.8 * x2 + rng.normal(0.0, 0.7, size=n)

    group3 = rng.choice(np.array(["low", "mid", "high"]), size=n, p=[0.34, 0.33, 0.33])
    group4 = rng.choice(
        np.array(["alpha", "beta", "gamma", "delta"]),
        size=n,
        p=[0.29, 0.27, 0.24, 0.20],
    )
    group_effect = np.select(
        [group3 == "low", group3 == "mid"],
        [0.0, 2.5],
        default=5.0,
    )
    value_anova = group_effect + rng.normal(0.0, 2.2, size=n)

    frame = pd.DataFrame(
        {
            "region": region,
            "category": category,
            "channel": channel,
            "flag": flag,
            "revenue": revenue,
            "units": units,
            "discount": discount,
            "score": score,
            "x1": x1,
            "x2": x2,
            "y": y,
            "group3": group3,
            "group4": group4,
            "value_anova": value_anova,
        }
    )

    # Deterministic missingness exercises listwise deletion paths.
    frame.loc[frame.index % 29 == 0, "x1"] = np.nan
    frame.loc[frame.index % 31 == 0, "y"] = np.nan
    frame.loc[frame.index % 17 == 0, "discount"] = np.nan
    return frame


def _plan(objective: str, operation: str, params: dict) -> AnalysisPlan:
    return AnalysisPlan(
        objective=objective,
        steps=[PlanStep(step_id=0, operation=operation, params=params)],
        charts=[],
    )


def build_fixtures(seed: int = DEFAULT_SEED, n: int = DEFAULT_ROWS) -> list[Fixture]:
    frame = make_frame(seed=seed, n=n)
    return [
        Fixture(
            fixture_id="group_aggregate_simple",
            operation="group_aggregate",
            description="Single-key sums and means over the full frame.",
            frame=frame,
            plan=_plan(
                "aggregate revenue by region",
                "group_aggregate",
                {
                    "group_by": "region",
                    "metrics": [
                        {"column": "revenue", "agg": "sum", "alias": "total_revenue"},
                        {"column": "units", "agg": "mean", "alias": "avg_units"},
                    ],
                    "sort_by": "total_revenue",
                },
            ),
            tags=("baseline", "single-key"),
            numeric_targets=(
                NumericTarget("total_revenue"),
                NumericTarget("avg_units"),
            ),
        ),
        Fixture(
            fixture_id="group_aggregate_filtered",
            operation="group_aggregate",
            description="Filtered group aggregation with a count metric.",
            frame=frame,
            plan=_plan(
                "aggregate flagged revenue by category",
                "group_aggregate",
                {
                    "group_by": "category",
                    "metrics": [
                        {"column": "revenue", "agg": "sum", "alias": "revenue_total"},
                        {"agg": "count", "alias": "n"},
                    ],
                    "sort_by": "revenue_total",
                    "filters": [{"column": "flag", "op": "eq", "value": 1}],
                },
            ),
            tags=("filtered", "count-metric"),
            numeric_targets=(NumericTarget("revenue_total"),),
            count_targets=(CountTarget("n"),),
        ),
        Fixture(
            fixture_id="group_aggregate_multikey",
            operation="group_aggregate",
            description="Two grouping keys, sort, and top-N projection.",
            frame=frame,
            plan=_plan(
                "aggregate revenue by region and channel",
                "group_aggregate",
                {
                    "group_by": ["region", "channel"],
                    "metrics": [
                        {"column": "revenue", "agg": "mean", "alias": "avg_revenue"}
                    ],
                    "sort_by": "avg_revenue",
                    "top_n": 8,
                },
            ),
            tags=("multi-key", "top_n"),
            numeric_targets=(NumericTarget("avg_revenue"),),
        ),
        Fixture(
            fixture_id="group_aggregate_other",
            operation="group_aggregate",
            description="Rare levels folded into an Other bucket.",
            frame=frame,
            plan=_plan(
                "aggregate revenue by category with tail folding",
                "group_aggregate",
                {
                    "group_by": "category",
                    "metrics": [
                        {"column": "revenue", "agg": "sum", "alias": "sum_revenue"}
                    ],
                    "other_after": 4,
                },
            ),
            tags=("other_after",),
            numeric_targets=(NumericTarget("sum_revenue"),),
        ),
        Fixture(
            fixture_id="top_n_all",
            operation="top_n",
            description="Top rows by revenue with all source columns.",
            frame=frame,
            plan=_plan(
                "top revenue rows",
                "top_n",
                {"sort_by": "revenue", "n": 10},
            ),
            tags=("baseline",),
            numeric_targets=(NumericTarget("revenue"),),
        ),
        Fixture(
            fixture_id="top_n_projected_filtered",
            operation="top_n",
            description="Filtered top-N with a projected column set.",
            frame=frame,
            plan=_plan(
                "top flagged revenue rows",
                "top_n",
                {
                    "sort_by": "revenue",
                    "n": 5,
                    "columns": ["region", "revenue"],
                    "filters": [{"column": "flag", "op": "eq", "value": 1}],
                },
            ),
            tags=("filtered", "projected"),
            numeric_targets=(NumericTarget("revenue"),),
        ),
        Fixture(
            fixture_id="value_counts_simple",
            operation="value_counts",
            description="Category frequencies with normalized proportions.",
            frame=frame,
            plan=_plan(
                "category frequency",
                "value_counts",
                {"column": "category", "normalize": True},
            ),
            tags=("baseline", "proportion"),
            numeric_targets=(NumericTarget("proportion", bounds=(0.0, 1.0)),),
            count_targets=(CountTarget("count"),),
        ),
        Fixture(
            fixture_id="value_counts_other",
            operation="value_counts",
            description="Region frequencies with tail folding.",
            frame=frame,
            plan=_plan(
                "region frequency with Other",
                "value_counts",
                {
                    "column": "region",
                    "other_after": 3,
                    "top_n": 5,
                    "normalize": True,
                },
            ),
            tags=("other_after", "proportion"),
            numeric_targets=(NumericTarget("proportion", bounds=(0.0, 1.0)),),
            count_targets=(CountTarget("count"),),
        ),
        Fixture(
            fixture_id="value_counts_filtered",
            operation="value_counts",
            description="Filtered channel frequencies without normalization.",
            frame=frame,
            plan=_plan(
                "flagged channel frequency",
                "value_counts",
                {
                    "column": "channel",
                    "filters": [{"column": "flag", "op": "eq", "value": 1}],
                },
            ),
            tags=("filtered",),
            count_targets=(CountTarget("count"),),
        ),
        Fixture(
            fixture_id="correlation_matrix",
            operation="correlation",
            description="Four-column Pearson matrix with missing pairwise values.",
            frame=frame,
            plan=_plan(
                "correlate commercial metrics",
                "correlation",
                {"columns": ["revenue", "units", "discount", "score"]},
            ),
            tags=("baseline", "matrix"),
            correlation_targets=(
                CorrelationTarget("revenue", "score", symmetric_pair=False),
                CorrelationTarget(
                    "revenue", "score", symmetric_pair=True, expectation="known_gap"
                ),
                CorrelationTarget("revenue", "revenue", symmetric_pair=False),
            ),
            calculation_targets=(
                CalculationTarget(
                    "strongest_correlation",
                    bounds=(-1.0, 1.0),
                    expectation="known_gap",
                ),
            ),
        ),
        Fixture(
            fixture_id="compare_subsets_mean",
            operation="compare_subsets",
            description="Mean revenue comparison with confidence intervals.",
            frame=frame,
            plan=_plan(
                "compare North and South revenue",
                "compare_subsets",
                {
                    "metric": {"column": "revenue", "agg": "mean"},
                    "subset_a": [{"column": "region", "op": "eq", "value": "North"}],
                    "subset_b": [{"column": "region", "op": "eq", "value": "South"}],
                    "label_a": "North",
                    "label_b": "South",
                },
            ),
            tags=("baseline", "confidence_interval"),
            numeric_targets=(NumericTarget("value"),),
            count_targets=(CountTarget("rows"),),
            calculation_targets=(
                CalculationTarget("absolute_delta"),
                CalculationTarget("ci95_low_North"),
                CalculationTarget("ci95_high_South"),
            ),
        ),
        Fixture(
            fixture_id="compare_subsets_sum_filtered",
            operation="compare_subsets",
            description="Filtered sum comparison without confidence intervals.",
            frame=frame,
            plan=_plan(
                "compare revenue for flagged and unflagged rows",
                "compare_subsets",
                {
                    "metric": {"column": "revenue", "agg": "sum"},
                    "subset_a": [{"column": "flag", "op": "eq", "value": 0}],
                    "subset_b": [{"column": "flag", "op": "eq", "value": 1}],
                    "label_a": "Control",
                    "label_b": "Treatment",
                },
            ),
            tags=("filtered", "sum"),
            numeric_targets=(NumericTarget("value"),),
            count_targets=(CountTarget("rows"),),
            calculation_targets=(CalculationTarget("absolute_delta"),),
        ),
        Fixture(
            fixture_id="anova_three_groups",
            operation="anova_test",
            description="Three-group ANOVA with full group summaries.",
            frame=frame,
            plan=_plan(
                "compare value_anova across group3",
                "anova_test",
                {"group_column": "group3", "value_column": "value_anova"},
            ),
            tags=("baseline", "three_groups"),
            numeric_targets=(
                NumericTarget("mean"),
                NumericTarget("std"),
            ),
            count_targets=(CountTarget("n"),),
            calculation_targets=(
                CalculationTarget("statistic"),
                CalculationTarget("p_value", bounds=(0.0, 1.0)),
                CalculationTarget("effect_size", bounds=(0.0, 1.0)),
            ),
        ),
        Fixture(
            fixture_id="anova_filtered_missing",
            operation="anova_test",
            description="Filtered four-group ANOVA with listwise missingness.",
            frame=frame,
            plan=_plan(
                "compare y across group4 for flagged rows",
                "anova_test",
                {
                    "group_column": "group4",
                    "value_column": "y",
                    "filters": [{"column": "flag", "op": "eq", "value": 1}],
                },
            ),
            tags=("filtered", "missing", "four_groups"),
            numeric_targets=(NumericTarget("mean"),),
            count_targets=(CountTarget("n"),),
            calculation_targets=(
                CalculationTarget("statistic"),
                CalculationTarget("p_value", bounds=(0.0, 1.0)),
                CalculationTarget("effect_size", bounds=(0.0, 1.0)),
            ),
        ),
        Fixture(
            fixture_id="regression_single",
            operation="regression",
            description="Single-feature OLS with incomplete rows excluded.",
            frame=frame,
            plan=_plan(
                "regress y on x1",
                "regression",
                {"target_column": "y", "feature_columns": ["x1"]},
            ),
            tags=("baseline", "missing"),
            numeric_targets=(
                NumericTarget("coefficient"),
                NumericTarget("std_error"),
                NumericTarget("p_value", bounds=(0.0, 1.0)),
                NumericTarget("standardized_beta"),
            ),
            calculation_targets=(
                CalculationTarget("r_squared", bounds=(0.0, 1.0)),
                CalculationTarget("adj_r_squared", bounds=(0.0, 1.0)),
                CalculationTarget("n_observations"),
                CalculationTarget("rows_excluded_incomplete"),
            ),
        ),
        Fixture(
            fixture_id="regression_two_filtered",
            operation="regression",
            description="Two-feature filtered OLS.",
            frame=frame,
            plan=_plan(
                "regress y on x1 and x2 for flagged rows",
                "regression",
                {
                    "target_column": "y",
                    "feature_columns": ["x1", "x2"],
                    "filters": [{"column": "flag", "op": "eq", "value": 1}],
                },
            ),
            tags=("filtered", "two_features"),
            numeric_targets=(
                NumericTarget("coefficient"),
                NumericTarget("p_value", bounds=(0.0, 1.0)),
            ),
            calculation_targets=(
                CalculationTarget("r_squared", bounds=(0.0, 1.0)),
                CalculationTarget("n_observations"),
            ),
        ),
    ]


def quick_fixture_ids() -> set[str]:
    return {
        "group_aggregate_simple",
        "top_n_all",
        "value_counts_simple",
        "correlation_matrix",
        "compare_subsets_mean",
        "anova_three_groups",
        "regression_single",
    }
