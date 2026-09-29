import numpy as np
import pandas as pd
import pytest

from insight.analytics.operations import OPERATIONS, OperationError
from insight.validation.grounding import find_ungrounded_numbers


@pytest.fixture
def messy_df():
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "region": ["North", "South", "East", "West"] * (n // 4),
        "category": ["A", "B", "C"] * (n // 3),
        "revenue": [100 + ((i * 37) % 200) for i in range(n)],
        "units": [(i % 5) + 1 for i in range(n)],
        "treated": rng.integers(0, 2, n),
    })
    corrupt = rng.random(n)
    df.loc[corrupt < 0.30, "revenue"] = np.nan
    df.loc[corrupt > 0.85, "category"] = None
    df.loc[(corrupt > 0.4) & (corrupt < 0.45), "date"] = None
    return df


OPS_UNDER_NULLS = [
    ("summarize", {"columns": ["revenue"]}),
    ("value_counts", {"column": "category"}),
    ("value_counts", {"column": "category", "normalize": True}),
    ("top_n", {"sort_by": "revenue", "n": 5}),
    ("group_aggregate", {
        "group_by": "region",
        "metrics": [
            {"column": "revenue", "agg": "count"},
            {"column": "revenue", "agg": "mean", "alias": "avg_rev"},
        ],
    }),
    ("distribution", {"column": "revenue"}),
    ("detect_outliers", {"column": "revenue"}),
    ("trend", {"date_column": "date", "metric_column": "revenue", "freq": "month"}),
    ("correlation", {"columns": ["revenue", "units"]}),
    ("compare_subsets", {
        "metric": {"column": "revenue", "agg": "mean"},
        "subset_a": [{"column": "region", "op": "eq", "value": "North"}],
        "subset_b": [{"column": "region", "op": "eq", "value": "South"}],
    }),
    ("anova_test", {"group_column": "region", "value_column": "revenue"}),
    ("regression", {"target_column": "units", "feature_columns": ["revenue"]}),
    ("isotonic_calibration", {"score_column": "revenue", "outcome_column": "treated"}),
    ("filter_rows", {"column": "region", "op": "eq", "value": "North"}),
]


@pytest.mark.parametrize("operation,params", OPS_UNDER_NULLS)
def test_ops_survive_30_percent_missingness(messy_df, operation, params):
    spec = OPERATIONS[operation]
    try:
        output = spec.run(messy_df, spec.params_model(**params), step_id=0)
    except OperationError:
        return
    except Exception as exc:
        pytest.fail(f"{operation} crashed on missing data: {type(exc).__name__}: {exc}")

    if output.table is not None:
        for row in output.table.rows:
            for cell in row:
                if isinstance(cell, float):
                    assert np.isfinite(cell), f"{operation} emitted non-finite value"


def test_normalize_invariant_under_missingness():
    rng = np.random.default_rng(11)
    values = rng.choice(["a", "b", "c"], 500)
    values[rng.random(500) < 0.4] = None
    df = pd.DataFrame({"col": values})
    out = OPERATIONS["value_counts"].run(
        df, OPERATIONS["value_counts"].params_model(column="col", normalize=True), step_id=0
    )
    table = out.table.to_dataframe()
    props = [float(v) for v in table["proportion"]]
    assert abs(sum(props) - 1.0) < 1e-6


class _Calc:
    def __init__(self, label, value=None, detail=None):
        self.label, self.value, self.detail = label, value, detail


class _Table:
    def __init__(self, columns, rows):
        self.columns, self.rows = columns, rows


def test_grounding_flags_fabricated_totals():
    calcs = [_Calc("groups", 4)]
    table = _Table(["region", "total_revenue"], [["West", 48805.62], ["East", 33774.76]])
    violations = find_ungrounded_numbers(
        "Based on **13,650 records**, West leads with 48805.62.",
        calcs, [table],
    )
    assert violations == ["13,650"] or violations == ["13650"] or any("650" in v for v in violations)


def test_grounding_accepts_verbatim_percentages_and_years():
    calcs = [_Calc("mean", 285.31)]
    table = _Table(["year", "count"], [[2024, 150], [2025, 330]])
    answer = "Mean is 285.31 (i.e. about 28531%). Top year: 2024 with count 330; rank 1."
    assert find_ungrounded_numbers(answer, calcs, [table]) == []


def test_grounding_accepts_comma_variants():
    table = _Table(["region", "total"], [["W", 13221]])
    assert find_ungrounded_numbers("Total was 13,221.", [], [table]) == []
