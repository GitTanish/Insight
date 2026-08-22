import pandas as pd
import pytest

from insight.analytics.operations import (
    OPERATIONS,
    OperationError,
    apply_filters,
    validate_plan_columns,
)
from insight.domain.analysis import Filter, FilterOperator


def test_group_aggregate_matches_pandas(sales_df):
    out = OPERATIONS["group_aggregate"].run(
        sales_df,
        OPERATIONS["group_aggregate"].params_model(
            group_by="region",
            metrics=[{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
            sort_by="total_revenue",
        ),
        step_id=0,
    )
    table = out.table.to_dataframe()
    expected = sales_df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    assert list(table.columns) == ["region", "total_revenue"]
    assert list(table["region"]) == list(expected.index)
    assert table.iloc[0]["total_revenue"] == float(expected.iloc[0])


def test_group_count_not_truncated_by_top_n(sales_df):
    params = OPERATIONS["group_aggregate"].params_model(
        group_by="region",
        metrics=[{"column": "revenue", "agg": "sum"}],
        top_n=1,
    )
    out = OPERATIONS["group_aggregate"].run(sales_df, params, step_id=0)
    groups_calc = next(c for c in out.calculations if c.label == "groups")
    assert groups_calc.value == 4
    assert len(out.table.rows) == 1


def test_filters_case_insensitive_eq(sales_df):
    filtered = apply_filters(sales_df, [Filter(column="region", op=FilterOperator.eq, value="north")])
    assert len(filtered) == 15
    assert set(filtered["region"]) == {"North"}


def test_filter_between_and_in():
    import pandas as pd

    df = pd.DataFrame({"v": [5, 10, 15, 20], "g": ["a", "b", "c", "d"]})
    between = apply_filters(df, [Filter(column="v", op=FilterOperator.between, value=[10, 15])])
    assert sorted(between["v"]) == [10, 15]
    inside = apply_filters(df, [Filter(column="g", op=FilterOperator.in_, value=["B", "C"])])
    assert sorted(inside["g"]) == ["b", "c"]
    contains = apply_filters(df, [Filter(column="g", op=FilterOperator.contains, value="A")])
    assert list(contains["g"]) == ["a"]


def test_unknown_column_raises(sales_df):
    with pytest.raises(OperationError):
        OPERATIONS["group_aggregate"].run(
            sales_df,
            OPERATIONS["group_aggregate"].params_model(
                group_by="nope", metrics=[{"column": "revenue", "agg": "sum"}]
            ),
            step_id=0,
        )


def test_correlation_matrix(sales_df):
    out = OPERATIONS["correlation"].run(
        sales_df,
        OPERATIONS["correlation"].params_model(columns=["revenue", "units"]),
        step_id=0,
    )
    table = out.table.to_dataframe()
    diag = table[table["column"] == "revenue"]["revenue"].iloc[0]
    assert abs(diag - 1.0) < 1e-6
    assert any(c.label == "method" for c in out.calculations)


def test_correlation_reports_strong_pairs():
    import pandas as pd

    df = pd.DataFrame({
        "a": [1.0 * i for i in range(20)],
        "b": [2.0 * i + 0.01 * ((i * 7) % 3) for i in range(20)],
    })
    out = OPERATIONS["correlation"].run(
        df, OPERATIONS["correlation"].params_model(columns=["a", "b"]), step_id=0,
    )
    pair = next(c for c in out.calculations if c.label == "a x b")
    assert abs(pair.value - 1.0) < 0.01


def test_distribution_bins_sum(sales_df):
    out = OPERATIONS["distribution"].run(
        sales_df,
        OPERATIONS["distribution"].params_model(column="revenue", bins=5),
        step_id=0,
    )
    counts = [r[1] for r in out.table.rows]
    assert sum(counts) == len(sales_df)


def test_detect_outliers_flags_injected(sales_df):
    spiked = sales_df.copy()
    spiked.loc[0, "revenue"] = 100000.0
    out = OPERATIONS["detect_outliers"].run(
        spiked, OPERATIONS["detect_outliers"].params_model(column="revenue"), step_id=0
    )
    outlier_rows = next(c for c in out.calculations if c.label == "outlier_rows")
    assert outlier_rows.value == 1
    assert out.table is not None


def test_outliers_zero_variance_errors():
    import pandas as pd

    df = pd.DataFrame({"v": [7.0] * 10})
    with pytest.raises(OperationError):
        OPERATIONS["detect_outliers"].run(
            df, OPERATIONS["detect_outliers"].params_model(column="v", method="zscore"),
            step_id=0,
        )


def test_trend_monthly_sums_match_manual(sales_df):
    out = OPERATIONS["trend"].run(
        sales_df,
        OPERATIONS["trend"].params_model(date_column="date", metric_column="revenue", freq="month"),
        step_id=0,
    )
    table = out.table.to_dataframe()
    manual = (
        sales_df.assign(d=pd.to_datetime(sales_df["date"]))
        .assign(p=lambda frame: frame["d"].dt.to_period("M"))
        .groupby("p")["revenue"].sum()
    )
    assert list(table.columns) == ["period", "sum_revenue"]
    assert list(table["sum_revenue"]) == [float(v) for v in manual.values]


def test_compare_subsets_delta(sales_df):
    out = OPERATIONS["compare_subsets"].run(
        sales_df,
        OPERATIONS["compare_subsets"].params_model(
            metric={"column": "revenue", "agg": "mean"},
            subset_a=[{"column": "region", "op": "eq", "value": "North"}],
            subset_b=[{"column": "region", "op": "eq", "value": "South"}],
            label_a="N",
            label_b="S",
        ),
        step_id=0,
    )
    means = {
        row[0]: row[2]
        for row in out.table.rows
    }
    expected_n = sales_df.loc[sales_df.region == "North", "revenue"].mean()
    assert abs(means["N"] - expected_n) < 1e-6
    delta = next(c for c in out.calculations if c.label == "absolute_delta")
    assert delta.detail is not None


def test_value_counts_normalize(sales_df):
    out = OPERATIONS["value_counts"].run(
        sales_df,
        OPERATIONS["value_counts"].params_model(column="category", normalize=True),
        step_id=0,
    )
    props = [r[2] for r in out.table.rows]
    assert all(abs(p - round(20 / 60, 4)) < 1e-6 for p in props)


def test_validate_plan_columns_canonicalizes(sales_df, profile):
    from insight.domain.analysis import AnalysisPlan, PlanStep

    plan = AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=0, operation="group_aggregate", params={
                "group_by": " REGION ",
                "metrics": [{"column": "Revenue", "agg": "mean"}],
            }),
            PlanStep(step_id=1, operation="made_up_op", params={}),
        ],
        charts=[],
    )
    fixed, issues = validate_plan_columns(plan, profile)
    assert fixed.steps[0].params["group_by"] == "region"
    assert fixed.steps[0].params["metrics"][0]["column"] == "revenue"
    assert any("unknown operation" in i for i in issues)


def test_chart_source_step_validated(sales_df, profile):
    from insight.domain.analysis import AnalysisPlan, ChartSpec, ChartType, PlanStep

    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="value_counts", params={"column": "region"})],
        charts=[ChartSpec(chart_type=ChartType.bar, title="x", source_step=3)],
    )
    _, issues = validate_plan_columns(plan, profile)
    assert any("source_step" in i for i in issues)
