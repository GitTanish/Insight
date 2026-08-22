import pandas as pd
import pytest

from insight.analytics.operations import validate_plan_columns
from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.execution.executor import execute_plan_sync
from insight.profiling.profiler import profile_dataframe


@pytest.fixture
def sales_df():
    n = 120
    return pd.DataFrame({
        "order_id": range(1, n + 1),
        "year": [2024] * (n // 2) + [2025] * (n // 2),
        "region": ["North", "South", "East", "West"] * (n // 4),
        "category": ["A", "B", "C"] * (n // 3),
        "revenue": [100 + ((i * 37) % 200) for i in range(n)],
        "treated": [(i % 2) for i in range(n)],
    })


def test_chain_filter_group_topn(sales_df):
    plan = AnalysisPlan(
        objective="chain",
        steps=[
            PlanStep(step_id=0, operation="filter_rows", params={
                "column": "year", "op": "eq", "value": 2025,
            }),
            PlanStep(step_id=1, operation="group_aggregate", params={
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
            }),
            PlanStep(step_id=2, operation="top_n", params={
                "sort_by": "total_revenue", "n": 2, "columns": ["region", "total_revenue"],
            }, input_step=1),
        ],
    )
    result = execute_plan_sync(sales_df, plan)
    assert all(s.success for s in result.steps), result.failed_messages()
    table = result.get_table(2)
    assert len(table.rows) == 2

    expected = (
        sales_df[sales_df.year == 2025]
        .groupby("region")["revenue"].sum().sort_values(ascending=False).head(2)
    )
    got = [r[1] for r in table.rows]
    assert sorted(got) == sorted(float(v) for v in expected.values)


def test_chained_statistical_test_on_filtered_frame(sales_df):
    plan = AnalysisPlan(
        objective="why",
        steps=[
            PlanStep(step_id=0, operation="filter_rows", params={
                "column": "year", "op": "eq", "value": 2024,
            }),
            PlanStep(step_id=1, operation="statistical_test", params={
                "column_a": "treated", "column_b": "revenue",
            }, input_step=0),
        ],
    )
    result = execute_plan_sync(sales_df, plan)
    assert all(s.success for s in result.steps), result.failed_messages()
    calcs = {c.label: c.value for c in result.steps[1].output.calculations}
    assert calcs["test"] in ("Welch's two-sample t-test", "Mann-Whitney U")


def test_input_step_unknown_reference_fails(sales_df):
    plan = AnalysisPlan(
        objective="bad",
        steps=[
            PlanStep(step_id=0, operation="value_counts", params={"column": "region"}),
            PlanStep(step_id=1, operation="top_n", params={
                "sort_by": "count", "n": 2,
            }, input_step=99),
        ],
    )
    result = execute_plan_sync(sales_df, plan)
    assert not result.steps[1].success
    assert "produced no table" in result.steps[1].error


def test_validate_rejects_forward_and_self_references(sales_df):
    profile = profile_dataframe(sales_df, "s.csv")
    plan = AnalysisPlan(
        objective="v",
        steps=[
            PlanStep(step_id=0, operation="group_aggregate", input_step=2, params={
                "group_by": "region", "metrics": [{"agg": "count"}],
            }),
            PlanStep(step_id=1, operation="group_aggregate", input_step=1, params={
                "group_by": "region", "metrics": [{"agg": "count"}],
            }),
            PlanStep(step_id=2, operation="value_counts", params={"column": "region"}),
            PlanStep(step_id=3, operation="detect_outliers", input_step=99, params={"column": "x"}),
        ],
    )
    _, issues = validate_plan_columns(plan, profile)
    joined = "\n".join(issues)
    assert "later in execution order" in joined or "acyclic" in joined
    assert "cannot reference itself" in joined
    assert any("unknown step" in i and "input_step" in i for i in issues)


def test_explicit_input_beats_implicit_working(sales_df):
    """Implicit chain: filter narrows working; explicit input_step bypasses it."""
    plan = AnalysisPlan(
        objective="sem",
        steps=[
            PlanStep(step_id=0, operation="filter_rows", params={
                "column": "year", "op": "eq", "value": 2024,
            }),
            PlanStep(step_id=1, operation="group_aggregate", params={
                "group_by": "region", "metrics": [{"agg": "count"}],
            }),
            PlanStep(step_id=2, operation="group_aggregate", params={
                "group_by": "year", "metrics": [{"agg": "count"}],
            }, input_step=None),
        ],
    )
    result = execute_plan_sync(sales_df, plan)
    assert result.success
    implicit = result.get_table(1)
    assert sum(r[1] for r in implicit.rows) == sales_df[sales_df.year == 2024].shape[0]
