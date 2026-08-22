import pandas as pd
import pytest

from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.execution.executor import execute_plan_sync
from insight.validation.verification import independently_verify


def _plan(steps):
    return AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=i, operation=op, params=p)
            for i, (op, p) in enumerate(steps)
        ],
        charts=[],
    )


def test_clean_plan_passes_recomputation(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=0, operation="group_aggregate", params={
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
            }),
            PlanStep(step_id=1, operation="value_counts", params={"column": "category"}),
            PlanStep(step_id=2, operation="correlation", params={
                "columns": ["revenue", "units"],
            }),
        ],
        charts=[],
    )
    execution = execute_plan_sync(sales_df, plan)
    issues = independently_verify(sales_df, plan, execution)
    assert issues == [], [i.message for i in issues]


def test_tampered_aggregate_is_caught(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="group_aggregate", params={
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
        })],
        charts=[],
    )
    execution = execute_plan_sync(sales_df, plan)
    table = execution.tables_by_step()[0]
    table.rows[0][1] = float(table.rows[0][1]) * 3 + 17
    issues = independently_verify(sales_df, plan, execution)
    assert any("recomputation mismatch" in i.message for i in issues)


def test_tampered_value_counts_top_is_caught(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="value_counts", params={
            "column": "category",
        })],
        charts=[],
    )
    execution = execute_plan_sync(sales_df, plan)
    table = execution.tables_by_step()[0]
    table.rows[0][0] = "GhostCategory"
    issues = independently_verify(sales_df, plan, execution)
    assert any("most frequent" in i.message for i in issues)


def test_correlation_asymmetry_detected():
    from insight.domain.execution import DataTable

    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="correlation", params={})],
        charts=[],
    )
    tampered = DataTable(
        step_id=0,
        name="correlations",
        columns=["column", "a", "b"],
        rows=[
            ["a", 1.0, 0.8],
            ["b", -0.4, 1.0],
        ],
        total_rows=2,
    )
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 1.0, 5.0, 2.0]})
    fake_execution = execute_plan_sync(df, AnalysisPlan(objective="x", steps=[
        PlanStep(step_id=0, operation="correlation", params={"columns": ["a", "b"]})
    ], charts=[]))
    fake_execution.steps[0].output.table = tampered
    issues = independently_verify(df, plan, fake_execution)
    assert any("symmetric" in i.message for i in issues)
