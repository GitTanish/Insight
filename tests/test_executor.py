from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.execution.executor import execute_plan_sync


def test_filter_mutation_flows_to_next_step(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=0, operation="filter_rows", params={
                "column": "units", "op": "gte", "value": 3,
            }),
            PlanStep(step_id=1, operation="group_aggregate", params={
                "group_by": "region",
                "metrics": [{"agg": "count"}],
            }),
        ],
        charts=[],
    )
    result = execute_plan_sync(sales_df, plan)
    assert result.success
    assert result.final_row_count == (sales_df["units"] >= 3).sum()
    table = result.get_table(1)
    assert sum(r[1] for r in table.rows) == result.final_row_count


def test_unknown_operation_recorded():
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="quantum_analysis", params={})],
        charts=[],
    )
    result = execute_plan_sync(_df(), plan)
    assert not result.success
    assert "unknown operation" in result.steps[0].error


def _df():
    import pandas as pd

    return pd.DataFrame({"v": [1, 2, 3]})


def test_invalid_params_reported(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="group_aggregate", params={"group_by": "region"})],
        charts=[],
    )
    result = execute_plan_sync(sales_df, plan)
    assert not result.steps[0].success
    assert "invalid params" in result.steps[0].error
    assert not result.success


def test_partial_failure_still_yields_tables(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=0, operation="value_counts", params={"column": "region"}),
            PlanStep(step_id=1, operation="value_counts", params={"column": "missing_col"}),
        ],
        charts=[],
    )
    result = execute_plan_sync(sales_df, plan)
    assert result.get_table(0) is not None
    assert result.get_table(1) is None
    assert len(result.failed_messages()) == 1
