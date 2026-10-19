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


def test_count_metric_with_alias_is_verified(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="group_aggregate", params={
            "group_by": "region",
            "metrics": [{"agg": "count", "alias": "n"}],
        })],
        charts=[],
    )
    execution = execute_plan_sync(sales_df, plan)
    issues = independently_verify(sales_df, plan, execution)
    assert issues == [], [i.message for i in issues]

    table = execution.tables_by_step()[0]
    assert "n" in table.columns
    table.rows[0][table.columns.index("n")] += 7
    issues = independently_verify(sales_df, plan, execution)
    assert any(
        "recomputation mismatch (n)" in i.message
        for i in issues
    ), [i.message for i in issues]


def test_group_aggregate_top_n_verifies_clean():
    df = pd.DataFrame({"g": list("abcde"), "v": [5, 4, 3, 2, 1]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={
            "group_by": "g",
            "metrics": [{"column": "v", "agg": "sum"}],
            "sort_by": "sum_v",
            "top_n": 2,
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("sum_v")] += 3
    issues = independently_verify(df, plan, execution)
    assert any("recomputation mismatch" in i.message for i in issues)


def test_group_aggregate_other_after_verifies_clean():
    df = pd.DataFrame({
        "g": ["A"] * 10 + ["B"] * 9 + ["C"] * 8 + ["D"] * 7 + ["E"] * 6 + ["F"] * 5,
        "v": list(range(45)),
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={
            "group_by": "g",
            "metrics": [{"column": "v", "agg": "mean"}],
            "other_after": 4,
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    issues = independently_verify(df, plan, execution)
    assert issues == [], [i.message for i in issues]


def test_truncated_high_cardinality_group_aggregate_verifies_clean():
    df = pd.DataFrame({"g": [f"g{i}" for i in range(150)], "v": list(range(150))})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={"group_by": "g", "metrics": [{"column": "v", "agg": "sum"}]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    table = execution.tables_by_step()[0]
    assert table.truncated
    issues = independently_verify(df, plan, execution)
    assert issues == [], [i.message for i in issues]


def test_nan_metric_group_verifies_clean():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, None, None]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={"group_by": "g", "metrics": [{"column": "v", "agg": "mean"}]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    issues = independently_verify(df, plan, execution)
    assert issues == [], [i.message for i in issues]


def test_nunique_string_metric_verifies_clean_and_tamper_caught():
    df = pd.DataFrame({"g": ["a", "a", "b"], "c": ["x", "y", "x"]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={"group_by": "g", "metrics": [{"column": "c", "agg": "nunique"}]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("nunique_c")] = 99
    issues = independently_verify(df, plan, execution)
    assert any("recomputation mismatch" in i.message for i in issues)


def test_list_group_by_tamper_is_caught():
    df = pd.DataFrame({
        "g1": ["a", "a", "b", "b"],
        "g2": ["x", "y", "x", "y"],
        "v": [1, 2, 3, 4],
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={
            "group_by": ["g1", "g2"],
            "metrics": [{"column": "v", "agg": "sum"}],
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("sum_v")] += 10
    issues = independently_verify(df, plan, execution)
    assert any("recomputation mismatch" in i.message for i in issues)


def test_nan_group_key_tamper_is_caught():
    df = pd.DataFrame({"g": ["a", "a", None, None, None], "v": [1, 2, 3, 4, 5]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={"group_by": "g", "metrics": [{"agg": "count"}]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[-1][table.columns.index("row_count")] = 99
    issues = independently_verify(df, plan, execution)
    assert any("recomputation mismatch" in i.message for i in issues)


def test_value_counts_respects_filters_and_catches_count_tamper():
    df = pd.DataFrame({"c": ["A"] * 30 + ["B"] * 20 + ["C"] * 10})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="value_counts",
        params={"column": "c", "filters": [{"column": "c", "op": "ne", "value": "A"}]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("count")] += 5
    issues = independently_verify(df, plan, execution)
    assert any("value_counts" in i.message for i in issues)


def test_top_n_projected_columns_tamper_is_caught():
    df = pd.DataFrame({"region": ["N", "S", "E"], "revenue": [10, 30, 20]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="top_n",
        params={"sort_by": "revenue", "n": 2, "columns": ["region"]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("region")] = "E"
    issues = independently_verify(df, plan, execution)
    assert any("top_n" in i.message for i in issues)


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
