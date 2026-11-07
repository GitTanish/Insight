import numpy as np
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


def test_top_n_dropped_projected_column_is_detected():
    df = pd.DataFrame({"region": ["N", "S", "E"], "revenue": [10, 30, 20]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="top_n",
        params={"sort_by": "revenue", "n": 2, "columns": ["region"]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    table = execution.tables_by_step()[0]
    index = table.columns.index("region")
    table.columns.pop(index)
    for row in table.rows:
        row.pop(index)
    issues = independently_verify(df, plan, execution)
    assert any(
        "top_n output column 'region' missing" in i.message for i in issues
    ), [i.message for i in issues]


def test_top_n_projected_columns_with_filter_verifies_clean():
    df = pd.DataFrame({
        "region": ["N", "S", "E", "W", "N"],
        "revenue": [10, 30, 20, 5, 40],
        "flag": [1, 1, 0, 1, 1],
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="top_n",
        params={
            "sort_by": "revenue",
            "n": 2,
            "columns": ["region", "revenue"],
            "filters": [{"column": "flag", "op": "eq", "value": 1}],
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    issues = independently_verify(df, plan, execution)
    assert issues == [], [i.message for i in issues]

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("revenue")] += 1
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
    assert any(
        "correlation" in i.message and "recomputes" in i.message for i in issues
    ), [i.message for i in issues]


def test_correlation_symmetric_tamper_and_calculation_are_detected():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 1.0, 5.0, 2.0]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0, operation="correlation", params={"columns": ["a", "b"]}
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    a_idx = table.columns.index("a")
    b_idx = table.columns.index("b")
    table.rows[0][b_idx] = 0.8
    table.rows[1][a_idx] = 0.8
    issues = independently_verify(df, plan, execution)
    assert any(
        "correlation" in i.message and "recomputes" in i.message for i in issues
    ), [i.message for i in issues]

    execution = execute_plan_sync(df, plan)
    calc = next(
        c for c in execution.steps[0].output.calculations
        if c.label == "strongest_correlation"
    )
    calc.value = round(float(calc.value) + 0.5, 6)
    issues = independently_verify(df, plan, execution)
    assert any("strongest_correlation" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_correlation_dropped_column_is_detected():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3.0, 1.0, 5.0, 2.0]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0, operation="correlation", params={"columns": ["a", "b"]}
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    table = execution.tables_by_step()[0]
    index = table.columns.index("b")
    table.columns.pop(index)
    for row in table.rows:
        row.pop(index)
    issues = independently_verify(df, plan, execution)
    assert any("column 'b' missing" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_group_aggregate_duplicate_row_and_dropped_metric_are_detected():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "v": [1.0, 2.0, 3.0, 4.0]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="group_aggregate",
        params={
            "group_by": "g",
            "metrics": [{"column": "v", "agg": "sum", "alias": "total_v"}],
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows.append(list(table.rows[0]))
    issues = independently_verify(df, plan, execution)
    assert any("returned 3 rows, expected 2" in i.message for i in issues), [
        i.message for i in issues
    ]

    execution = execute_plan_sync(df, plan)
    table = execution.tables_by_step()[0]
    index = table.columns.index("total_v")
    table.columns.pop(index)
    for row in table.rows:
        row.pop(index)
    issues = independently_verify(df, plan, execution)
    assert any("column 'total_v' missing" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_value_counts_nonfinite_proportion_is_detected():
    df = pd.DataFrame({"category": ["a", "a", "b", "c"]})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="value_counts",
        params={"column": "category", "normalize": True},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("proportion")] = float("nan")
    issues = independently_verify(df, plan, execution)
    assert any("proportion" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_regression_recomputation_verifies_and_catches_tamper():
    rng = np.random.default_rng(5)
    n = 80
    x1 = rng.normal(0, 1.5, n)
    x2 = rng.normal(0, 1.0, n)
    y = 2.5 + 1.7 * x1 - 0.9 * x2 + rng.normal(0, 0.2, n)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="regression",
        params={"target_column": "y", "feature_columns": ["x1", "x2"]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[1][table.columns.index("coefficient")] += 1.0
    issues = independently_verify(df, plan, execution)
    assert any(
        "regression" in i.message and "coefficient" in i.message for i in issues
    ), [i.message for i in issues]


def test_regression_calculation_tamper_is_caught():
    rng = np.random.default_rng(6)
    n = 50
    x = rng.normal(0, 1, n)
    y = 4.0 + 2.0 * x + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"x": x, "y": y})
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="regression",
        params={"target_column": "y", "feature_columns": ["x"]},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    calc = next(
        c for c in execution.steps[0].output.calculations if c.label == "r_squared"
    )
    calc.value = float(calc.value) - 0.05
    issues = independently_verify(df, plan, execution)
    assert any("regression r_squared" in i.message for i in issues)


def test_anova_recomputation_verifies_and_catches_statistic_tamper():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "group": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        "value": np.concatenate([
            rng.normal(10, 2, 40),
            rng.normal(14, 2, 40),
            rng.normal(18, 2, 40),
        ]),
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="anova_test",
        params={"group_column": "group", "value_column": "value"},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    calc = next(
        c for c in execution.steps[0].output.calculations if c.label == "statistic"
    )
    calc.value = float(calc.value) + 1.0
    issues = independently_verify(df, plan, execution)
    assert any("anova_test statistic" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_anova_group_summary_tamper_is_caught():
    df = pd.DataFrame({
        "group": ["a"] * 10 + ["b"] * 10 + ["c"] * 10,
        "value": list(range(30)),
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="anova_test",
        params={"group_column": "group", "value_column": "value"},
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[0][table.columns.index("mean")] += 5.0
    issues = independently_verify(df, plan, execution)
    assert any("anova_test group" in i.message for i in issues)


def test_compare_subsets_recomputation_verifies_and_catches_tamper():
    df = pd.DataFrame({
        "group": ["a"] * 40 + ["b"] * 40,
        "value": list(range(80)),
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="compare_subsets",
        params={
            "metric": {"column": "value", "agg": "mean"},
            "subset_a": [{"column": "group", "op": "eq", "value": "a"}],
            "subset_b": [{"column": "group", "op": "eq", "value": "b"}],
            "label_a": "A",
            "label_b": "B",
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    table = execution.tables_by_step()[0]
    table.rows[1][table.columns.index("value")] += 5.0
    issues = independently_verify(df, plan, execution)
    assert any("compare_subsets" in i.message for i in issues), [
        i.message for i in issues
    ]


def test_compare_subsets_delta_tamper_is_caught():
    df = pd.DataFrame({
        "group": ["a"] * 10 + ["b"] * 10,
        "value": [float(v) for v in range(20)],
    })
    plan = AnalysisPlan(objective="t", steps=[PlanStep(
        step_id=0,
        operation="compare_subsets",
        params={
            "metric": {"column": "value", "agg": "sum"},
            "subset_a": [{"column": "group", "op": "eq", "value": "a"}],
            "subset_b": [{"column": "group", "op": "eq", "value": "b"}],
        },
    )], charts=[])
    execution = execute_plan_sync(df, plan)
    assert independently_verify(df, plan, execution) == []

    calc = next(
        c for c in execution.steps[0].output.calculations
        if c.label == "absolute_delta"
    )
    calc.value = float(calc.value) + 3.0
    issues = independently_verify(df, plan, execution)
    assert any("absolute_delta" in i.message for i in issues)


def test_new_statistical_verifiers_tolerate_filters_and_missingness():
    rng = np.random.default_rng(12)
    n = 120
    df = pd.DataFrame({
        "group": ["a", "b", "c"] * (n // 3),
        "flag": [1, 0, 1, 0] * (n // 4),
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
    })
    df["y"] = 3 + 1.2 * df["x1"] - 0.4 * df["x2"] + rng.normal(0, 0.1, n)
    df.loc[::19, "x1"] = np.nan
    df.loc[::23, "y"] = np.nan
    plans = [
        AnalysisPlan(objective="t", steps=[PlanStep(
            step_id=0,
            operation="regression",
            params={
                "target_column": "y",
                "feature_columns": ["x1", "x2"],
                "filters": [{"column": "flag", "op": "eq", "value": 1}],
            },
        )], charts=[]),
        AnalysisPlan(objective="t", steps=[PlanStep(
            step_id=0,
            operation="anova_test",
            params={
                "group_column": "group",
                "value_column": "y",
                "filters": [{"column": "flag", "op": "eq", "value": 1}],
            },
        )], charts=[]),
        AnalysisPlan(objective="t", steps=[PlanStep(
            step_id=0,
            operation="compare_subsets",
            params={
                "metric": {"column": "y", "agg": "mean"},
                "subset_a": [{"column": "group", "op": "eq", "value": "a"}],
                "subset_b": [{"column": "group", "op": "eq", "value": "b"}],
                "label_a": "A",
                "label_b": "B",
            },
        )], charts=[]),
    ]
    for plan in plans:
        execution = execute_plan_sync(df, plan)
        issues = independently_verify(df, plan, execution)
        assert issues == [], [i.message for i in issues]
