import pytest

from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.validation.result_validator import validate_execution


def _plan(steps_params):
    return AnalysisPlan(
        objective="t",
        steps=[
            PlanStep(step_id=i, operation="value_counts", params=p)
            for i, p in enumerate(steps_params)
        ],
        charts=[],
    )


def _run(plan, df):
    from insight.execution.executor import execute_plan_sync

    return execute_plan_sync(df, plan)


def test_nan_in_output_is_error(sales_df):
    dirty = sales_df.copy()
    plan = _plan([{"column": "region"}])
    execution = _run(plan, dirty)
    table = execution.tables_by_step()[0]
    table.rows.append(["GhostRegion", float("nan")])
    validation = validate_execution(plan, execution)
    assert not validation.valid
    assert any("non-finite" in e.message for e in validation.errors)


def test_empty_filtered_table_warns_not_errors(sales_df):
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="top_n", params={
            "sort_by": "revenue", "n": 5,
            "filters": [{"column": "region", "op": "eq", "value": "Atlantis"}],
        })],
        charts=[],
    )
    execution = _run(plan, sales_df)
    validation = validate_execution(plan, execution)
    assert validation.valid
    assert any("(filters matched no rows)" in w.message for w in validation.warnings)


def test_failed_step_blocks_validity(sales_df):
    plan = _plan([{"column": "does_not_exist"}])
    execution = _run(plan, sales_df)
    validation = validate_execution(plan, execution)
    assert not validation.valid
    assert any("failed" in e.message for e in validation.errors)


def test_chart_missing_axis_column_is_error(sales_df):
    from insight.domain.analysis import ChartSpec, ChartType

    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="value_counts", params={"column": "region"})],
        charts=[ChartSpec(chart_type=ChartType.bar, title="x", source_step=0,
                          x="region", y="no_such_column")],
    )
    execution = _run(plan, sales_df)
    validation = validate_execution(plan, execution)
    assert not validation.valid
    assert any("references column 'no_such_column'" in e.message for e in validation.errors)


def test_small_sample_warning(sales_df):
    tiny = sales_df.head(6).copy()
    plan = AnalysisPlan(
        objective="t",
        steps=[PlanStep(step_id=0, operation="group_aggregate", params={
            "group_by": "region", "metrics": [{"agg": "count"}],
        })],
        charts=[],
    )
    execution = _run(plan, tiny)
    validation = validate_execution(plan, execution)
    assert validation.valid
    assert any("small sample sizes" in w.message for w in validation.warnings)


def _profile_with_missing(df):
    from insight.profiling.profiler import profile_dataframe

    return profile_dataframe(df, "t.csv", content_hash="cafecafe" * 4)


def test_mcar_warning_for_referenced_column(sales_df):
    dirty = sales_df.copy()
    dirty.loc[dirty.index[:15], "revenue"] = None
    profile = _profile_with_missing(dirty)
    plan = _plan([{"column": "region", "filters": [{"column": "revenue", "op": "gt", "value": 0}]}])
    validation = validate_execution(plan, _run(plan, dirty), df=dirty, profile=profile)
    mcar = [w for w in validation.warnings if "MCAR" in w.message]
    assert mcar and "revenue" in mcar[0].message
    assert "missingness_caveats" in validation.checks_run


def test_no_mcar_warning_below_threshold_or_unreferenced(sales_df):
    profile = _profile_with_missing(sales_df)
    plan = _plan([{"column": "region"}])
    validation = validate_execution(plan, _run(plan, sales_df), df=sales_df, profile=profile)
    assert not [w for w in validation.warnings if "MCAR" in w.message]

    dirty = sales_df.copy()
    dirty.loc[dirty.index[:15], "units"] = None
    profile2 = _profile_with_missing(dirty)
    validation2 = validate_execution(plan, _run(plan, dirty), df=dirty, profile=profile2)
    assert not [w for w in validation2.warnings if "MCAR" in w.message]
