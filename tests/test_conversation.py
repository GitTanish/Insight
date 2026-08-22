import json

from insight.conversation.state import (
    AnalysisSessionState,
    extract_state_from_plan,
    serialize_state_for_planner,
)
from insight.domain.analysis import AnalysisPlan
from insight.domain.query import AnalysisRequest
from insight.orchestrator import run_analysis_sync


def _plan(payload: str) -> AnalysisPlan:
    return AnalysisPlan.model_validate(json.loads(payload))


def test_extract_replaces_same_column_op_filter():
    plan = _plan("""
    {
      "objective": "o",
      "steps": [
        {"step_id": 0, "operation": "filter_rows",
         "params": {"column": "region", "op": "eq", "value": "North"}},
        {"step_id": 1, "operation": "filter_rows",
         "params": {"column": "region", "op": "eq", "value": "West"}},
        {"step_id": 2, "operation": "filter_rows",
         "params": {"column": "revenue", "op": "gte", "value": 100}}
      ]
    }
    """)
    state = extract_state_from_plan(plan)
    assert len(state.active_filters) == 2
    region = next(f for f in state.active_filters if f.column == "region")
    assert region.value == "West"
    assert state.turns_analyzed == 1


def test_extract_collects_dims_and_metrics():
    plan = _plan("""
    {
      "objective": "o",
      "steps": [
        {"step_id": 0, "operation": "group_aggregate",
         "params": {"group_by": ["region", "category"],
                    "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}]}}
      ]
    }
    """)
    state = extract_state_from_plan(plan)
    assert state.dims == ["region", "category"]
    assert state.metrics == ["total_revenue"]


def test_extract_accumulates_over_base():
    base = AnalysisSessionState(
        active_filters=[{"column": "region", "op": "eq", "value": "North"}],
        dims=["region"],
        turns_analyzed=2,
    )
    plan = _plan("""
    {
      "objective": "o",
      "steps": [
        {"step_id": 0, "operation": "group_aggregate",
         "params": {"group_by": "category",
                    "metrics": [{"column": "revenue", "agg": "mean"}]}}
      ]
    }
    """)
    state = extract_state_from_plan(plan, base=base)
    assert [f.value for f in state.active_filters] == ["North"]
    assert state.dims == ["region", "category"]
    assert state.metrics == ["revenue"]
    assert state.turns_analyzed == 3


def test_serialize_state_for_planner():
    empty = serialize_state_for_planner(None)
    assert empty == "(no active filters or tracked fields)"

    state = AnalysisSessionState(
        active_filters=[{"column": "region", "op": "in", "value": ["North", "South"]}],
        dims=["region"],
        metrics=["total_revenue"],
    )
    text = serialize_state_for_planner(state)
    assert "region in" in text
    assert '["North", "South"]' in text
    assert "dimensions analyzed so far: region" in text
    assert "metrics analyzed so far: total_revenue" in text


FILTER_PLAN = """
{
  "objective": "West revenue by category",
  "steps": [
    {"step_id": 0, "operation": "filter_rows",
     "params": {"column": "region", "op": "eq", "value": "West"},
     "reason": "narrow scope"},
    {"step_id": 1, "operation": "group_aggregate",
     "params": {"group_by": "category",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
                "sort_by": "total_revenue"},
     "reason": "aggregate"}
  ],
  "charts": [
    {"chart_type": "bar", "title": "Revenue by Category (West)", "source_step": 1,
     "x": "category", "y": "total_revenue"}
  ]
}
"""


def test_orchestrator_returns_analysis_state(stub_chain_factory, sales_df, profile, tmp_path):
    stub_chain_factory([FILTER_PLAN.strip(), "Filtered answer."])

    response = run_analysis_sync(
        sales_df,
        profile,
        AnalysisRequest(question="Revenue for West only"),
        artifacts_dir=tmp_path,
    )

    assert response.error is None
    state = response.meta["analysis_state"]
    filters = {(f["column"], f["op"]): f["value"] for f in state["active_filters"]}
    assert filters[("region", "eq")] == "West"
    assert state["dims"] == ["category"]
    assert state["metrics"] == ["total_revenue"]
    assert state["turns_analyzed"] == 1
    tables = response.tables
    assert len(tables) == 1
    assert tables[0].columns == ["category", "total_revenue"]
    assert {row[0] for row in tables[0].rows} <= {"A", "B", "C"}


def test_orchestrator_injects_session_context_into_planner(
    stub_chain_factory, sales_df, profile, tmp_path
):
    llm = stub_chain_factory([FILTER_PLAN.strip(), "Follow-up answer."])
    carried = AnalysisSessionState(
        active_filters=[{"column": "region", "op": "eq", "value": "West"}],
        turns_analyzed=1,
    )

    run_analysis_sync(
        sales_df,
        profile,
        AnalysisRequest(question="Now break it down by category"),
        artifacts_dir=tmp_path,
        state=carried,
    )

    system_prompt = llm.requests[0].messages[0].content
    assert "Active analysis state" in system_prompt
    assert 'region eq "West"' in system_prompt
