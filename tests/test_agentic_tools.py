"""Offline tests for the opt-in agentic tool-calling layer (insight/tools).

No live provider: a scripted router stands in for the model so the loop,
dispatch, budget, and throttle paths are all exercised deterministically —
mirroring the repo's existing ScriptedLLM/stub-chain convention.
"""
import asyncio

import pandas as pd
import pytest

from insight.agentic import suggest_followups  # noqa: F401  (guard: name not shadowed)
from insight.analytics.operations import OPERATIONS
from insight.domain.dataset import DatasetProfile
from insight.llm.base import ToolCall
from insight.profiling.profiler import profile_dataframe
from insight.tools.loop import LLMOperationRouter, LoopBudget, LoopState, run_loop
from insight.tools.runtime import ToolCall as RuntimeToolCall
from insight.tools.runtime import dispatch
from insight.tools.tooling import (
    build_anthropic_tools,
    build_openai_tools,
    parameters_schema,
)


@pytest.fixture
def sales_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["North", "South", "East", "West"] * 15,
            "revenue": [100 + (i * 37) % 200 for i in range(60)],
        }
    )


@pytest.fixture
def profile(sales_df) -> DatasetProfile:
    return profile_dataframe(sales_df, "sales.csv", content_hash="ab" * 16)


# ---- tooling -------------------------------------------------------------


def test_every_op_becomes_a_tool():
    tools = build_openai_tools()
    assert len(tools) == len(OPERATIONS)
    for tool in tools:
        fn = tool["function"]
        assert fn["name"] in OPERATIONS
        assert fn["parameters"]["type"] == "object"


def test_schemas_are_self_contained():
    import json

    for tool in build_openai_tools() + build_anthropic_tools():
        blob = json.dumps(tool)
        assert "$ref" not in blob
        assert "$defs" not in blob


def test_anthropic_shape_uses_input_schema():
    tools = build_anthropic_tools()
    assert tools and all("input_schema" in t for t in tools)


def test_nested_metric_spec_is_inlined():
    schema = parameters_schema(OPERATIONS["group_aggregate"].params_model)
    assert "metrics" in schema["properties"]


# ---- runtime dispatch ----------------------------------------------------


def test_dispatch_valid_op(profile, sales_df):
    result = dispatch(
        RuntimeToolCall(
            id="1",
            name="group_aggregate",
            arguments={
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
            },
        ),
        sales_df,
        profile,
    )
    assert result.ok
    assert "total_revenue" in result.content


def test_dispatch_rejects_hallucinated_column(profile, sales_df):
    result = dispatch(
        RuntimeToolCall(
            id="2",
            name="group_aggregate",
            arguments={"group_by": "profit", "metrics": [{"column": "revenue", "agg": "sum"}]},
        ),
        sales_df,
        profile,
    )
    assert not result.ok
    assert "profit" in result.content


def test_dispatch_rejects_unknown_op(profile, sales_df):
    result = dispatch(RuntimeToolCall(id="3", name="drop_table", arguments={}), sales_df, profile)
    assert not result.ok
    assert "Unknown operation" in result.content


def test_dispatch_rejects_bad_params(profile, sales_df):
    result = dispatch(
        RuntimeToolCall(
            id="4",
            name="group_aggregate",
            arguments={"group_by": "region", "metrics": [{"agg": "sum"}]},
        ),
        sales_df,
        profile,
    )
    assert not result.ok



# ---- loop ----------------------------------------------------------------


class _ScriptedRouter:
    """Yields a scripted sequence of tool calls, then a final answer."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def next_action(self, state: LoopState):
        self.calls += 1
        if self._script:
            return {"tool_call": self._script.pop(0)}
        return {"final": "done"}


def _group_call(call_id="c1"):
    return RuntimeToolCall(
        id=call_id,
        name="group_aggregate",
        arguments={
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
        },
    )


class _GreedyRouter:
    """Always returns a tool call, so the loop never finishes on its own."""

    def __init__(self):
        self.calls = 0

    async def next_action(self, state):
        self.calls += 1
        return {"tool_call": _group_call(f"c{self.calls}")}


def test_loop_runs_tool_then_finishes(profile, sales_df):
    router = _ScriptedRouter([_group_call()])
    out = asyncio.run(
        run_loop("q", sales_df, profile, None, "m", router=router,
                 budget=LoopBudget(max_iterations=3, max_llm_calls=6))
    )
    assert out.success
    assert out.operations_used == ["group_aggregate"]
    assert out.llm_calls == 2  # one tool turn + one final turn
    assert out.final_answer == "done"


def test_loop_respects_iteration_cap(profile, sales_df):
    out = asyncio.run(
        run_loop("q", sales_df, profile, None, "m", router=_GreedyRouter(),
                 budget=LoopBudget(max_iterations=2, max_llm_calls=10))
    )
    assert out.iterations == 2
    assert out.error == "iteration_cap_reached"


def test_loop_respects_llm_call_budget(profile, sales_df):
    out = asyncio.run(
        run_loop("q", sales_df, profile, None, "m", router=_GreedyRouter(),
                 budget=LoopBudget(max_iterations=10, max_llm_calls=2))
    )
    assert out.llm_calls == 2  # exactly the cap, not fewer — `<=` would hide a no-op loop
    assert out.error == "llm_call_budget_exhausted"


def test_loop_marks_rate_limit_as_throttled(profile, sales_df):
    class _Throttled:
        async def next_action(self, state):
            raise RuntimeError("429 rate_limit_exceeded tpm")

    out = asyncio.run(
        run_loop("q", sales_df, profile, None, "m", router=_Throttled(), budget=LoopBudget())
    )
    assert out.throttled
    assert out.error and out.error.startswith("throttled")


def test_llm_router_parses_tool_call_response(profile, sales_df):
    class _Resp:
        content = ""
        tool_calls = [ToolCall(id="x", name="group_aggregate", arguments={
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum"}],
        })]

    async def _gen(request):
        assert request.tools, "router must attach tools to the request"
        assert request.tool_choice == "auto"
        return _Resp()

    router = LLMOperationRouter(_gen, "m", "sys")
    state = LoopState(question="q", df=sales_df, profile=profile)
    action = asyncio.run(router.next_action(state))
    assert [c.name for c in action["tool_calls"]] == ["group_aggregate"]
    assert action["tool_calls"][0].id == "x"


def test_llm_router_falls_back_to_final_text(profile, sales_df):
    class _Resp:
        content = "Here is the answer."
        tool_calls = []

    async def _gen(request):
        return _Resp()

    router = LLMOperationRouter(_gen, "m", "sys")
    state = LoopState(question="q", df=sales_df, profile=profile)
    action = asyncio.run(router.next_action(state))
    assert action["final"] == "Here is the answer."
