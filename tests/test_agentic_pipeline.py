"""Offline tests for the agentic tool-use pipeline (loop -> rebuild -> tail).

No live provider: scripted tool-call responses stand in for the model so the
orchestrator branch, plan/execution rebuild, chart synthesis, grounding,
budgets, throttling and the webapp agentic path are all exercised
deterministically — mirroring the repo's scripted-LLM/stub-chain convention.
"""
import asyncio

import pandas as pd
import pytest

from insight.domain.query import AnalysisRequest
from insight.llm.base import LLMResponse, ToolCall
from insight.orchestrator import analyze_stream, run_analysis_sync
from insight.tools.loop import LoopBudget, LoopState, run_loop
from insight.tools.pipeline import (
    agentic_meta,
    chart_specs_from_results,
    execution_from_outcome,
    plan_from_outcome,
)
from insight.tools.runtime import ToolCall as RuntimeToolCall
from insight.tools.tooling import build_anthropic_tools


# ---- scripted tool-call LLM ------------------------------------------------


class ScriptedToolLLM:
    """Fake generate() returning scripted responses (tool calls or text)."""

    def __init__(self, responses: list):
        self.responses = list(responses)
        self.requests: list = []

    async def generate(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("ScriptedToolLLM exhausted")
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class ToolStubChain:
    """StubChain equivalent whose responses may carry tool calls."""

    def __init__(self, llm: ScriptedToolLLM):
        self.llm = llm
        self.temperature = 0.0
        self.reasoning_effort = None
        self.last_used = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.cached_tokens_in = 0
        self.llm_calls = 0
        self.llm_latency_ms = 0

    @property
    def primary_model(self) -> str:
        return "fake-1"

    async def generate(self, request):
        response = await self.llm.generate(request)
        self.last_used = "fake/fake-1"
        self.llm_calls += 1
        self.llm_latency_ms += response.latency_ms
        return response

    async def stream(self, request):
        response = await self.generate(request)
        text = response.content or ""
        if not text:
            return
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        yield text[midpoint:]


@pytest.fixture
def tool_stub_chain_factory(monkeypatch):
    def _install(responses: list) -> ToolStubChain:
        import insight.orchestrator as orch

        llm = ScriptedToolLLM(responses)
        chain = ToolStubChain(llm)
        monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: chain)
        return chain

    return _install


def _tool_call(name: str, call_id: str = "c1", **arguments) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _tool_response(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        content="", provider="fake", model="fake-1", latency_ms=1,
        tool_calls=list(calls),
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        content=text, provider="fake", model="fake-1", latency_ms=1
    )


def _group_call(call_id: str = "c1") -> ToolCall:
    return _tool_call(
        "group_aggregate",
        call_id,
        group_by="region",
        metrics=[{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
    )


class _ScriptedRouter:
    """Yields a scripted sequence of actions, then a final answer."""

    def __init__(self, script):
        self._script = list(script)

    async def next_action(self, state: LoopState):
        if self._script:
            return self._script.pop(0)
        return {"final": "done"}


async def _collect(df, profile, request, **kwargs):
    events = []
    async for event in analyze_stream(df, profile, request, **kwargs):
        events.append(event)
    return events


def _result(events):
    results = [e["response"] for e in events if e["type"] == "result"]
    return results[0] if results else None


def _types(events):
    return [e["type"] for e in events]


# ---- loop behaviours --------------------------------------------------------


def test_loop_dispatches_multiple_tool_calls_in_one_turn(profile, sales_df):
    router = _ScriptedRouter([{
        "tool_calls": [
            RuntimeToolCall(id="a", name="group_aggregate", arguments={
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
            }),
            RuntimeToolCall(id="b", name="value_counts", arguments={"column": "category"}),
        ]
    }])
    out = asyncio.run(run_loop("q", sales_df, profile, None, "m", router=router))
    assert out.success
    assert out.operations_used == ["group_aggregate", "value_counts"]
    assert len(out.tool_results) == 2
    assert all(r.ok for r in out.tool_results)
    # the failed-input guard leaves one assistant message carrying both calls
    # plus one tool message per result
    assert len(out.tool_results) == 2


def test_loop_chains_via_input_step(profile, sales_df):
    router = _ScriptedRouter([
        {"tool_calls": [RuntimeToolCall(id="a", name="group_aggregate", arguments={
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
        })]},
        {"tool_calls": [RuntimeToolCall(id="b", name="top_n", arguments={
            "sort_by": "total_revenue", "n": 2, "input_step": 0,
        })]},
    ])
    out = asyncio.run(run_loop("q", sales_df, profile, None, "m", router=router))
    assert out.success
    assert out.tool_results[1].ok
    assert out.tool_results[1].input_step == 0
    assert len(out.tool_results[1].table.rows) == 2
    assert "total_revenue" in out.tool_results[1].content


def test_loop_rejects_input_step_pointing_nowhere(profile, sales_df):
    router = _ScriptedRouter([
        {"tool_calls": [RuntimeToolCall(id="a", name="top_n", arguments={
            "sort_by": "revenue", "n": 2, "input_step": 42,
        })]},
    ])
    out = asyncio.run(run_loop("q", sales_df, profile, None, "m", router=router))
    assert not out.success
    assert "input_step 42" in out.tool_results[0].content


def test_filter_rows_narrows_working_frame_for_later_calls(profile, sales_df):
    # Implicit chaining: without input_step, filter_rows narrows the working
    # row-set that later calls run on (loop.py `working = result.frame`).
    router = _ScriptedRouter([
        {"tool_calls": [RuntimeToolCall(id="a", name="filter_rows", arguments={
            "column": "region", "op": "eq", "value": "West",
        })]},
        {"tool_calls": [RuntimeToolCall(id="b", name="group_aggregate", arguments={
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
        })]},
    ])
    out = asyncio.run(run_loop("q", sales_df, profile, None, "m", router=router))
    assert out.success
    table = out.tool_results[1].table
    assert [row[0] for row in table.rows] == ["West"]
    west_total = float(sales_df.loc[sales_df["region"] == "West", "revenue"].sum())
    assert abs(float(table.rows[0][1]) - west_total) < 0.01


def test_loop_stops_after_repeated_identical_failure(profile, sales_df):
    class _Broken:
        async def next_action(self, state):
            return {"tool_call": RuntimeToolCall(
                id="x", name="group_aggregate",
                arguments={"group_by": "nope", "metrics": [{"column": "revenue", "agg": "sum"}]},
            )}

    out = asyncio.run(run_loop(
        "q", sales_df, profile, None, "m", router=_Broken(),
        budget=LoopBudget(max_iterations=10, max_llm_calls=10),
    ))
    assert out.error == "repeated_failing_call"
    assert out.llm_calls == 3


def test_loop_emits_events_for_sse_trail(profile, sales_df):
    seen: list[dict] = []

    async def _sink(event):
        seen.append(event)

    router = _ScriptedRouter([{"tool_calls": [_group_call()]}])
    out = asyncio.run(run_loop(
        "q", sales_df, profile, None, "m", router=router, on_event=_sink,
    ))
    assert out.success
    kinds = [e["type"] for e in seen]
    assert kinds == ["tool_call", "tool_result"]
    assert seen[0]["name"] == "group_aggregate"
    assert seen[0]["arguments"]["group_by"] == "region"
    assert seen[1]["ok"] is True
    assert "total_revenue" in seen[1]["preview"]


# ---- rebuild: plan / execution / charts / meta ------------------------------


def _loop_outcome():
    from insight.tools.loop import LoopOutcome
    from insight.tools.runtime import ToolResult

    return LoopOutcome(
        tool_results=[
            ToolResult(
                id="c1", name="group_aggregate", ok=True, content="region table",
                step_id=0,
                params={"group_by": "region", "metrics": [
                    {"column": "revenue", "agg": "sum", "alias": "total_revenue"}
                ]},
            ),
            ToolResult(
                id="c2", name="top_n", ok=True, content="ranked rows",
                step_id=1, input_step=0,
                params={"sort_by": "total_revenue", "n": 3},
            ),
            ToolResult(
                id="c3", name="value_counts", ok=False,
                content="Invalid arguments: unknown column 'nope'",
                step_id=None,
            ),
        ],
        operations_used=["group_aggregate", "top_n", "value_counts"],
        iterations=3,
        llm_calls=4,
        final_answer="West leads with 4,584.",
    )


def test_rebuild_plan_and_execution_include_only_successful_steps():
    outcome = _loop_outcome()
    plan = plan_from_outcome(outcome, "Which region leads revenue?")
    assert [s.operation for s in plan.steps] == ["group_aggregate", "top_n"]
    assert plan.steps[1].input_step == 0
    assert plan.steps[0].params["group_by"] == "region"

    execution = execution_from_outcome(outcome, pd.DataFrame({"a": [1]}))
    assert execution.success
    assert [s.operation for s in execution.steps] == ["group_aggregate", "top_n"]


def test_rebuild_requires_at_least_one_successful_step():
    from insight.tools.loop import LoopOutcome
    from insight.tools.runtime import ToolResult

    outcome = LoopOutcome(tool_results=[
        ToolResult(id="x", name="value_counts", ok=False, content="boom"),
    ])
    with pytest.raises(ValueError):
        plan_from_outcome(outcome, "q")


def test_chart_synthesis_from_group_aggregate_result(profile, sales_df):
    router = _ScriptedRouter([{"tool_calls": [_group_call()]}])
    out = asyncio.run(run_loop("q", sales_df, profile, None, "m", router=router))
    charts = chart_specs_from_results(out.tool_results)
    assert len(charts) == 1
    chart = charts[0]
    assert chart.chart_type.value == "bar"
    assert chart.x == "region"
    assert chart.y == "total_revenue"
    assert chart.source_step == 0
    assert "total_revenue by region" == chart.title


def test_chart_synthesis_ignores_failed_and_non_table_results(profile, sales_df):
    class _OneFailThenGroup:
        async def next_action(self, state):
            return {"tool_call": RuntimeToolCall(
                id="x", name="value_counts",
                arguments={"column": "not_a_column"},
            )}

    asyncio.run(run_loop("q", sales_df, profile, None, "m", router=_OneFailThenGroup()))
    # direct call: failed results never produce charts
    from insight.tools.runtime import ToolResult

    charts = chart_specs_from_results([
        ToolResult(id="x", name="value_counts", ok=False, content="boom"),
    ])
    assert charts == []


def test_agentic_meta_reports_telemetry():
    outcome = _loop_outcome()
    meta = agentic_meta(outcome)
    assert meta["agentic"] is True
    assert meta["agentic_iterations"] == 3
    assert meta["agentic_llm_calls"] == 4
    assert meta["agentic_ops"] == ["group_aggregate", "top_n", "value_counts"]
    assert len(meta["agentic_trail"]) == 3
    assert meta["agentic_trail"][2]["ok"] is False


# ---- orchestrator integration ----------------------------------------------


AGENTIC_ANSWER = "West leads with the highest total revenue."


def test_agentic_pipeline_end_to_end(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    chain = tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response("West leads with 4,584 total revenue."),
        _text_response(AGENTIC_ANSWER),
    ])
    request = AnalysisRequest(question="Which region has the highest revenue?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    types = _types(events)
    assert "tool_call" in types and "tool_result" in types
    assert "result" in types

    response = _result(events)
    assert response is not None and response.error is None
    assert AGENTIC_ANSWER in response.answer
    assert response.meta["agentic"] is True
    assert response.meta["agentic_llm_calls"] == 2
    assert response.meta["agentic_iterations"] == 2
    assert response.meta["steps"][0]["operation"] == "group_aggregate"
    assert response.validation.valid
    # rebuilt charts render from the synthesized spec
    assert response.plots
    assert chain.llm_calls == 3  # one tool turn + one final + explainer


def test_agentic_answer_numbers_are_ground_checked(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response("West leads."),
        # explainer cites a figure no computation produced -> grounding fires
        _text_response("West leads with 999999 dollars, far ahead of every region."),
        # the rewrite re-offends, so the flag survives into the response
        _text_response("West still leads with 999999 dollars."),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert "999999" in response.meta["unverified_figures"]
    assert any("not found in computed results" in e.label for e in response.evidence)
    # `grounding_retry_ms is not None` would also pass on 0 — pin the retry on
    # the stream event that only a real rewrite emits.
    assert "delta_reset" in _types(events)


def test_agentic_failed_calls_become_warnings_not_errors(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    bad_call = _tool_call("group_aggregate", "c0", group_by="nope",
                          metrics=[{"column": "revenue", "agg": "sum"}])
    tool_stub_chain_factory([
        _tool_response(bad_call),
        _tool_response(_group_call("c1")),
        _text_response("West leads."),
        _text_response(AGENTIC_ANSWER),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error is None
    assert response.validation.valid
    assert any("group_aggregate" in w.label and "recovered" in w.label
               for w in response.evidence if w.kind == "warning")
    # the failed call is not part of the rebuilt plan
    assert [s["operation"] for s in response.meta["steps"]] == ["group_aggregate"]
    assert len(response.meta["agentic_trail"]) == 2


def test_agentic_throttled_returns_error_without_inventing(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    class RateLimitError(Exception):
        pass

    tool_stub_chain_factory([RateLimitError("429 rate_limit_exceeded")])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error and response.error.startswith("throttled")
    assert "rate-limited" in response.answer
    assert response.meta["agentic_throttled"] is True


def test_agentic_llm_failure_returns_error(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    tool_stub_chain_factory([RuntimeError("boom")])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error == "RuntimeError: boom"
    assert "no answer was invented" in response.answer.lower() or "failed" in response.answer.lower()


def test_agentic_exhausted_script_errors_gracefully(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    tool_stub_chain_factory([_tool_response(_group_call("c1"))])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error is not None
    assert "AssertionError" in response.error


def test_sync_entrypoint_supports_agentic(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response("West leads."),
        _text_response(AGENTIC_ANSWER),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    response = run_analysis_sync(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    )
    assert response.error is None
    assert response.meta["agentic"] is True


def test_agentic_defaults_follow_settings(monkeypatch, tool_stub_chain_factory, profile, sales_df, tmp_path):
    from insight.settings import get_settings

    monkeypatch.setenv("INSIGHT_AGENTIC", "on")
    get_settings.cache_clear()
    try:
        tool_stub_chain_factory([
            _tool_response(_group_call("c1")),
            _text_response("West leads."),
            _text_response(AGENTIC_ANSWER),
        ])
        request = AnalysisRequest(question="Which region leads?")
        events = asyncio.run(_collect(
            sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
        ))
        response = _result(events)
        assert response.meta["agentic"] is True
    finally:
        get_settings.cache_clear()


def test_agentic_cache_key_separates_modes(profile, tmp_path, monkeypatch):
    from insight import cache as result_cache

    monkeypatch.setattr(result_cache, "get_settings", lambda: type(
        "S", (), {"query_cache_enabled": True, "temperature": 0.0,
                  "default_model_id": "x", "agentic": False},
    )())
    planned = result_cache.cache_key_for(
        profile, AnalysisRequest(question="same question")
    )
    agentic = result_cache.cache_key_for(
        profile, AnalysisRequest(question="same question", agentic=True)
    )
    assert planned != agentic


def test_plan_only_takes_precedence_over_agentic(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    """HITL needs a reviewable plan; agentic execution must not slip past it."""
    tool_stub_chain_factory([_text_response(
        '{"objective":"Revenue by region","steps":[{"step_id":0,'
        '"operation":"group_aggregate","params":{"group_by":"region",'
        '"metrics":[{"column":"revenue","agg":"sum","alias":"total_revenue"}]}}]}'
    )])
    request = AnalysisRequest(question="Which region leads?", agentic=True, plan_only=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    types = _types(events)
    assert "tool_call" not in types
    assert "plan_review" in types
    review = next(e for e in events if e["type"] == "plan_review")
    assert review["plan"]["steps"][0]["operation"] == "group_aggregate"


def test_pipeline_runs_without_artifacts_dir(profile, sales_df, monkeypatch, tool_stub_chain_factory):
    tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response("West leads."),
        _text_response(AGENTIC_ANSWER),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    response = run_analysis_sync(sales_df, profile, request, use_cache=False)
    assert response.error is None
    assert response.plots


def test_agentic_grounding_compliant_rewrite_ships_the_clean_answer(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    """The happy grounding path: a fabricated figure forces a rewrite, the
    rewrite complies, and the compliant text — not the fabricated one — ships."""
    totals = sales_df.groupby("region")["revenue"].sum()
    leader = str(totals.idxmax())
    compliant = f"{leader} leads with {int(totals.max()):,} in total revenue."
    tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response(f"{leader} leads."),
        _text_response(f"{leader} leads with 999999 dollars, far ahead."),
        _text_response(compliant),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error is None
    assert "delta_reset" in _types(events)  # the retry actually streamed
    assert compliant in response.answer
    assert "999999" not in response.answer
    assert response.meta["unverified_figures"] == []
    assert not any("not found in computed results" in e.label for e in response.evidence)


def test_agentic_tail_runs_independent_recomputation(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    """INV-13, negative-summoned: don't just assert `validation.valid` — prove
    the independent verification code path actually ran on the agentic tail."""
    from insight.validation import verification

    calls = {"n": 0}
    real = verification.independently_verify

    def _spy(df, plan, execution):
        calls["n"] += 1
        return real(df, plan, execution)

    monkeypatch.setattr(verification, "independently_verify", _spy)
    tool_stub_chain_factory([
        _tool_response(_group_call("c1")),
        _text_response("West leads."),
        _text_response(AGENTIC_ANSWER),
    ])
    request = AnalysisRequest(question="Which region leads?", agentic=True)
    events = asyncio.run(_collect(
        sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
    ))
    response = _result(events)
    assert response.error is None
    assert calls["n"] >= 1


def test_plan_approval_setting_takes_precedence_over_agentic(profile, sales_df, tmp_path, monkeypatch, tool_stub_chain_factory):
    """INV-14 via the setting (not just the request flag): INSIGHT_PLAN_APPROVAL
    forces every question down the reviewable planner, agentic flag or not."""
    from insight.settings import get_settings

    monkeypatch.setenv("INSIGHT_PLAN_APPROVAL", "on")
    get_settings.cache_clear()
    try:
        tool_stub_chain_factory([_text_response(
            '{"objective":"Revenue by region","steps":[{"step_id":0,'
            '"operation":"group_aggregate","params":{"group_by":"region",'
            '"metrics":[{"column":"revenue","agg":"sum","alias":"total_revenue"}]}}]}'
        )])
        request = AnalysisRequest(question="Which region leads?", agentic=True)
        events = asyncio.run(_collect(
            sales_df, profile, request, artifacts_dir=tmp_path, use_cache=False
        ))
        types = _types(events)
        assert "tool_call" not in types
        assert "plan_review" in types
    finally:
        get_settings.cache_clear()


# ---- provider tool-call shapes ----------------------------------------------


def test_anthropic_tools_and_payload_carry_tool_use(profile, sales_df):
    from insight.llm.base import LLMRequest, Message
    from insight.llm.providers.anthropic import AnthropicProvider

    tools = build_anthropic_tools(["group_aggregate"])
    assert tools and all("input_schema" in t for t in tools)
    assert "input_step" in tools[0]["input_schema"]["properties"]

    provider = AnthropicProvider(api_key="test")
    request = LLMRequest(
        model="claude-sonnet-4",
        messages=[
            Message(role="system", content="sys"),
            Message(role="user", content="Which region leads?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[{
                    "id": "toolu_1", "name": "group_aggregate",
                    "arguments": {"group_by": "region", "metrics": [
                        {"column": "revenue", "agg": "sum"}
                    ]},
                }],
            ),
            Message(
                role="tool", content="| region | sum_revenue |",
                tool_call_id="toolu_1", name="group_aggregate",
            ),
        ],
        tools=tools,
        tool_choice="auto",
    )
    payload = provider._payload(request)
    assert payload["tools"] == tools
    assert payload["tool_choice"] == {"type": "auto"}
    assert payload["system"] == "sys"

    assistant = next(m for m in payload["messages"] if m["role"] == "assistant")
    assert assistant["content"][0]["type"] == "tool_use"
    assert assistant["content"][0]["name"] == "group_aggregate"
    assert assistant["content"][0]["input"]["group_by"] == "region"

    tool_msg = next(
        m for m in payload["messages"]
        if m["role"] == "user" and isinstance(m["content"], list)
    )
    assert tool_msg["content"][0]["type"] == "tool_result"
    assert tool_msg["content"][0]["tool_use_id"] == "toolu_1"


def test_openai_wire_shape_for_assistant_tool_calls():
    """The internal {id, name, arguments} shape must become OpenAI's
    {id, type: function, function: {name, arguments: json-string}} on the
    wire — sending the internal shape got a live HTTP 400 from Groq."""
    import json as _json

    from insight.llm.base import LLMRequest, Message
    from insight.llm.providers.openai_compat import OpenAICompatibleProvider

    provider = OpenAICompatibleProvider(api_key="test")
    request = LLMRequest(
        model="m",
        messages=[
            Message(role="user", content="Which region leads?"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "call_1", "name": "group_aggregate",
                 "arguments": {"group_by": "region"}},
            ]),
            Message(role="tool", content="| region |",
                    tool_call_id="call_1", name="group_aggregate"),
        ],
        tools=[{"type": "function", "function": {"name": "group_aggregate"}}],
        tool_choice="auto",
    )
    payload = provider._payload(request)
    call = payload["messages"][1]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["id"] == "call_1"
    assert call["function"]["name"] == "group_aggregate"
    assert _json.loads(call["function"]["arguments"]) == {"group_by": "region"}
    assert payload["messages"][2]["tool_call_id"] == "call_1"


def test_openai_second_turn_passes_a_strict_wire_validator():
    """End-to-end through the HTTP layer (MockTransport): the payload of a
    tool-result round trip is what a strict OpenAI-compatible gateway accepts."""
    import json as _json

    import httpx

    from insight.llm.base import LLMRequest, Message
    from insight.llm.providers.openai_compat import OpenAICompatibleProvider

    seen: dict = {}

    def handler(request):
        body = _json.loads(request.content)
        for message in body["messages"]:
            for call in message.get("tool_calls") or []:
                assert call["type"] == "function"
                assert isinstance(call["function"]["arguments"], str)
        seen["body"] = body
        return httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {"content": "West leads."}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        })

    class _Provider(OpenAICompatibleProvider):
        name = "strict-gateway"
        base_url = "https://strict.test/v1"

    provider = _Provider(api_key="k", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.generate(LLMRequest(
        model="m",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="", tool_calls=[
                {"id": "call_1", "name": "group_aggregate",
                 "arguments": {"group_by": "region"}},
            ]),
            Message(role="tool", content="| region |",
                    tool_call_id="call_1", name="group_aggregate"),
        ],
        tools=[{"type": "function", "function": {"name": "group_aggregate"}}],
        tool_choice="auto",
    )))
    assert response.content == "West leads."
    assert seen["body"]["messages"][1]["tool_calls"][0]["type"] == "function"


def test_openai_inbound_tool_call_parsing_variants():
    """The server->domain direction: wire tool_calls become internal
    {id, name, arguments} objects, tolerating string/dict/blank arguments and
    skipping nameless or unparseable entries."""
    import json as _json

    import httpx

    from insight.llm.base import LLMRequest
    from insight.llm.providers.openai_compat import OpenAICompatibleProvider

    def handler(request):
        return httpx.Response(200, json={
            "model": "m",
            "choices": [{"message": {
                "content": "",
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {
                        "name": "group_aggregate",
                        "arguments": _json.dumps({"group_by": "region"})}},
                    {"id": "2", "type": "function", "function": {
                        "name": "top_n", "arguments": {"n": 2}}},
                    {"id": "3", "type": "function", "function": {
                        "name": "value_counts", "arguments": "   "}},
                    {"id": "4", "type": "function", "function": {
                        "name": "broken_json", "arguments": "{oops"}},
                    {"id": "5", "type": "function", "function": {}},
                    {"id": "6", "type": "function", "function": {
                        "name": "numeric_args", "arguments": 123}},
                    {"id": "7", "type": "function", "function": {
                        "name": "detect_outliers<|channel|>commentary",
                        "arguments": {"column": "revenue"}}},
                    {"id": "8", "type": "function", "function": {
                        "name": "<|channel|>final", "arguments": {}}},
                ],
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    class _Provider(OpenAICompatibleProvider):
        name = "inbound-test"
        base_url = "https://inbound.test/v1"

    provider = _Provider(api_key="k", transport=httpx.MockTransport(handler))
    response = asyncio.run(provider.generate(LLMRequest(model="m", messages=[])))
    calls = response.tool_calls
    assert [c.name for c in calls] == [
        "group_aggregate", "top_n", "value_counts", "broken_json",
        "numeric_args", "detect_outliers",
    ]
    assert calls[0].arguments == {"group_by": "region"}
    assert calls[1].arguments == {"n": 2}
    assert calls[2].arguments == {}
    assert calls[3].arguments == {}
    assert calls[4].arguments == {}
    assert calls[5].arguments == {"column": "revenue"}


def test_openai_malformed_response_raises_provider_error():
    import httpx

    from insight.domain.errors import ProviderError
    from insight.llm.base import LLMRequest
    from insight.llm.providers.openai_compat import OpenAICompatibleProvider

    class _Provider(OpenAICompatibleProvider):
        name = "malformed-test"
        base_url = "https://malformed.test/v1"

    provider = _Provider(
        api_key="k",
        max_retries=1,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"oops": 1})
        ),
    )
    with pytest.raises(ProviderError, match="malformed response"):
        asyncio.run(provider.generate(LLMRequest(model="m", messages=[])))
