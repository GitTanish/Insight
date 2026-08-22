import asyncio

import pytest
import json

from insight.domain.errors import PlanValidationError
from insight.llm.base import LLMRequest, LLMResponse
from insight.planning.planner import Planner


def _make_planner(contents: list[str], calls: list):
    async def generate(request: LLMRequest) -> LLMResponse:
        calls.append(request)
        return LLMResponse(
            content=contents[len(calls) - 1], provider="fake", model="fake-1",
            latency_ms=1,
        )

    return Planner(generate=generate, model_name="fake-1", max_repair_attempts=1)


BAD_COLUMN_PLAN = json.dumps({
    "objective": "t",
    "steps": [{"step_id": 0, "operation": "group_aggregate",
               "params": {"group_by": "banana",
                          "metrics": [{"column": "revenue", "agg": "sum"}]}}],
})

GOOD_PLAN = json.dumps({
    "objective": "t",
    "steps": [{"step_id": 0, "operation": "group_aggregate",
               "params": {"group_by": "region",
                          "metrics": [{"column": "revenue", "agg": "sum"}]}}],
})


def test_repair_recovers_from_unknown_column(profile):
    calls: list[LLMRequest] = []
    planner = _make_planner([BAD_COLUMN_PLAN, GOOD_PLAN], calls)
    plan = asyncio.run(planner.plan("group revenue", profile))

    assert plan.steps[0].params["group_by"] == "region"
    assert len(calls) == 2
    repair_message = calls[1].messages[-1].content
    assert "rejected by the validator" in repair_message
    assert "unknown column 'banana'" in repair_message


def test_exhausts_repairs_then_raises(profile):
    calls: list[LLMRequest] = []
    planner = _make_planner(["not json {", "still not json {"], calls)

    with pytest.raises(PlanValidationError):
        asyncio.run(planner.plan("anything", profile))
    assert len(calls) == 2


def test_json_mode_and_reasoning_requested(profile):
    calls: list[LLMRequest] = []
    good = GOOD_PLAN

    async def generate(request: LLMRequest) -> LLMResponse:
        calls.append(request)
        return LLMResponse(content=good, provider="fake", model="fake-1", latency_ms=1)

    planner = Planner(
        generate=generate, model_name="fake-1",
        max_repair_attempts=0, reasoning_effort="low",
    )
    plan = asyncio.run(planner.plan("q", profile))
    assert plan.objective == "t"
    assert calls[0].json_mode is True
    assert calls[0].reasoning_effort == "low"
