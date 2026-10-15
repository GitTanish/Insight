import pytest

from insight.domain.errors import InsightError
from insight.llm.base import LLMRequest, LLMResponse, ToolCall
from insight.llm.registry import ModelInfo
from insight.orchestrator import _Chain


def _info(provider, model):
    return ModelInfo(
        model_id=f"{provider}/{model}", provider=provider, model=model, tier="fast"
    )


class _FakeProvider:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            content=item, provider=self.name, model="m", latency_ms=1
        )


class _ResponseProvider:
    """Like _FakeProvider, but the scripted items are complete LLMResponse
    objects (needed to script tool-call-only turns)."""

    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


REQUEST = LLMRequest(messages=[{"role": "user", "content": "hi"}], model="m")


@pytest.mark.asyncio
async def test_empty_content_falls_through_to_next_provider():
    empty_provider = _FakeProvider("flaky", [""])
    good_provider = _FakeProvider("groq", ["proper answer"])
    chain = _Chain(
        entries=[
            (empty_provider, "m1", _info("flaky", "m1")),
            (good_provider, "m2", _info("groq", "m2")),
        ],
        temperature=0.0,
        reasoning_effort=None,
    )

    response = await chain.generate(REQUEST)

    assert response.content == "proper answer"
    assert chain.last_used == "groq/m2"
    assert empty_provider.calls == 1 and good_provider.calls == 1


@pytest.mark.asyncio
async def test_tool_call_only_response_is_not_treated_as_empty():
    """Regression pin: the empty-content failover guard must exempt responses
    that carry tool calls (agentic mode), or every tool turn would fail over
    past the provider that actually answered."""
    tool_provider = _ResponseProvider("groq", [
        LLMResponse(
            content="", provider="groq", model="m", latency_ms=1,
            tool_calls=[ToolCall(id="t1", name="group_aggregate",
                                 arguments={"group_by": "region"})],
        )
    ])
    backup = _ResponseProvider("backup", ["should not be used"])
    chain = _Chain(
        entries=[
            (tool_provider, "m1", _info("groq", "m1")),
            (backup, "m2", _info("backup", "m2")),
        ],
        temperature=0.0,
        reasoning_effort=None,
    )

    response = await chain.generate(REQUEST)

    assert response.tool_calls and response.tool_calls[0].name == "group_aggregate"
    assert tool_provider.calls == 1 and backup.calls == 0


@pytest.mark.asyncio
async def test_all_empty_responses_raise_all_providers_failed():
    providers = [_FakeProvider("a", [""]), _FakeProvider("b", ["  "])]
    chain = _Chain(
        entries=[(p, f"m{i}", _info(p.name, f"m{i}")) for i, p in enumerate(providers)],
        temperature=0.0,
        reasoning_effort=None,
    )

    with pytest.raises(Exception):
        await chain.generate(REQUEST)
