import pytest

from insight.domain.errors import AllProvidersFailedError, InsightError
from insight.llm.base import LLMRequest, LLMResponse, ToolCall
from insight.llm.registry import ModelInfo
from insight.orchestrator import _Chain


def _info(provider, model, **overrides):
    return ModelInfo(
        model_id=f"{provider}/{model}", provider=provider, model=model, tier="fast",
        **overrides,
    )


class _FakeProvider:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = 0
        self.requests: list = []

    async def generate(self, request):
        self.calls += 1
        self.requests.append(request)
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
        self.requests: list = []

    async def generate(self, request):
        self.calls += 1
        self.requests.append(request)
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

    with pytest.raises(AllProvidersFailedError) as excinfo:
        await chain.generate(REQUEST)

    assert "returned empty content" in str(excinfo.value)


@pytest.mark.asyncio
async def test_empty_content_error_reports_exhausted_budget():
    """The 'empty content' error must surface the truncation signal: NIM's
    gpt-oss-20b burned an entire max_tokens budget in the reasoning channel and
    returned finish_reason=length with content='', which surfaced as a bare
    'returned empty content' and cost hours of diagnosis."""
    provider = _ResponseProvider("nvidia", [
        LLMResponse(
            content="", provider="nvidia", model="openai/gpt-oss-20b",
            latency_ms=1, completion_tokens=3000,
        )
    ])
    chain = _Chain(
        entries=[(provider, "openai/gpt-oss-20b", _info("nvidia", "openai/gpt-oss-20b"))],
        temperature=0.0,
        reasoning_effort=None,
    )
    request = REQUEST.model_copy(update={"max_tokens": 3000})

    with pytest.raises(AllProvidersFailedError) as excinfo:
        await chain.generate(request)

    message = str(excinfo.value)
    assert "returned empty content" in message
    assert "hit max_tokens=3000" in message


@pytest.mark.asyncio
async def test_reasoning_effort_stripped_when_model_does_not_support_it():
    provider = _FakeProvider("plain", ["ok"])
    chain = _Chain(
        entries=[(provider, "m", _info("plain", "m"))],
        temperature=0.0,
        reasoning_effort="low",
    )
    request = REQUEST.model_copy(update={"reasoning_effort": "low"})

    await chain.generate(request)

    assert provider.requests[0].reasoning_effort is None


@pytest.mark.asyncio
async def test_reasoning_effort_passes_through_when_model_supports_it():
    provider = _FakeProvider("oss", ["ok"])
    chain = _Chain(
        entries=[
            (provider, "m", _info("oss", "m", supports_reasoning_effort=True))
        ],
        temperature=0.0,
        reasoning_effort="low",
    )
    request = REQUEST.model_copy(update={"reasoning_effort": "low"})

    await chain.generate(request)

    assert provider.requests[0].reasoning_effort == "low"


@pytest.mark.asyncio
async def test_provider_throttle_fails_over_before_first_token():
    from insight.domain.errors import RateLimitError

    throttled = _FakeProvider(
        "limited", [RateLimitError("limited", "m1", "429", retry_after_s=2)]
    )
    healthy = _FakeProvider("healthy", ["served"])
    chain = _Chain(
        entries=[
            (throttled, "m1", _info("limited", "m1")),
            (healthy, "m2", _info("healthy", "m2")),
        ],
        temperature=0.0,
        reasoning_effort=None,
    )

    response = await chain.generate(REQUEST)

    assert response.content == "served"
    assert chain.last_used == "healthy/m2"
    assert throttled.calls == 1 and healthy.calls == 1


@pytest.mark.asyncio
async def test_all_providers_down_raises_typed_error_with_causes():
    from insight.domain.errors import ProviderError

    first = _FakeProvider("a", [ProviderError("a", "m1", "boom", status_code=500)])
    second = _FakeProvider("b", [ProviderError("b", "m2", "down", status_code=503)])
    chain = _Chain(
        entries=[
            (first, "m1", _info("a", "m1")),
            (second, "m2", _info("b", "m2")),
        ],
        temperature=0.0,
        reasoning_effort=None,
    )

    with pytest.raises(AllProvidersFailedError) as excinfo:
        await chain.generate(REQUEST)

    assert len(excinfo.value.errors) == 2
    assert "boom" in str(excinfo.value)
    assert "down" in str(excinfo.value)
