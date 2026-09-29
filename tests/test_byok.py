import io
import os

import pytest

from insight.llm.registry import create_provider
from insight.settings import get_settings


def test_create_provider_uses_user_key_override():
    provider = create_provider("groq", get_settings(), api_key="sk-user-key")
    assert provider._api_key == "sk-user-key"


def test_create_provider_without_override_falls_back_to_env():
    settings = get_settings()
    if not settings.groq_api_key:
        pytest.skip("no server GROQ key configured")
    provider = create_provider("groq", settings)
    assert provider._api_key == settings.groq_api_key


@pytest.fixture
def byok_mode(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "require_user_key", True)
    yield


class _RecordingChain:
    temperature = 0.0
    reasoning_effort = None
    last_used = "fake/fake-1"
    tokens_in = 0
    tokens_out = 0
    cached_tokens_in = 0
    llm_calls = 0
    llm_latency_ms = 0

    def __init__(self):
        self.calls = 0
        self.kwargs = {}

    async def stream(self, request):
        response = await self.generate(request)
        if response.content:
            yield response.content

    @property
    def primary_model(self):
        return "fake-1"

    async def generate(self, request):
        from insight.llm.base import LLMResponse

        self.calls += 1
        return LLMResponse(content="**ok**", provider="fake", model="fake-1", latency_ms=1)


CSV = (
    b"region,revenue\nNorth,10\nSouth,20\nEast,30\nWest,40\n"
)


@pytest.fixture
def recording_client(monkeypatch):
    os.environ.setdefault("INSIGHT_TRACING", "off")
    from fastapi.testclient import TestClient
    from webapp.app import app
    import insight.orchestrator as orch

    chain = _RecordingChain()

    def factory(*args, **kwargs):
        chain.kwargs = kwargs
        return chain

    monkeypatch.setattr(orch, "_build_chain_for", factory)
    with TestClient(app) as c:
        c.post(
            "/upload",
            files={"file": ("t.csv", io.BytesIO(CSV), "text/csv")},
            follow_redirects=False,
        )
        yield c, chain


PLAN = """
{"objective":"o","steps":[{"step_id":0,"operation":"value_counts",
"params":{"column":"region"}}],"charts":[]}
"""


class _PlanExplainChain(_RecordingChain):
    async def generate(self, request):
        from insight.llm.base import LLMResponse

        self.calls += 1
        content = PLAN if self.calls % 2 == 1 else "**done**"
        return LLMResponse(content=content, provider="fake", model="fake-1", latency_ms=1)


def test_query_passes_user_key_header_to_chain(recording_client, monkeypatch):
    client, chain = recording_client
    plan_chain = _PlanExplainChain()
    plan_chain.kwargs = chain.kwargs
    import insight.orchestrator as orch

    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: plan_chain)

    r = client.post(
        "/query",
        data={"question": "counts"},
        headers={"X-Insight-Key": "sk-visitor"},
    )
    assert r.status_code == 200

    recorder = _PlanExplainChain()

    def factory(*a, **kwargs):
        recorder.kwargs = kwargs
        return recorder

    monkeypatch.setattr(orch, "_build_chain_for", factory)

    r = client.post(
        "/query",
        data={"question": "counts again"},
        headers={"X-Insight-Key": "sk-visitor"},
    )
    assert r.status_code == 200
    assert recorder.kwargs.get("api_key") == "sk-visitor"


def test_require_user_key_blocks_missing_header(recording_client, monkeypatch):
    client, _ = recording_client
    settings = get_settings()
    monkeypatch.setattr(settings, "require_user_key", True)

    blocked = client.post("/query", data={"question": "counts"})
    assert blocked.status_code == 401
    assert "bring-your-own-key" in blocked.json()["error"]

    api_blocked = client.post("/api/query", json={"question": "counts"})
    assert api_blocked.status_code == 401


def test_require_user_key_allows_header_through(recording_client, monkeypatch):
    client, chain = recording_client
    settings = get_settings()
    monkeypatch.setattr(settings, "require_user_key", True)

    import insight.orchestrator as orch

    plan_chain = _PlanExplainChain()

    def factory(*a, **kwargs):
        plan_chain.kwargs = kwargs
        return plan_chain

    monkeypatch.setattr(orch, "_build_chain_for", factory)

    r = client.post(
        "/query",
        data={"question": "counts"},
        headers={"X-Insight-Key": "sk-visitor"},
    )
    assert r.status_code == 200
