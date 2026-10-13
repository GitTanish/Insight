"""Offline webapp tests for agentic (tool-calling) mode over SSE and the JSON API."""
import io
import json
import os

import pytest
from fastapi.testclient import TestClient

CSV_BYTES = (
    b"order_id,date,region,category,revenue,units\n"
    b"1,2025-01-01,North,A,100,1\n"
    b"2,2025-01-02,South,B,220,2\n"
    b"3,2025-01-03,East,C,150,3\n"
    b"4,2025-01-04,West,A,300,4\n"
    b"5,2025-01-05,North,B,120,2\n"
    b"6,2025-01-06,South,C,900,9\n"
    b"7,2025-01-07,East,A,110,1\n"
    b"8,2025-01-08,West,B,240,3\n"
    b"9,2025-01-09,North,C,160,2\n"
    b"10,2025-01-10,South,A,310,4\n"
)

PLAN = """
{"objective":"Revenue by region","steps":[
 {"step_id":0,"operation":"group_aggregate","params":{"group_by":"region",
  "metrics":[{"column":"revenue","agg":"sum","alias":"total_revenue"}],"sort_by":"total_revenue"}}],
 "charts":[{"chart_type":"bar","title":"Revenue by Region","source_step":0,"x":"region","y":"total_revenue"}]}
"""


class AgentWebStubChain:
    """Stub chain that speaks the tool-calling protocol when tools are attached."""

    temperature = 0.0
    reasoning_effort = None
    last_used = None
    tokens_in = 0
    tokens_out = 0
    cached_tokens_in = 0
    llm_calls = 0
    llm_latency_ms = 0

    def __init__(self):
        self._tool_sent = False

    @property
    def primary_model(self):
        return "fake-1"

    async def generate(self, request):
        from insight.llm.base import LLMResponse, ToolCall

        self.llm_calls += 1
        self.last_used = "fake/fake-1"
        if request.tools:
            if not self._tool_sent:
                self._tool_sent = True
                return LLMResponse(
                    content="", provider="fake", model="fake-1", latency_ms=1,
                    tool_calls=[ToolCall(
                        id="c1", name="group_aggregate",
                        arguments={
                            "group_by": "region",
                            "metrics": [{"column": "revenue", "agg": "sum",
                                         "alias": "total_revenue"}],
                        },
                    )],
                )
            return LLMResponse(
                content="West leads with the highest total revenue.",
                provider="fake", model="fake-1", latency_ms=1,
            )
        return LLMResponse(
            content=PLAN, provider="fake", model="fake-1", latency_ms=1
        )

    async def stream(self, request):
        text = "West leads revenue."
        self.llm_calls += 1
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        yield text[midpoint:]


@pytest.fixture
def client(monkeypatch):
    os.environ.setdefault("INSIGHT_TRACING", "off")
    from webapp.app import app
    import insight.orchestrator as orch

    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: AgentWebStubChain())
    with TestClient(app) as c:
        yield c


def parse_sse(body: str):
    events = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = line[5:].strip()
        if ev:
            try:
                events.append((ev, json.loads(data)))
            except Exception:
                events.append((ev, data))
    return events


def _upload(client):
    r = client.post(
        "/upload",
        files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 303


def test_query_streams_agent_tool_trail(client):
    _upload(client)
    q = client.post(
        "/query",
        data={"question": "Which region has the highest revenue?", "agentic": "1"},
    )
    assert q.status_code == 200
    events = parse_sse(q.text)
    types = [e for e, _ in events]
    assert "tool_call" in types and "tool_result" in types

    tool_call = next(d for e, d in events if e == "tool_call")
    assert tool_call["name"] == "group_aggregate"
    assert tool_call["arguments"]["group_by"] == "region"
    tool_result = next(d for e, d in events if e == "tool_result")
    assert tool_result["ok"] is True

    result = next(d for e, d in events if e == "result")
    assert "mode" in result["html"] and "agent" in result["html"]
    assert "call(s)" in result["html"]  # agentic telemetry chip


def test_query_agentic_off_overrides_server_default(client):
    _upload(client)
    q = client.post(
        "/query",
        data={"question": "Which region has the highest revenue?", "agentic": "0"},
    )
    assert q.status_code == 200
    types = [e for e, _ in parse_sse(q.text)]
    assert "tool_call" not in types
    assert "result" in types


def test_api_query_accepts_agentic_flag(client):
    _upload(client)
    r = client.post(
        "/api/query",
        json={"question": "Which region has the highest revenue?", "agentic": True},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["meta"]["agentic"] is True
    assert body["meta"]["agentic_llm_calls"] >= 2
    assert body["meta"]["steps"][0]["operation"] == "group_aggregate"


def test_api_query_planner_mode_omits_agentic_meta(client):
    _upload(client)
    r = client.post(
        "/api/query",
        json={"question": "Which region has the highest revenue?"},
    )
    assert r.status_code == 200
    assert not r.json()["meta"].get("agentic")
