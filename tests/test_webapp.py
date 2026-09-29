import io
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


CSV_BYTES = (
    b"order_id,date,region,category,revenue,units,treated\n"
    b"1,2025-01-01,North,A,100,1,0\n"
    b"2,2025-01-02,South,B,220,2,0\n"
    b"3,2025-01-03,East,C,150,3,1\n"
    b"4,2025-01-04,West,A,300,4,1\n"
    b"5,2025-01-05,North,B,120,2,0\n"
    b"6,2025-01-06,South,C,900,9,1\n"
    b"7,2025-01-07,East,A,110,1,0\n"
    b"8,2025-01-08,West,B,240,3,0\n"
    b"9,2025-01-09,North,C,160,2,1\n"
    b"10,2025-01-10,South,A,310,4,1\n"
    b"11,2025-01-11,East,B,130,2,0\n"
    b"12,2025-01-12,West,C,250,3,0\n"
    b"13,2025-01-13,North,A,105,1,0\n"
    b"14,2025-01-14,South,B,225,2,0\n"
    b"15,2025-01-15,East,C,155,3,1\n"
    b"16,2025-01-16,West,A,305,4,1\n"
)

PLAN = """
{"objective":"Revenue by region","steps":[
 {"step_id":0,"operation":"group_aggregate","params":{"group_by":"region",
  "metrics":[{"column":"revenue","agg":"sum","alias":"total_revenue"}],"sort_by":"total_revenue"}}],
 "charts":[{"chart_type":"bar","title":"Revenue by Region","source_step":0,"x":"region","y":"total_revenue"}]}
"""
EXPLAIN = "**West leads revenue.**"


class AlternateStubChain:
    temperature = 0.0
    reasoning_effort = None
    last_used = None
    tokens_in = 0
    tokens_out = 0
    cached_tokens_in = 0
    llm_calls = 0
    llm_latency_ms = 0

    def __init__(self):
        self.calls = 0

    @property
    def primary_model(self):
        return "fake-1"

    async def generate(self, request):
        from insight.llm.base import LLMResponse

        self.calls += 1
        content = PLAN if self.calls % 2 == 1 else EXPLAIN
        self.last_used = "fake/fake-1"
        return LLMResponse(content=content, provider="fake", model="fake-1", latency_ms=1)

    async def stream(self, request):
        response = await self.generate(request)
        text = response.content or ""
        if not text:
            return
        midpoint = max(1, len(text) // 2)
        yield text[:midpoint]
        yield text[midpoint:]


@pytest.fixture
def client(monkeypatch):
    os.environ.setdefault("INSIGHT_TRACING", "off")
    from webapp.app import app
    import insight.orchestrator as orch

    stub = AlternateStubChain()
    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: stub)
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


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_index_sets_cookie_and_masthead(client):
    r = client.get("/")
    assert r.status_code == 200
    assert ("THE INSIGHT" in r.text) or ("logo" in r.text)
    assert "insight_sid" in r.cookies or "insight_sid" in client.cookies.get_dict()


def test_full_flow_upload_query_export_clear(client):
    r = client.post(
        "/upload",
        files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 303

    page = client.get("/")
    assert "sales.csv loaded" in page.text
    assert "Top Findings" in page.text or "briefing" in page.text.lower()

    q = client.post("/query", data={"question": "Which region has the highest revenue?"})
    assert q.status_code == 200
    events = parse_sse(q.text)
    types = [e for e, _ in events]
    assert "stage" in types and "result" in types and "end" in types
    result_payload = next(d for e, d in events if e == "result")
    assert "West leads revenue" in result_payload["html"]
    assert "Suggested next" in result_payload["html"]

    followup_events = [d for e, d in events if e == "suggestions"]
    assert followup_events and followup_events[-1]["followups"]

    page = client.get("/")
    import re as _re

    img_srcs = _re.findall(r'src="/artifacts/([^"]+)/([^"]+)"', page.text)
    assert img_srcs, "expected at least one chart image"
    for data_id, fname in img_srcs:
        assert ":" not in fname, f"absolute path leaked into URL: {fname}"
        assert "\\" not in fname and "/" not in fname, f"non-basename in URL: {fname}"
        served = client.get(f"/artifacts/{data_id}/{fname}")
        assert served.status_code == 200, f"chart URL 404s: {fname}"

    page = client.get("/")
    assert "West leads revenue." in page.text
    assert "View calculation" in page.text

    docx = client.get("/export/docx")
    assert docx.status_code == 200 and docx.content[:2] == b"PK"

    from docx import Document as _Doc

    doc = _Doc(io.BytesIO(docx.content))
    paragraphs = "\n".join(p.text for p in doc.paragraphs)
    assert "Which region has the highest revenue?" in paragraphs
    assert "Top Findings" in paragraphs

    second_q = client.post(
        "/query", data={"question": "Show me a visualization"}
    )
    assert second_q.status_code == 200
    stacked = client.get("/export/docx")
    stacked_doc = _Doc(io.BytesIO(stacked.content))
    stacked_text = "\n".join(p.text for p in stacked_doc.paragraphs)
    assert "Which region has the highest revenue?" in stacked_text
    assert "Show me a visualization" in stacked_text

    custom = client.post("/export/docx/custom", data={"turn": "1"})
    custom_doc = _Doc(io.BytesIO(custom.content))
    custom_text = "\n".join(p.text for p in custom_doc.paragraphs)
    assert "Show me a visualization" in custom_text
    assert "Which region has the highest revenue?" not in custom_text

    none_selected = client.post("/export/docx/custom", data={}, follow_redirects=False)
    assert none_selected.status_code == 303

    api = client.post("/api/query", json={"question": "top region"})
    assert api.status_code == 200
    assert api.json()["answer"].startswith("**West leads")

    cleared = client.post("/clear", follow_redirects=False)
    assert cleared.status_code == 303
    assert "West leads revenue" not in client.get("/").text


def test_query_without_dataset_rejected(client):
    r = client.post("/query", data={"question": "anything"}, )
    assert r.status_code == 400


def test_docx_button_always_visible_and_friendly_redirect(client):
    page = client.get("/")
    assert "Export DOCX" in page.text

    from webapp.app import app as webapp_app

    fresh = TestClient(webapp_app)
    r = fresh.get("/export/docx", follow_redirects=False)
    assert r.status_code == 303
    assert "flash=" in r.headers["location"]


def test_streaming_ui_contract_present(client):
    page = client.get("/")
    assert 'id="stop-btn"' in page.text
    assert 'id="pipeline-status"' in page.text
    assert 'id="composer"' in page.text

    js = (Path(__file__).resolve().parent.parent / "webapp" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    # stage rail covers every server-emitted stage key
    for key in ("planning", "executing", "repairing", "rendering", "explaining", "grounding"):
        assert f"{key}:" in js, f"stage key {key} unmapped in app.js"
    assert "validating" in js
    # abort support and a non-auto-hiding error surface
    assert "AbortController" in js
    assert "setError" in js
    assert "setTimeout(hideStatus" not in js

    css = (Path(__file__).resolve().parent.parent / "webapp" / "static" / "style.css").read_text(
        encoding="utf-8"
    )
    assert ".stage-rail" in css and ".btn.stop" in css
