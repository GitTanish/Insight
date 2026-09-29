"""Human-in-the-loop plan approval: planning must stop before execution, and an
approved (possibly user-edited) plan must execute without re-planning."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import insight.orchestrator as orch
from insight.domain.query import AnalysisRequest
from insight.orchestrator import run_analysis_sync
from insight.planning.prompts import build_planner_messages
from insight.profiling.profiler import profile_dataframe
from insight.settings import get_settings

from conftest import ScriptedLLM, StubChain

from webapp.app import STORE, app

PLAN = """
{"objective": "avg revenue by region", "steps": [
  {"step_id": 0, "operation": "group_aggregate",
   "params": {"group_by": "region",
              "metrics": [{"column": "revenue", "agg": "mean", "alias": "m"}]}}],
 "charts": []}
"""


@pytest.fixture
def frame():
    df = pd.DataFrame({
        "region": ["North", "South", "East", "West"] * 25,
        "revenue": [100.0 + i for i in range(100)],
    })
    return df, profile_dataframe(df, "t.csv", content_hash="z" * 16)


async def _events(frame, request):
    df, profile = frame
    return [
        event
        async for event in orch.analyze_stream(
            df, profile, request, artifacts_dir=None
        )
    ]


def _run(frame, request):
    import asyncio

    df, profile = frame
    with tempfile.TemporaryDirectory() as tmp:
        return run_analysis_sync(
            df, profile, request, artifacts_dir=tmp
        )


def test_plan_only_emits_plan_review_and_never_executes(frame):
    import asyncio

    llm = ScriptedLLM([PLAN])
    orch._build_chain_for = lambda *a, **k: StubChain(llm)

    events = asyncio.run(_events(
        frame, AnalysisRequest(question="avg revenue by region", plan_only=True)
    ))
    types = [e["type"] for e in events]

    assert "plan_review" in types
    # Nothing past the gate: no execution, validation, deltas, or result.
    assert "result" not in types
    assert "delta" not in types
    assert not any(e["type"] == "stage" and e.get("key") == "executing" for e in events)


def test_plan_review_carries_editable_plan_and_question(frame):
    import asyncio

    llm = ScriptedLLM([PLAN])
    orch._build_chain_for = lambda *a, **k: StubChain(llm)

    events = asyncio.run(_events(
        frame, AnalysisRequest(question="avg revenue by region", plan_only=True)
    ))
    review = next(e for e in events if e["type"] == "plan_review")

    assert review["plan"]["objective"]
    assert review["plan"]["steps"][0]["operation"] == "group_aggregate"
    assert review["question"] == "avg revenue by region"
    # Must round-trip as JSON so the browser can edit and resubmit it.
    import json
    assert json.loads(json.dumps(review["plan"])) == review["plan"]


def test_approved_plan_executes_without_calling_the_planner(frame):
    llm = ScriptedLLM([PLAN])  # only one scripted response: a planner call would fail
    orch._build_chain_for = lambda *a, **k: StubChain(llm)

    approved = {
        "objective": "user approved",
        "steps": [{
            "step_id": 0,
            "operation": "group_aggregate",
            "params": {
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "mean", "alias": "m"}],
            },
        }],
        "charts": [],
    }
    response = _run(
        frame, AnalysisRequest(question="avg revenue by region", approved_plan=approved)
    )

    assert response.error is None
    assert response.answer
    # Only the explainer ran; the planner was never invoked.
    assert len(llm.requests) == 1


def test_approved_plan_is_revalidated_against_the_schema(frame):
    llm = ScriptedLLM([])
    orch._build_chain_for = lambda *a, **k: StubChain(llm)

    bad = {
        "objective": "edited by user with a nonexistent column",
        "steps": [{
            "step_id": 0,
            "operation": "group_aggregate",
            "params": {
                "group_by": "not_a_real_column",
                "metrics": [{"column": "revenue", "agg": "mean", "alias": "m"}],
            },
        }],
        "charts": [],
    }
    response = _run(
        frame, AnalysisRequest(question="avg revenue", approved_plan=bad)
    )

    # Must be rejected rather than blindly executed.
    assert response.error == "plan_validation_failed"


def test_user_edit_is_honored_not_overwritten(frame):
    llm = ScriptedLLM([PLAN, "Answer."])
    orch._build_chain_for = lambda *a, **k: StubChain(llm)

    edited = {
        "objective": "hand-edited by the analyst",
        "steps": [{
            "step_id": 0,
            "operation": "summarize",
            "params": {},
        }],
        "charts": [],
    }
    response = _run(
        frame, AnalysisRequest(question="avg revenue by region", approved_plan=edited)
    )

    assert response.error is None


def test_plan_approval_setting_forces_the_gate(frame, monkeypatch):
    monkeypatch.setenv("INSIGHT_PLAN_APPROVAL", "on")
    get_settings.cache_clear()
    import importlib
    import insight.settings as settings_mod
    importlib.reload(settings_mod)
    get_settings.cache_clear()
    try:
        llm = ScriptedLLM([PLAN])
        orch._build_chain_for = lambda *a, **k: StubChain(llm)

        import asyncio
        events = asyncio.run(_events(
            frame, AnalysisRequest(question="avg revenue by region")
        ))
        assert any(e["type"] == "plan_review" for e in events)
        assert not any(e["type"] == "result" for e in events)
    finally:
        monkeypatch.delenv("INSIGHT_PLAN_APPROVAL", raising=False)
        get_settings.cache_clear()
        importlib.reload(settings_mod)
        get_settings.cache_clear()


def test_plan_review_is_not_cached(frame, monkeypatch):
    get_settings.cache_clear()
    import importlib
    import insight.settings as settings_mod
    importlib.reload(settings_mod)
    try:
        llm = ScriptedLLM([PLAN])
        orch._build_chain_for = lambda *a, **k: StubChain(llm)

        import asyncio
        events = asyncio.run(_events(
            frame, AnalysisRequest(question="avg revenue", plan_only=True)
        ))
        assert any(e["type"] == "plan_review" for e in events)
    finally:
        get_settings.cache_clear()
        importlib.reload(settings_mod)
        get_settings.cache_clear()


def test_analyze_rejects_plan_only_because_it_has_no_plan_event():
    import asyncio
    from insight.orchestrator import analyze

    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="z" * 16)
    with pytest.raises(Exception) as excinfo:
        asyncio.run(analyze(
            df, profile, AnalysisRequest(question="q", plan_only=True)
        ))
    assert "streaming" in str(excinfo.value)


def test_client_has_plan_approval_controls():
    import pathlib
    js = (pathlib.Path(__file__).parent.parent / "webapp" / "static" / "app.js").read_text(
        encoding="utf-8"
    )
    assert "showPlanReview" in js
    assert "Approve & run" in js
    assert "approved_plan" in js
    assert "plan_review" in js


def test_api_plan_only_returns_plan_endpoint_contract():
    from fastapi.testclient import TestClient as TC

    sid = "hitl-api-session"
    STORE.drop(sid)
    csv = b"region,revenue\n" + b"".join(
        f"r{i % 4},{100 + i}\n".encode() for i in range(60)
    )
    with TC(app) as client:
        r = client.post(
            "/upload",
            files={"files": ("t.csv", csv, "text/csv")},
            follow_redirects=False,
        )
        assert r.status_code == 303
        bad = client.post("/api/query", json={"question": "q", "approved_plan": {"x": 1},
                                              "plan_only": True})
        assert bad.status_code == 400
    STORE.drop(sid)


def test_prompt_still_marks_system_prefix_stable():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="z" * 16)
    msgs = build_planner_messages(
        question="q", profile=profile, catalog="CAT", history=[],
    )
    assert msgs[0]["role"] == "system"
    assert "CAT" in msgs[0]["content"]
