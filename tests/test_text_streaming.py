"""Streaming contract: the explainer must emit incremental `delta` events and
the client must render them into a live typing bubble."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from insight.domain.query import AnalysisRequest
from insight.orchestrator import run_analysis_sync
from insight.profiling.profiler import profile_dataframe

from conftest import ScriptedLLM, StubChain

import insight.orchestrator as orch

PLAN = """
{"objective": "t", "steps": [
  {"step_id": 0, "operation": "group_aggregate",
   "params": {"group_by": "region",
              "metrics": [{"column": "revenue", "agg": "mean", "alias": "m"}]}}],
 "charts": []}
"""


async def _collect_events(agen):
    return [event async for event in agen]


def test_explainer_emits_incremental_deltas():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="c" * 16)
    llm = ScriptedLLM([PLAN, "First chunk text. Second chunk text."])
    chain = StubChain(llm)
    orch._build_chain_for = lambda *a, **k: chain

    import asyncio
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        events = asyncio.run(_collect_events(orch.analyze_stream(
            df, profile, AnalysisRequest(question="avg revenue by region"),
            artifacts_dir=tmp,
        )))

    deltas = [e for e in events if e.get("type") == "delta"]
    assert len(deltas) >= 2, f"expected streamed chunks, got {len(deltas)}"
    # Deltas must arrive before the final result event.
    result_index = next(i for i, e in enumerate(events) if e.get("type") == "result")
    assert all(events.index(e) < result_index for e in deltas)
    joined = "".join(d["text"] for d in deltas)
    assert "First chunk text. Second chunk text." in joined


def test_first_token_ms_reported():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="c" * 16)
    llm = ScriptedLLM([PLAN, "Streamed answer."])
    chain = StubChain(llm)
    orch._build_chain_for = lambda *a, **k: chain

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        response = run_analysis_sync(
            df, profile, AnalysisRequest(question="avg revenue by region"),
            artifacts_dir=tmp,
        )

    assert response.meta.get("first_token_ms") is not None
    assert response.meta["first_token_ms"] >= 0


def test_client_renders_live_typing_bubble():
    app_js = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "webapp", "static", "app.js"),
        encoding="utf-8",
    ).read()
    assert "delta" in app_js
    assert "appendDelta" in app_js
    assert "delta_reset" in app_js
    assert "clearLiveBubble" in app_js
    # The live bubble must be replaced by the authoritative final render.
    assert app_js.index("clearLiveBubble()") < app_js.index('event === "result"')


def test_streaming_caret_is_accessible_and_respects_reduced_motion():
    css = open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "webapp", "static", "style.css"),
        encoding="utf-8",
    ).read()
    assert ".answer-md.streaming" in css
    reduced = css[css.rindex("@media (prefers-reduced-motion: reduce)"):]
    assert ".answer-md.streaming" in reduced
    assert "animation: none" in reduced
