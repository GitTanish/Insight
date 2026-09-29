"""Prefix-caching contract: the planner's system prefix must stay byte-identical
across turns so providers can serve it from their prompt cache."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from insight.analytics.operations import operation_catalog
from insight.domain.query import AnalysisRequest
from insight.orchestrator import run_analysis_sync
from insight.planning.prompts import build_planner_messages
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


def test_system_prefix_identical_across_turns():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="c" * 16)
    catalog = operation_catalog()

    a = build_planner_messages(
        question="first question", profile=profile, catalog=catalog,
        history=[], session_context=None, sql_schema=None,
    )
    b = build_planner_messages(
        question="totally different question", profile=profile, catalog=catalog,
        history=[("user", "earlier"), ("assistant", "earlier answer")],
        session_context="filters: region eq West", sql_schema=None,
    )

    assert a[0]["content"] == b[0]["content"], "system prefix must not vary per turn"
    assert a[0] == b[0]
    # The varying parts must live in later messages.
    joined_b = "\n".join(m["content"] for m in b)
    assert "totally different question" in joined_b
    assert "region eq West" in joined_b


def test_planner_requests_cache_prompt_flag():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="c" * 16)
    llm = ScriptedLLM([PLAN, "Answer."])
    chain = StubChain(llm)
    orch._build_chain_for = lambda *a, **k: chain

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_analysis_sync(
            df, profile, AnalysisRequest(question="avg revenue by region"),
            artifacts_dir=tmp,
        )

    assert llm.requests, "planner should have issued a request"
    assert llm.requests[0].cache_prompt is True


def test_meta_reports_cached_token_accounting():
    df = pd.DataFrame({"region": ["N", "S"] * 10, "revenue": [1.0, 2.0] * 10})
    profile = profile_dataframe(df, "t.csv", content_hash="c" * 16)
    llm = ScriptedLLM([PLAN, "Answer."])
    chain = StubChain(llm)
    chain.cached_tokens_in = 1234
    orch._build_chain_for = lambda *a, **k: chain

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        response = run_analysis_sync(
            df, profile, AnalysisRequest(question="avg revenue by region"),
            artifacts_dir=tmp,
        )

    assert response.meta.get("cached_prompt_tokens") == 1234
    assert "cached_prompt_tokens" in response.meta
