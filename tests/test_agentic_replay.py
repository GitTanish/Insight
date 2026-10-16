"""Replay a recorded live agentic run offline.

The fixture is a trimmed transcript of a real pinned run on NVIDIA NIM
(openai/gpt-oss-20b): one tool call, then the final answer. Replaying the
exact recorded LLMResponses through the real loop proves the recorded tool
path is reproducible offline — the loop, tool dispatch and operation
execution all run for real; only the provider is replayed.

Regenerate after changes that legitimately alter the recorded behaviour:
    python evaluation/run_eval_agentic.py --model <id> --record --limit ...
then re-extract the case's exchanges into tests/fixtures/ (request bodies
are trimmed; the replay asserts the tool path, not byte-identical prompts).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "agentic_replay_sales_top_region.json"
)
DATA = ROOT / "evaluation" / "data" / "sales.csv"

pytestmark = pytest.mark.skipif(
    not DATA.exists(),
    reason="evaluation datasets not generated (python evaluation/generate_datasets.py)",
)


def test_replayed_live_run_reproduces_the_recorded_tool_path():
    from insight.analytics.operations import operation_catalog
    from insight.llm.base import LLMResponse
    from insight.planning.prompts import AGENTIC_SYSTEM, build_agentic_messages
    from insight.profiling.profiler import profile_dataframe
    from insight.tools.loop import LoopBudget, run_loop

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    responses = [LLMResponse(**r) for r in fixture["responses"]]

    class _ReplayProvider:
        def __init__(self):
            self.calls = 0

        async def generate(self, request):
            assert self.calls < len(responses), (
                "loop requested more provider calls than the fixture recorded"
            )
            response = responses[self.calls]
            self.calls += 1
            return response

    df = pd.read_csv(DATA)
    question = fixture["question"].removeprefix("Question: ")
    profile = profile_dataframe(
        df, "sales.csv", content_hash=f"eval{'sales.csv'}{len(df)}"[:32]
    )
    initial_messages = build_agentic_messages(
        question, profile, operation_catalog(), sql_schema=None
    )

    provider = _ReplayProvider()
    outcome = asyncio.run(
        run_loop(
            question, df, profile, provider.generate, "openai/gpt-oss-20b",
            duck_ctx=None,
            budget=LoopBudget(max_iterations=5, max_llm_calls=9),
            system_prompt=AGENTIC_SYSTEM,
            initial_messages=initial_messages,
        )
    )

    assert provider.calls == fixture["expected_llm_calls"]
    assert outcome.operations_used == fixture["expected_operations"]
    assert fixture["final_answer_excerpt"] in outcome.final_answer
    assert outcome.error is None
