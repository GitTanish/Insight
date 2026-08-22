import os

os.environ.setdefault("INSIGHT_TRACING", "off")
os.environ.setdefault("INSIGHT_QUERY_CACHE", "off")

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from insight.domain.dataset import DatasetProfile
from insight.llm.base import LLMResponse
from insight.profiling.profiler import profile_dataframe


@pytest.fixture
def sales_df() -> pd.DataFrame:
    n = 60
    return pd.DataFrame(
        {
            "order_id": range(1, n + 1),
            "date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
            "region": ["North", "South", "East", "West"] * (n // 4),
            "category": ["A", "B", "C"] * (n // 3),
            "revenue": [100 + ((i * 37) % 200) for i in range(n)],
            "units": [(i % 5) + 1 for i in range(n)],
        }
    )


@pytest.fixture
def profile(sales_df) -> DatasetProfile:
    return profile_dataframe(sales_df, "sales.csv", content_hash="deadbeef" * 4)


class ScriptedLLM:
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.requests: list = []

    async def generate(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("ScriptedLLM exhausted")
        content = self.responses.pop(0)
        return LLMResponse(
            content=content, provider="fake", model="fake-1", latency_ms=1
        )


class StubChain:
    def __init__(self, llm: ScriptedLLM):
        self.llm = llm
        self.temperature = 0.0
        self.reasoning_effort = None
        self.last_used = None
        self.tokens_in = 0
        self.tokens_out = 0
        self.llm_calls = 0
        self.llm_latency_ms = 0

    @property
    def primary_model(self) -> str:
        return "fake-1"

    async def generate(self, request):
        response = await self.llm.generate(request)
        self.last_used = "fake/fake-1"
        self.llm_calls += 1
        self.llm_latency_ms += response.latency_ms
        return response


@pytest.fixture
def stub_chain_factory(monkeypatch):
    def _install(responses: list[str]) -> ScriptedLLM:
        import insight.orchestrator as orch

        llm = ScriptedLLM(responses)
        chain = StubChain(llm)
        monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: chain)
        return llm

    return _install


VALID_PLAN = """
{
  "objective": "Revenue per region",
  "steps": [
    {"step_id": 0, "operation": "group_aggregate",
     "params": {"group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total_revenue"}],
                "sort_by": "total_revenue"},
     "reason": "aggregate"}
  ],
  "charts": [
    {"chart_type": "bar", "title": "Revenue by Region", "source_step": 0,
     "x": "region", "y": "total_revenue"}
  ]
}
"""

EXPLANATION = "**West leads** with the highest total revenue."
