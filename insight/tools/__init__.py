"""Opt-in agentic tool-calling layer for Insight.

Everything here is gated behind ``INSIGHT_AGENTIC=on`` and is a strict superset
of the existing single-shot planner: the LLM still never executes free-form
code and never invents numbers, because every tool call routes through the same
deterministic op registry (``insight.analytics.operations``), the same column
validator, and the same grounding/recompute guards the base pipeline uses.

The module is deliberately split so each piece is independently testable with
no live provider:

- :mod:`insight.tools.tooling`  — op registry -> provider tool/function schemas
- :mod:`insight.tools.runtime`  — dispatch a tool call through the executor
- :mod:`insight.tools.loop`     — the bounded observe/plan/act/refine loop
"""

from __future__ import annotations
