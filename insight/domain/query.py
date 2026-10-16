from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str


class AnalysisRequest(BaseModel):
    question: str
    history: list[ChatTurn] = Field(default_factory=list)
    model_id: Optional[str] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None
    # Human-in-the-loop: plan_only stops after planning and returns the plan for
    # review; approved_plan supplies an (optionally user-edited) plan to execute
    # instead of asking the planner again.
    plan_only: bool = False
    approved_plan: Optional[dict] = None
    # Opt-in agentic tool-use mode. None follows INSIGHT_AGENTIC (server
    # default); True/False forces the mode per question. Every tool call still
    # routes through the deterministic op registry and the same guards.
    agentic: Optional[bool] = None
    # Pin the run to exactly the requested model: disables the cross-provider
    # fallback chain, so a failure surfaces as an error instead of being
    # silently answered by a different model. Used by the evaluation runners
    # to make single-model measurements provable (models_used attribution).
    no_fallback: bool = False
