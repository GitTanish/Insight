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
