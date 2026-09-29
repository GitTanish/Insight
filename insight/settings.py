from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class Settings(BaseModel):
    groq_api_key: str | None = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY") or None
    )
    mistral_api_key: str | None = Field(
        default_factory=lambda: os.getenv("MISTRAL_API_KEY") or None
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY") or None
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY") or None
    )
    openrouter_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY") or None
    )
    ollama_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL") or None
    )
    enable_ollama: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_ENABLE_OLLAMA", "").lower()
        in {"1", "true", "yes"}
    )
    discovery_timeout_s: float = 8.0
    discovery_max_per_provider: int = 40
    default_model_id: str = "groq/openai/gpt-oss-120b"
    fast_model_id: str = "groq/openai/gpt-oss-20b"
    temperature: float = 0.0
    reasoning_effort: str = "low"
    request_timeout_s: float = 60.0
    max_retries_per_call: int = 3
    planner_max_tokens: int = 3000
    explainer_max_tokens: int = 1500
    max_repair_attempts: int = 1
    # When on, every question stops at a reviewable plan before execution.
    plan_approval: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_PLAN_APPROVAL", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    require_user_key: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_REQUIRE_USER_KEY", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    answer_grounded_check: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_ANSWER_GROUNDING", "on")
        .strip()
        .lower()
        not in {"0", "off", "false", "no"}
    )
    sql_enabled: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_SQL", "on").strip().lower()
        not in {"0", "off", "false", "no"}
    )
    sql_timeout_s: float = Field(
        default_factory=lambda: float(os.getenv("INSIGHT_SQL_TIMEOUT_S", "10") or 10)
    )
    sql_max_output_rows: int = Field(
        default_factory=lambda: int(os.getenv("INSIGHT_SQL_MAX_ROWS", "10000") or 10000)
    )
    query_cache_enabled: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_QUERY_CACHE", "on")
        .strip()
        .lower()
        not in {"0", "off", "false", "no"}
    )
    query_cache_ttl_days: int = Field(
        default_factory=lambda: int(os.getenv("INSIGHT_QUERY_CACHE_TTL_DAYS", "14") or 14)
    )
    query_cache_max_mb: int = Field(
        default_factory=lambda: int(os.getenv("INSIGHT_QUERY_CACHE_MAX_MB", "512") or 512)
    )
    briefing_llm_polish: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_BRIEFING_LLM_POLISH", "")
        .strip()
        .lower()
        in {"1", "true", "yes", "on"}
    )
    max_upload_rows: int = 1_000_000
    max_table_rows: int = 100
    artifacts_dir: Path = ROOT_DIR / ".artifacts"
    custom_base_url: str | None = Field(
        default_factory=lambda: os.getenv("INSIGHT_CUSTOM_BASE_URL") or None
    )
    custom_api_key: str | None = Field(
        default_factory=lambda: os.getenv("INSIGHT_CUSTOM_API_KEY") or None
    )
    custom_models: str | None = Field(
        default_factory=lambda: os.getenv("INSIGHT_CUSTOM_MODELS") or None
    )
    custom_name: str = Field(
        default_factory=lambda: os.getenv("INSIGHT_CUSTOM_NAME", "custom")
    )
    custom_discovery: bool = Field(
        default_factory=lambda: os.getenv("INSIGHT_CUSTOM_DISCOVERY", "on")
        .strip()
        .lower()
        not in {"0", "off", "false", "no"}
    )
    langsmith_api_key: str | None = Field(
        default_factory=lambda: os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
        or None
    )
    langsmith_project: str | None = Field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "insight"
    )
    tracing_enabled: bool = Field(
        default_factory=lambda: bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
