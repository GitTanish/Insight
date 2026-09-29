from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from insight.domain.errors import ConfigError
from insight.llm.base import LLMProvider
from insight.llm.providers.anthropic import AnthropicProvider
from insight.llm.providers.groq import GroqProvider
from insight.llm.providers.mistral import MistralProvider
from insight.llm.providers.ollama import OllamaProvider
from insight.llm.providers.openai import OpenAIProvider
from insight.llm.providers.openai_compat import OpenAICompatibleProvider
from insight.llm.providers.openrouter import OpenRouterProvider
from insight.settings import Settings, get_settings

ModelTier = Literal["fast", "balanced", "accurate", "discovered"]


class ModelInfo(BaseModel):
    model_id: str
    provider: str
    model: str
    tier: ModelTier = "balanced"
    supports_json_mode: bool = True
    supports_reasoning_effort: bool = False
    discovered: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    cls: type[LLMProvider]
    api_key_env: str | None
    curated_models: tuple[tuple[str, str, dict], ...] = ()
    supports_discovery: bool = False
    requires_key: bool = True
    default_timeout_s: float = 60.0
    settings_key_attr: str | None = None


PROVIDERS: dict[str, ProviderSpec] = {
    "groq": ProviderSpec(
        name="groq",
        cls=GroqProvider,
        api_key_env="GROQ_API_KEY",
        curated_models=(
            ("openai/gpt-oss-120b", "accurate",
             {"supports_json_mode": True, "supports_reasoning_effort": True}),
            ("openai/gpt-oss-20b", "fast",
             {"supports_json_mode": True, "supports_reasoning_effort": True}),
            ("qwen/qwen3.8-27b", "balanced", {}),
        ),
    ),
    "mistral": ProviderSpec(
        name="mistral",
        cls=MistralProvider,
        api_key_env="MISTRAL_API_KEY",
        curated_models=(
            ("mistral-medium-latest", "accurate", {}),
            ("mistral-small-latest", "balanced", {}),
        ),
    ),
    "openai": ProviderSpec(
        name="openai",
        cls=OpenAIProvider,
        api_key_env="OPENAI_API_KEY",
        curated_models=(
            ("gpt-4o-mini", "fast", {}),
            ("gpt-4o", "accurate", {}),
        ),
        supports_discovery=True,
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        cls=AnthropicProvider,
        api_key_env="ANTHROPIC_API_KEY",
        curated_models=(
            ("claude-haiku-4-5", "fast", {}),
            ("claude-sonnet-4-5", "accurate", {}),
        ),
        supports_discovery=True,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        cls=OpenRouterProvider,
        api_key_env="OPENROUTER_API_KEY",
        curated_models=(
            ("openai/gpt-4o-mini", "fast", {}),
        ),
        supports_discovery=True,
        default_timeout_s=90.0,
    ),
    "ollama": ProviderSpec(
        name="ollama",
        cls=OllamaProvider,
        api_key_env=None,
        requires_key=False,
        curated_models=(),
        supports_discovery=True,
        default_timeout_s=120.0,
    ),
}

_NON_TEXT_KEYWORDS = (
    "embed", "embedding", "whisper", "tts", "audio", "guard", "safeguard",
    "moderation", "dall-e", "image", "vision-encode", "transcribe",
    "devstral-embed", "bge", "voyage", "orpheus",
)

MODEL_CATALOG: dict[str, ModelInfo] = {}
DISCOVERED: dict[str, ModelInfo] = {}


def _build_static_catalog() -> None:
    MODEL_CATALOG.clear()
    for spec in PROVIDERS.values():
        for model, tier, extra in spec.curated_models:
            model_id = f"{spec.name}/{model}"
            MODEL_CATALOG[model_id] = ModelInfo(
                model_id=model_id,
                provider=spec.name,
                model=model,
                tier=tier,  # type: ignore[arg-type]
                **extra,
            )


_build_static_catalog()


def _ensure_custom_registered(settings: Settings) -> None:
    """Register a user-defined OpenAI-compatible provider from env vars.

    Requires INSIGHT_CUSTOM_BASE_URL, INSIGHT_CUSTOM_API_KEY and
    (optionally) INSIGHT_CUSTOM_MODELS as a comma-separated list. When the
    model list is omitted, discovery via GET {base_url}/models is attempted.
    """
    name = (settings.custom_name or "custom").strip().lower()
    if not (
        settings.custom_base_url
        and settings.custom_api_key
        and name not in PROVIDERS
    ):
        return

    cls = type(
        f"{name.title()}Provider",
        (OpenAICompatibleProvider,),
        {
            "name": name,
            "base_url": settings.custom_base_url.rstrip("/"),
        },
    )
    models = tuple(
        (m.strip(), "balanced", {})
        for m in (settings.custom_models or "").split(",")
        if m.strip()
    )
    PROVIDERS[name] = ProviderSpec(
        name=name,
        cls=cls,
        api_key_env=None,
        curated_models=models,
        supports_discovery=bool(settings.custom_discovery),
        requires_key=True,
        default_timeout_s=90.0,
        settings_key_attr="custom_api_key",
    )
    _build_static_catalog()


def _api_key_for(spec: ProviderSpec, settings: Settings) -> str | None:
    if not spec.requires_key:
        return "local"
    attr = spec.settings_key_attr or f"{spec.name}_api_key"
    return getattr(settings, attr, None)


def provider_enabled(spec: ProviderSpec, settings: Settings) -> bool:
    if not spec.requires_key:
        return settings.enable_ollama or bool(settings.ollama_base_url)
    return bool(_api_key_for(spec, settings))


def is_probably_text_model(model_id: str) -> bool:
    lowered = model_id.lower()
    return not any(kw in lowered for kw in _NON_TEXT_KEYWORDS)


def has_variant_suffix(model_id: str) -> bool:
    local = model_id.split("/", 1)[-1]
    return ":" in local


def dedupe_variants(model_ids: list[str]) -> list[str]:
    """Drop 'model:flag' entries whose base model exists as a standalone id."""
    bases = {
        m for m in model_ids if ":" not in m.split("/", 1)[-1]
    }
    return [
        m for m in model_ids
        if ":" not in m.split("/", 1)[-1] or m.split(":", 1)[0] not in bases
    ]


def get_model_info(model_id: str) -> ModelInfo:
    info = MODEL_CATALOG.get(model_id) or DISCOVERED.get(model_id)
    if info is None:
        raise ConfigError(
            f"unknown model '{model_id}'. known models: "
            f"{sorted(set(MODEL_CATALOG) | set(DISCOVERED))}"
        )
    return info


def available_models(settings: Settings | None = None) -> list[ModelInfo]:
    settings = settings or get_settings()
    _ensure_custom_registered(settings)
    enabled = [
        spec for spec in PROVIDERS.values() if provider_enabled(spec, settings)
    ]
    enabled_names = {spec.name for spec in enabled}
    static = [info for info in MODEL_CATALOG.values() if info.provider in enabled_names]
    dynamic = [
        info for info in DISCOVERED.values() if info.provider in enabled_names
    ]
    seen = {i.model_id for i in static}
    return static + [i for i in dynamic if i.model_id not in seen]


def register_discovered(provider_name: str, model_ids: list[str]) -> list[ModelInfo]:
    added: list[ModelInfo] = []
    for mid in model_ids:
        if not is_probably_text_model(mid):
            continue
        model_id = f"{provider_name}/{mid}"
        if model_id in DISCOVERED:
            added.append(DISCOVERED[model_id])
            continue
        info = ModelInfo(
            model_id=model_id,
            provider=provider_name,
            model=mid,
            tier="discovered",
            discovered=True,
        )
        DISCOVERED[model_id] = info
        added.append(info)
    return added


def create_provider(
    provider_name: str,
    settings: Settings | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    settings = settings or get_settings()
    _ensure_custom_registered(settings)
    spec = PROVIDERS.get(provider_name)
    if spec is None:
        raise ConfigError(f"no provider implementation for '{provider_name}'")
    resolved_key = api_key or _api_key_for(spec, settings)
    kwargs: dict = {
        "timeout_s": spec.default_timeout_s,
        "max_retries": settings.max_retries_per_call,
    }
    if provider_name == "ollama":
        kwargs["base_url_override"] = settings.ollama_base_url
    return spec.cls(api_key=resolved_key or "", **kwargs)


async def discover_models(
    settings: Settings | None = None, force: bool = False
) -> list[ModelInfo]:
    settings = settings or get_settings()
    _ensure_custom_registered(settings)
    results: list[ModelInfo] = []
    for spec in PROVIDERS.values():
        if not spec.supports_discovery or not provider_enabled(spec, settings):
            continue
        try:
            provider = create_provider(spec.name, settings)
            ids = await asyncio.wait_for(
                provider.list_models(), timeout=settings.discovery_timeout_s
            )
        except Exception:
            ids = []
        if spec.name == (settings.custom_name or "custom").strip().lower():
            env_ids = [
                m.strip() for m in (settings.custom_models or "").split(",")
                if m.strip()
            ]
            ids = list(dict.fromkeys(list(ids) + env_ids))

        text_ids = [i for i in ids if is_probably_text_model(i)]
        text_ids = dedupe_variants(text_ids)
        ranker = getattr(provider, "rank_models", None)
        if callable(ranker):
            text_ids = ranker(text_ids)
        cap = max(1, settings.discovery_max_per_provider)
        results.extend(register_discovered(spec.name, text_ids[:cap]))
    return results


def build_llm_chain(
    preferred_model_id: str | None,
    settings: Settings | None = None,
    api_key: str | None = None,
) -> tuple[ModelInfo, list[tuple[LLMProvider, str, ModelInfo]]]:
    settings = settings or get_settings()
    _ensure_custom_registered(settings)
    primary_info = get_model_info(preferred_model_id or settings.default_model_id)

    chain: list[tuple[LLMProvider, str, ModelInfo]] = []
    seen: set[str] = set()

    def add(model_id: str) -> None:
        if model_id in seen:
            return
        try:
            info = get_model_info(model_id)
            provider = create_provider(info.provider, settings, api_key=api_key)
        except ConfigError:
            return
        seen.add(model_id)
        chain.append((provider, info.model, info))

    same_provider = [
        m for m in {**MODEL_CATALOG, **DISCOVERED}.values()
        if m.provider == primary_info.provider and m.model_id != primary_info.model_id
    ]
    other_providers = [
        m for m in {**MODEL_CATALOG, **DISCOVERED}.values()
        if m.provider != primary_info.provider and m.supports_json_mode
    ]

    add(primary_info.model_id)
    for info in sorted(same_provider, key=lambda m: m.tier != "accurate"):
        add(info.model_id)
    for info in sorted(other_providers, key=lambda m: m.tier != "accurate"):
        add(info.model_id)

    return primary_info, chain


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(f"no valid JSON object found in response: {text[:200]!r}")
