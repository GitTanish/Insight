import pytest

from insight.domain.errors import ConfigError
from insight.llm.registry import (
    DISCOVERED,
    MODEL_CATALOG,
    available_models,
    build_llm_chain,
    create_provider,
    dedupe_variants,
    get_model_info,
    is_probably_text_model,
    register_discovered,
)


def settings_with(**keys):
    from insight.settings import Settings

    isolated = {
        "groq_api_key": None,
        "mistral_api_key": None,
        "openai_api_key": None,
        "anthropic_api_key": None,
        "openrouter_api_key": None,
        "ollama_base_url": None,
        "enable_ollama": False,
        "custom_base_url": None,
        "custom_api_key": None,
    }
    isolated.update(keys)
    return Settings(**isolated)


def test_text_model_filter():
    assert is_probably_text_model("openai/gpt-4o") is True
    assert is_probably_text_model("text-embedding-3-large") is False
    assert is_probably_text_model("whisper-large-v3") is False
    assert is_probably_text_model("meta-llama/llama-prompt-guard-2") is False
    assert is_probably_text_model("canopylabs/orpheus-v1-english") is False


def test_dedupe_variants():
    ids = ["vendor/model", "vendor/model:batch", "vendor/other:free"]
    assert dedupe_variants(ids) == ["vendor/model", "vendor/other:free"]


def test_available_models_respects_enabled_providers():
    only_groq = available_models(settings_with(groq_api_key="k"))
    providers_seen = {m.provider for m in only_groq}
    assert providers_seen == {"groq"}
    assert any(m.model_id == "groq/openai/gpt-oss-120b" for m in only_groq)

    groq_and_mistral = available_models(
        settings_with(groq_api_key="k", mistral_api_key="k2")
    )
    providers_seen = {m.provider for m in groq_and_mistral}
    assert providers_seen == {"groq", "mistral"}

    none = available_models(settings_with())
    assert none == []


def test_register_discovered_and_get_model_info():
    added = register_discovered("ollama", ["llama3:8b-instruct", "minicem/embed"])
    model_ids = [a.model_id for a in added]
    assert "ollama/llama3:8b-instruct" in model_ids
    assert all(not i.model_id.endswith("embed") for i in added)
    assert all(i.tier == "discovered" for i in added)

    info = get_model_info("ollama/llama3:8b-instruct")
    assert info.discovered is True

    again = register_discovered("ollama", ["llama3:8b-instruct"])
    assert len(again) == 1

    with pytest.raises(ConfigError):
        get_model_info("nope/never")


def test_custom_provider_registration_from_env():
    s = settings_with(
        custom_base_url="https://api.example.com/v1",
        custom_api_key="sk-x",
        custom_models="model-a, model-b",
        custom_name="acme",
    )
    models = available_models(s)
    acme_ids = [m.model_id for m in models if m.provider == "acme"]
    assert "acme/model-a" in acme_ids and "acme/model-b" in acme_ids

    provider = create_provider("acme", s)
    assert provider.name == "acme"


def test_chain_orders_same_provider_first_then_others():
    s = settings_with(groq_api_key="k", mistral_api_key="k2", openrouter_api_key="k3")
    primary, chain = build_llm_chain("groq/openai/gpt-oss-120b", s)

    assert primary.provider == "groq"
    chain_providers = [p.name for p, _, _ in chain]
    assert chain_providers[0] == "groq"
    assert len(chain) >= 3
    assert "mistral" in chain_providers and "openrouter" in chain_providers


def test_static_catalog_contains_curated_entries():
    assert "groq/openai/gpt-oss-120b" in MODEL_CATALOG
    assert MODEL_CATALOG["groq/openai/gpt-oss-120b"].supports_reasoning_effort
    assert "mistral/mistral-small-latest" in MODEL_CATALOG
    assert DISCOVERED.get("nonexistent") is None
