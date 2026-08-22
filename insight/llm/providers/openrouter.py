from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "openrouter"
    base_url: ClassVar[str] = "https://openrouter.ai/api/v1"

    _PREFERRED_VENDOR_PREFIXES: ClassVar[tuple[str, ...]] = (
        "openai/",
        "anthropic/",
        "google/",
        "meta-llama/",
        "qwen/",
        "mistralai/",
        "deepseek/",
        "moonshotai/",
    )

    def rank_models(self, model_ids: list[str]) -> list[str]:
        preferred = [
            m for m in model_ids
            if m.lower().startswith(self._PREFERRED_VENDOR_PREFIXES)
        ]
        rest = [m for m in model_ids if m not in set(preferred)]
        return (
            sorted(preferred, key=lambda m: (len(m), m))
            + sorted(rest, key=lambda m: (len(m), m))
        )
