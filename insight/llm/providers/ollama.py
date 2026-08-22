from __future__ import annotations

from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class OllamaProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "ollama"
    base_url: ClassVar[str] = "http://localhost:11434/v1"

    def __init__(
        self,
        api_key: str = "local",
        timeout_s: float = 120.0,
        max_retries: int = 2,
        base_url_override: str | None = None,
        transport: object | None = None,
    ):
        super().__init__(
            api_key=api_key,
            timeout_s=timeout_s,
            max_retries=max_retries,
            transport=transport,
        )
        if base_url_override:
            self.base_url = base_url_override.rstrip("/")
