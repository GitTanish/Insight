from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "openai"
    base_url: ClassVar[str] = "https://api.openai.com/v1"
