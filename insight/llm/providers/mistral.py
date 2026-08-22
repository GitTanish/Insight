from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class MistralProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "mistral"
    base_url: ClassVar[str] = "https://api.mistral.ai/v1"
