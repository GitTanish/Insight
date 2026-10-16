from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class NvidiaProvider(OpenAICompatibleProvider):
    """NVIDIA NIM (build.nvidia.com) — OpenAI-compatible chat completions.

    Hosts its own model catalog (tool-calling models included) behind the
    standard OpenAI wire format; discovery via GET /models.
    """

    name: ClassVar[str] = "nvidia"
    base_url: ClassVar[str] = "https://integrate.api.nvidia.com/v1"
