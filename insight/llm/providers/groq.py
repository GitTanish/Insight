from typing import ClassVar

from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    name: ClassVar[str] = "groq"
    base_url: ClassVar[str] = "https://api.groq.com/openai/v1"


class Groq2Provider(GroqProvider):
    """Second Groq account — same endpoint, separate quota pool.

    Groq rate limits are per API key, and the free tier is tight; a second
    key (`GROQ_API_KEY2`) doubles both fallback depth and rate-limit headroom.
    The provider name stays distinct so live-run telemetry can prove which
    account served a call.
    """

    name: ClassVar[str] = "groq2"
