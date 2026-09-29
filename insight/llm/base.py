from __future__ import annotations

import abc
import asyncio
import random
from typing import AsyncIterator, ClassVar, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMRequest(BaseModel):
    messages: list[Message]
    model: str
    temperature: float | None = 0.0
    max_tokens: int | None = None
    json_mode: bool = False
    allow_server_json: bool = True
    reasoning_effort: str | None = None
    cache_prompt: bool = False


class LLMResponse(BaseModel):
    content: str
    provider: str
    model: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_prompt_tokens: int | None = None


class LLMProvider(abc.ABC):
    name: ClassVar[str] = "abstract"

    def __init__(
        self,
        api_key: str,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        transport: object | None = None,
    ):
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._transport = transport

    @abc.abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield answer text incrementally.

        Default implementation degrades to a single chunk so every provider is
        stream-capable; providers that speak SSE override this.
        """
        response = await self.generate(request)
        if response.content:
            yield response.content

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        ...

    async def _sleep_backoff(self, attempt: int) -> float:
        delay = min(8.0, 0.5 * (2**attempt)) + random.uniform(0, 0.25)
        await asyncio.sleep(delay)
        return delay

    def _run_sync(self, coro):
        return asyncio.run(coro)
