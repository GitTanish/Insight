from __future__ import annotations

import time
from typing import Any, ClassVar

import httpx

from insight.domain.errors import (
    ProviderAuthError,
    ProviderError,
    RateLimitError,
)
from insight.llm.base import LLMRequest, LLMResponse
from insight.llm.providers.openai_compat import OpenAICompatibleProvider
from insight.observability import trace_span


class AnthropicProvider(OpenAICompatibleProvider):
    """Anthropic /v1/messages adapter.

    Reuses the retry/backoff orchestration from the OpenAI-compatible base but
    overrides headers, payload construction and response parsing.
    """

    name: ClassVar[str] = "anthropic"
    base_url: ClassVar[str] = "https://api.anthropic.com/v1"
    api_version: ClassVar[str] = "2023-06-01"

    _default_max_tokens = 4096

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": self.api_version,
            "Content-Type": "application/json",
        }

    def _payload(self, request: LLMRequest) -> dict:
        system_text = "\n\n".join(
            m.content for m in request.messages if m.role == "system"
        )
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages
            if m.role != "system"
        ]
        last = request.messages[-1]
        if request.json_mode and messages and last.role == "user":
            messages[-1]["content"] += (
                "\nRespond with ONLY a single valid JSON object."
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "messages": messages,
        }
        if system_text:
            payload["system"] = system_text
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        return payload

    @trace_span("provider.generate", run_type="llm")
    async def _generate_once(self, request: LLMRequest) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            started = time.monotonic()
            response = await client.post(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=self._payload(request),
            )
            latency_ms = int((time.monotonic() - started) * 1000)

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                self.name, request.model, "authentication failed",
                status_code=response.status_code,
            )
        if response.status_code == 429:
            raw = response.headers.get("retry-after")
            retry_after = float(raw) if raw and raw.replace(".", "").isdigit() else None
            raise RateLimitError(
                self.name, request.model, response.text[:300],
                retry_after_s=retry_after,
            )
        if response.status_code >= 400:
            raise ProviderError(
                self.name, request.model, response.text[:300],
                status_code=response.status_code,
            )

        data = response.json()
        try:
            content = "".join(
                block.get("text", "") for block in data["content"]
                if block.get("type") == "text"
            )
        except (KeyError, TypeError) as exc:
            raise ProviderError(
                self.name, request.model, f"malformed response: {exc}"
            ) from exc

        usage = data.get("usage") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=data.get("model", request.model),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
        )

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            response = await client.get(
                f"{self.base_url}/models", headers=self._headers()
            )
        if response.status_code != 200:
            raise ProviderError(
                self.name, "-", f"list_models failed: {response.status_code}"
            )
        return sorted(m["id"] for m in response.json().get("data", []))
