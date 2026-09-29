from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, ClassVar

import httpx

from insight.domain.errors import (
    ProviderAuthError,
    ProviderError,
    RateLimitError,
)
from insight.llm.base import LLMProvider, LLMRequest, LLMResponse
from insight.observability import trace_span


class OpenAICompatibleProvider(LLMProvider):
    name: ClassVar[str] = "openai-compatible"
    base_url: ClassVar[str] = ""
    # Providers that support OpenRouter-style explicit cache routing.
    supports_prompt_cache_key: ClassVar[bool] = False

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, request: LLMRequest) -> dict:
        payload: dict = {
            "model": request.model,
            "messages": [m.model_dump() for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.json_mode and request.allow_server_json:
            payload["response_format"] = {"type": "json_object"}
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.cache_prompt and self.supports_prompt_cache_key:
            # OpenRouter routes a shared prefix to one backend so repeat calls
            # reuse its cache. Harmless when the prefix genuinely is stable.
            payload["prompt_cache_key"] = "insight-planner"
        return payload

    @trace_span("provider.generate", run_type="llm")
    async def generate(self, request: LLMRequest) -> LLMResponse:
        last_error: Exception | None = None
        current = request

        for attempt in range(self._max_retries):
            try:
                return await self._generate_once(current)
            except ProviderAuthError:
                raise
            except RateLimitError as exc:
                last_error = exc
                if exc.retry_after_s:
                    await asyncio.sleep(min(30.0, exc.retry_after_s))
                else:
                    await self._sleep_backoff(attempt)
            except ProviderError as exc:
                last_error = exc
                status = exc.status_code or 0
                if 400 <= status < 500:
                    if self._is_reasoning_param_error(exc.detail):
                        current = current.model_copy(
                            update={"reasoning_effort": None}
                        )
                        continue
                    raise
                await self._sleep_backoff(attempt)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                await self._sleep_backoff(attempt)

        raise ProviderError(
            self.name,
            request.model,
            f"failed after {self._max_retries} attempts: {last_error}",
        )

    @staticmethod
    def _is_reasoning_param_error(detail: str) -> bool:
        return "reasoning_effort" in detail.lower()

    async def _generate_once(self, request: LLMRequest) -> LLMResponse:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            started = time.monotonic()
            response = await client.post(
                f"{self.base_url}/chat/completions",
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
                self.name, request.model,
                response.text[:300], retry_after_s=retry_after,
            )
        if response.status_code >= 400:
            raise ProviderError(
                self.name, request.model, response.text[:300],
                status_code=response.status_code,
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                self.name, request.model, f"malformed response: {exc}"
            ) from exc

        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return LLMResponse(
            content=content,
            provider=self.name,
            model=data.get("model", request.model),
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cached_prompt_tokens=(
                details.get("cached_tokens")
                or usage.get("cache_read_input_tokens")
                or None
            ),
        )

    @trace_span("provider.stream", run_type="llm")
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield content deltas from the SSE endpoint.

        Falls back to a single non-streaming chunk if the endpoint rejects
        `stream: true` (older gateways, or JSON-mode-only models).
        """
        import json as _json

        payload = self._payload(request)
        payload["stream"] = True
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        await response.aread()
                        if response.status_code in (401, 403):
                            raise ProviderAuthError(
                                self.name, request.model, "authentication failed",
                                status_code=response.status_code,
                            )
                        if response.status_code == 429:
                            raise RateLimitError(
                                self.name, request.model, response.text[:200],
                            )
                        raise ProviderError(
                            self.name, request.model, response.text[:300],
                            status_code=response.status_code,
                        )

                    saw_delta = False
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        chunk = line[5:].strip()
                        if not chunk or chunk == "[DONE]":
                            continue
                        try:
                            event = _json.loads(chunk)
                        except ValueError:
                            continue
                        for choice in event.get("choices", []) or []:
                            delta = (choice.get("delta") or {}).get("content")
                            if delta:
                                saw_delta = True
                                yield delta
                    if saw_delta:
                        return
        except (ProviderError, ProviderAuthError, RateLimitError):
            raise
        except (httpx.HTTPError, httpx.TimeoutException, httpx.TransportError):
            pass

        response = await self.generate(request)
        if response.content:
            yield response.content

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
