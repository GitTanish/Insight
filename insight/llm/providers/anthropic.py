from __future__ import annotations

import time
from typing import Any, ClassVar

import httpx

from insight.domain.errors import (
    ProviderAuthError,
    ProviderError,
    RateLimitError,
)
from insight.llm.base import LLMRequest, LLMResponse, ToolCall
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
        messages: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                # Anthropic wants tool results inside a user turn.
                block = {
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id or "",
                    "content": m.content,
                }
                if (
                    messages
                    and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                ):
                    messages[-1]["content"].append(block)
                else:
                    messages.append({"role": "user", "content": [block]})
                continue
            if m.role == "assistant" and m.tool_calls:
                blocks: list[dict[str, Any]] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                for raw in m.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": raw.get("id", ""),
                        "name": raw.get("name", ""),
                        "input": raw.get("arguments") or {},
                    })
                messages.append({"role": "assistant", "content": blocks})
                continue
            messages.append({"role": m.role, "content": m.content})
        last = request.messages[-1] if request.messages else None
        if (
            request.json_mode
            and messages
            and last is not None
            and last.role == "user"
            and isinstance(messages[-1]["content"], str)
        ):
            messages[-1]["content"] += (
                "\nRespond with ONLY a single valid JSON object."
            )

        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens or self._default_max_tokens,
            "messages": messages,
        }
        if system_text:
            if request.cache_prompt:
                # Explicit ephemeral breakpoint so the stable planner prefix is
                # read from Anthropic's prompt cache on follow-up turns.
                payload["system"] = [{
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }]
            else:
                payload["system"] = system_text
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = request.tools
            if request.tool_choice:
                # The loop only ever asks for "auto"; Anthropic takes the
                # typed object form.
                payload["tool_choice"] = {"type": "auto"}
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
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            for block in data.get("content") or []:
                if block.get("type") == "text":
                    text_parts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("input") or {},
                        )
                    )
            content = "".join(text_parts)
        except (AttributeError, TypeError) as exc:
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
            tool_calls=tool_calls,
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
