import json

import httpx
import pytest

from insight.domain.errors import ProviderAuthError, ProviderError
from insight.llm.base import LLMRequest
from insight.llm.providers.anthropic import AnthropicProvider
from insight.llm.providers.ollama import OllamaProvider
from insight.llm.providers.openai_compat import OpenAICompatibleProvider


class TestCompatProvider(OpenAICompatibleProvider):
    name = "test-compat"
    base_url = "https://test.example/v1"


def _request(**kw) -> LLMRequest:
    defaults: dict = {
        "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ],
        "model": "test-model",
        "temperature": 0.0,
    }
    defaults.update(kw)
    return LLMRequest(**defaults)


def _ok_openai_body(content: str = "hello") -> dict:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def test_openai_compat_generate_and_payload():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(200, json=_ok_openai_body("hello"))

    provider = TestCompatProvider(
        api_key="k", timeout_s=5.0, transport=httpx.MockTransport(handler)
    )
    resp = provider._run_sync(provider.generate(_request(json_mode=True)))

    assert resp.content == "hello"
    assert resp.prompt_tokens == 3 and resp.completion_tokens == 2
    body = json.loads(bodies[0])
    assert body["response_format"] == {"type": "json_object"}
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.0


def test_server_json_suppressed_when_not_allowed():
    bodies: list[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(request.read())
        return httpx.Response(200, json=_ok_openai_body())

    provider = TestCompatProvider(
        api_key="k", timeout_s=5.0, transport=httpx.MockTransport(handler)
    )
    provider._run_sync(
        provider.generate(_request(json_mode=True, allow_server_json=False))
    )
    body = json.loads(bodies[0])
    assert "response_format" not in body


def test_retry_on_500_then_success():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(500, text="boom")
        return httpx.Response(200, json=_ok_openai_body())

    provider = TestCompatProvider(
        api_key="k", timeout_s=5.0, max_retries=3,
        transport=httpx.MockTransport(handler),
    )
    resp = provider._run_sync(provider.generate(_request()))
    assert resp.content == "hello"
    assert calls["n"] == 2


def test_auth_error_raises_without_retry():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, text="unauthorized")

    provider = TestCompatProvider(
        api_key="bad", timeout_s=5.0, max_retries=3,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderAuthError):
        provider._run_sync(provider.generate(_request()))
    assert calls["n"] == 1


def test_client_error_raises_without_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad request")

    provider = TestCompatProvider(
        api_key="k", timeout_s=5.0, max_retries=3,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ProviderError):
        provider._run_sync(provider.generate(_request()))


def test_anthropic_payload_parse_and_headers():
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = request.read()
        return httpx.Response(200, json={
            "model": "claude-test",
            "content": [
                {"type": "text", "text": "he"},
                {"type": "text", "text": "y"},
            ],
            "usage": {"input_tokens": 5, "output_tokens": 7},
        })

    provider = AnthropicProvider(
        api_key="sk-ant-test", timeout_s=5.0,
        transport=httpx.MockTransport(handler),
    )
    resp = provider._run_sync(provider.generate(_request(json_mode=True)))

    assert resp.content == "hey"
    assert resp.prompt_tokens == 5 and resp.completion_tokens == 7
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    body = json.loads(captured["body"])
    assert body["system"] == "SYS"
    assert all(m["role"] != "system" for m in body["messages"])
    assert body["max_tokens"] == 4096
    assert "Respond with ONLY" in body["messages"][0]["content"]


def test_ollama_base_url_override():
    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=_ok_openai_body())

    provider = OllamaProvider(
        timeout_s=5.0, transport=httpx.MockTransport(handler),
        base_url_override="http://127.0.0.1:9999/v1",
    )
    resp = provider._run_sync(provider.generate(_request()))

    assert resp.provider == "ollama"
    assert seen_urls[0].startswith("http://127.0.0.1:9999/v1/chat/completions")
