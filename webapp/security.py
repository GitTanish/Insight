"""Auth gate, identity resolution and response hardening for the webapp.

The gate is opt-in: set ``INSIGHT_AUTH_TOKEN`` to require a Bearer token (API)
or the browser cookie bootstrapped from ``/auth?token=...``. Deployments behind
an authenticating proxy can instead set ``INSIGHT_TRUST_PROXY_AUTH=on`` and
forward ``X-Forwarded-User``. With neither configured the app runs in
trusted-network mode (documented in PRODUCTION_READINESS.md).
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Iterable

PUBLIC_PATHS = frozenset({"/healthz", "/health", "/readyz", "/auth"})
AUTH_COOKIE = "insight_auth"
UNAUTHORIZED_BODY = b'{"error":"unauthorized"}'


def _headers(scope: dict) -> dict[str, str]:
    return {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in scope.get("headers", [])
    }


def bearer_token(scope: dict) -> str | None:
    header = _headers(scope).get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def cookie_values(scope: dict) -> dict[str, str]:
    raw = _headers(scope).get("cookie", "")
    values: dict[str, str] = {}
    for part in raw.split(";"):
        name, _, value = part.strip().partition("=")
        if name:
            values[name] = value
    return values


def request_identity(scope: dict, trust_proxy: bool | None = None) -> str:
    """Stable, non-PII identity key for rate limiting.

    Precedence: trusted proxy user → bearer token hash → client address.
    Cookies are deliberately not used: forging a fresh cookie must not mint a
    fresh rate-limit bucket. The proxy header is only honoured when the
    deployment opts in (otherwise a caller could rotate it to evade limits).
    """
    if trust_proxy is None:
        from insight.settings import get_settings

        trust_proxy = get_settings().trust_proxy_auth
    headers = _headers(scope)
    if trust_proxy:
        user = headers.get("x-forwarded-user", "").strip()
        if user:
            return "user:" + user[:120]
    token = bearer_token(scope)
    if token:
        return "bearer:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    client = scope.get("client") or ("unknown", 0)
    return "ip:" + str(client[0])


def authenticated(scope: dict, token: str, trust_proxy: bool) -> bool:
    headers = _headers(scope)
    if trust_proxy and headers.get("x-forwarded-user", "").strip():
        return True
    supplied = bearer_token(scope) or cookie_values(scope).get(AUTH_COOKIE)
    if not supplied:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), token.encode("utf-8"))


async def _reject(send, status: int = 401, body: bytes = UNAUTHORIZED_BODY) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class AuthMiddleware:
    """ASGI gate: Bearer token, browser cookie, or trusted proxy identity."""

    def __init__(
        self,
        app,
        token: str,
        *,
        trust_proxy: bool = False,
        public_paths: Iterable[str] = PUBLIC_PATHS,
    ):
        self.app = app
        self.token = token
        self.trust_proxy = trust_proxy
        self.public_paths = frozenset(public_paths)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("path") in self.public_paths:
            await self.app(scope, receive, send)
            return
        if self.trust_proxy and _headers(scope).get("x-forwarded-user", "").strip():
            await self.app(scope, receive, send)
            return
        if authenticated(scope, self.token, self.trust_proxy):
            await self.app(scope, receive, send)
            return
        await _reject(send)


class SecurityHeadersMiddleware:
    """Conservative response headers that are safe for the current UI."""

    def __init__(self, app, *, hsts: bool = False):
        self.app = app
        self.hsts = hsts

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"same-origin"),
                    ]
                )
                if self.hsts:
                    headers.append(
                        (
                            b"strict-transport-security",
                            b"max-age=31536000; includeSubDomains",
                        )
                    )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def session_cookie_kwargs(*, secure: bool) -> dict:
    return {"httponly": True, "samesite": "lax", "secure": secure}
