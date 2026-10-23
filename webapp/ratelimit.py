"""Per-identity sliding-window rate limiting for expensive routes."""

from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from functools import partial

from webapp.security import request_identity

QUERY_PATHS = frozenset({"/query", "/api/query", "/models/refresh"})
UPLOAD_PATHS = frozenset({"/upload"})


class SlidingWindowLimiter:
    def __init__(self, clock=time.monotonic, max_keys: int = 10_000):
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()
        self._clock = clock
        self._max_keys = max(1, max_keys)
        self._calls = 0

    def allow(
        self, key: str, limit: int, window_s: float = 60.0
    ) -> tuple[bool, float]:
        """Return ``(allowed, retry_after_s)``; ``limit <= 0`` disables."""
        if limit <= 0:
            return True, 0.0
        now = self._clock()
        with self._lock:
            events = self._events.setdefault(key, deque())
            while events and now - events[0] >= window_s:
                events.popleft()
            if len(events) >= limit:
                retry_after = max(0.0, window_s - (now - events[0]))
                return False, retry_after
            events.append(now)
            self._calls += 1
            if self._calls % 1024 == 0 or len(self._events) > self._max_keys:
                self._prune(now, window_s)
            return True, 0.0

    def _prune(self, now: float, window_s: float) -> None:
        stale = [
            key
            for key, events in self._events.items()
            if not events or now - events[-1] >= window_s
        ]
        for key in stale:
            self._events.pop(key, None)
        while len(self._events) > self._max_keys:
            self._events.pop(next(iter(self._events)))


class RateLimitMiddleware:
    """429 + Retry-After on query/upload bursts, per client identity."""

    def __init__(
        self,
        app,
        *,
        query_limit: int,
        upload_limit: int,
        window_s: float = 60.0,
        limiter: SlidingWindowLimiter | None = None,
        identity=None,
    ):
        self.app = app
        self.query_limit = query_limit
        self.upload_limit = upload_limit
        self.window_s = window_s
        self.limiter = limiter or SlidingWindowLimiter()
        if identity is None:
            from insight.settings import get_settings

            identity = partial(
                request_identity, trust_proxy=get_settings().trust_proxy_auth
            )
        self.identity = identity

    def _bucket_for(self, method: str, path: str) -> tuple[str, int] | None:
        if method not in {"POST", "PUT", "PATCH"}:
            return None
        if path in QUERY_PATHS:
            return "query", self.query_limit
        if path in UPLOAD_PATHS:
            return "upload", self.upload_limit
        return None

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        bucket = self._bucket_for(scope.get("method", ""), scope.get("path", ""))
        if bucket is None:
            await self.app(scope, receive, send)
            return
        name, limit = bucket
        key = f"{name}:{self.identity(scope)}"
        allowed, retry_after = self.limiter.allow(key, limit, self.window_s)
        if allowed:
            await self.app(scope, receive, send)
            return
        body = json.dumps(
            {
                "error": "rate limit exceeded",
                "limit": limit,
                "window_s": self.window_s,
                "retry_after_s": round(retry_after, 1),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", str(max(1, math.ceil(retry_after))).encode()),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
