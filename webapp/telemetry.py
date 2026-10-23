"""Request IDs, structured access logs and a minimal Prometheus endpoint.

No extra dependencies: the metric surface is intentionally small (counts,
duration sums, active sessions, uptime) and rendered in the text exposition
format that Prometheus/OpenTelemetry collectors can scrape.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict

LOG = logging.getLogger("insight.http")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)

_START = time.time()
_DYNAMIC_ARTIFACT = object()

_KNOWN_PATHS = frozenset(
    {
        "/",
        "/health",
        "/healthz",
        "/readyz",
        "/metrics",
        "/auth",
        "/upload",
        "/query",
        "/api/query",
        "/clear",
        "/export/docx",
        "/export/docx/custom",
        "/export/dbt",
        "/export/alert",
        "/models/refresh",
    }
)


def metric_path(path: str) -> str:
    if path.startswith("/static/"):
        return "/static/{file}"
    if path.startswith("/artifacts/"):
        return "/artifacts/{data_id}/{filename}"
    return path if path in _KNOWN_PATHS else "other"


class MetricsRegistry:
    def __init__(self):
        self._requests: dict[tuple[str, str, int], int] = defaultdict(int)
        self._duration_sum: dict[tuple[str, str, int], float] = defaultdict(float)
        self._duration_count: dict[tuple[str, str, int], int] = defaultdict(int)

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        key = (method, metric_path(path), int(status))
        self._requests[key] += 1
        self._duration_sum[key] += max(0.0, seconds)
        self._duration_count[key] += 1

    def render(self) -> str:
        lines = [
            "# HELP insight_http_requests_total HTTP requests by route/status.",
            "# TYPE insight_http_requests_total counter",
        ]
        for (method, path, status), count in sorted(self._requests.items()):
            lines.append(
                'insight_http_requests_total{method="%s",path="%s",status="%d"} %d'
                % (method, path, status, count)
            )
        lines.extend(
            [
                "# HELP insight_http_request_duration_seconds_sum Total request time.",
                "# TYPE insight_http_request_duration_seconds_sum counter",
            ]
        )
        for key in sorted(self._duration_sum):
            method, path, status = key
            lines.append(
                'insight_http_request_duration_seconds_sum{method="%s",path="%s",status="%d"} %.6f'
                % (method, path, status, self._duration_sum[key])
            )
        lines.extend(
            [
                "# HELP insight_http_request_duration_seconds_count Request count for the duration sum.",
                "# TYPE insight_http_request_duration_seconds_count counter",
            ]
        )
        for key in sorted(self._duration_count):
            method, path, status = key
            lines.append(
                'insight_http_request_duration_seconds_count{method="%s",path="%s",status="%d"} %d'
                % (method, path, status, self._duration_count[key])
            )
        try:
            from webapp.sessions import STORE

            sessions = len(STORE)
        except Exception:
            sessions = 0
        lines.extend(
            [
                "# HELP insight_sessions Active in-memory sessions.",
                "# TYPE insight_sessions gauge",
                f"insight_sessions {sessions}",
                "# HELP insight_uptime_seconds Process uptime.",
                "# TYPE insight_uptime_seconds gauge",
                f"insight_uptime_seconds {time.time() - _START:.1f}",
            ]
        )
        return "\n".join(lines) + "\n"


REGISTRY = MetricsRegistry()


class TelemetryMiddleware:
    """Attach a request ID, count the request and emit one JSON access line."""

    def __init__(self, app, registry: MetricsRegistry | None = None):
        self.app = app
        self.registry = registry or REGISTRY

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        request_id = (
            headers.get("x-request-id", "").strip()
            or uuid.uuid4().hex[:16]
        )
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        status = 500
        started = time.perf_counter()

        async def send_with_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message.get("status", 500))
                message.setdefault("headers", []).append(
                    (b"x-request-id", request_id.encode("latin-1"))
                )
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            elapsed = time.perf_counter() - started
            path = scope.get("path", "")
            self.registry.observe(scope.get("method", ""), path, status, elapsed)
            client = (scope.get("client") or ("unknown", 0))[0]
            LOG.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": scope.get("method", ""),
                        "path": path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 1),
                        "client": client,
                    },
                    separators=(",", ":"),
                )
            )
