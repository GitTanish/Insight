"""Per-event-loop shared httpx clients.

Providers historically created a fresh ``AsyncClient`` (and TLS handshake) for
every request. Reusing one client per provider endpoint keeps connections warm
and bounds the number of sockets. Custom transports (tests, proxies) keep the
old per-call lifecycle so injected transports are never shared by accident.
"""

from __future__ import annotations

import asyncio
import threading
import weakref
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx

_LIMITS = httpx.Limits(
    max_connections=20,
    max_keepalive_connections=10,
    keepalive_expiry=30.0,
)
_POOLS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, float], httpx.AsyncClient]]" = (
    weakref.WeakKeyDictionary()
)
_LOCK = threading.Lock()


def _client_for(base_url: str, timeout_s: float) -> httpx.AsyncClient:
    key = (base_url.rstrip("/"), float(timeout_s))
    loop = asyncio.get_running_loop()
    with _LOCK:
        clients = _POOLS.setdefault(loop, {})
        client = clients.get(key)
        if client is None or client.is_closed:
            client = httpx.AsyncClient(timeout=timeout_s, limits=_LIMITS)
            clients[key] = client
        return client


@asynccontextmanager
async def http_client(
    base_url: str,
    timeout_s: float,
    transport: object | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    if transport is not None:
        async with httpx.AsyncClient(
            timeout=timeout_s, transport=transport
        ) as client:
            yield client
        return
    yield _client_for(base_url, timeout_s)


async def close_http_clients() -> None:
    """Close clients belonging to the current event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    with _LOCK:
        clients = _POOLS.pop(loop, {})
    for client in clients.values():
        try:
            await client.aclose()
        except Exception:
            pass
