from __future__ import annotations

import contextlib
import os
import threading
from typing import Any, Callable, Iterator, Optional

from insight.settings import get_settings

try:
    from langsmith import traceable as _ls_traceable
    from langsmith import Client as _LSClient

    LANGSMITH_AVAILABLE = True
except ImportError:
    _ls_traceable = None
    _LSClient = None
    LANGSMITH_AVAILABLE = False


_CLIENT_LOCK = threading.Lock()
_CLIENT: Any = None
_CLIENT_FAILED = False


def _get_client() -> Any:
    """Reuse one LangSmith client; constructing one per query leaks sessions."""
    global _CLIENT, _CLIENT_FAILED

    if not LANGSMITH_AVAILABLE or _LSClient is None:
        return None
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        if _CLIENT_FAILED:
            return None
        try:
            _CLIENT = _LSClient()
        except Exception:
            _CLIENT_FAILED = True
            return None
        return _CLIENT


def close_langsmith_client() -> None:
    global _CLIENT, _CLIENT_FAILED

    with _CLIENT_LOCK:
        client = _CLIENT
        _CLIENT = None
        _CLIENT_FAILED = False
    if client is None:
        return
    try:
        session = getattr(client, "session", None)
        if session is not None and hasattr(session, "close"):
            session.close()
            return
        close = getattr(client, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def configure_langsmith() -> bool:
    settings = get_settings()
    api_key = (
        settings.langsmith_api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )
    project = (
        settings.langsmith_project
        or os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "insight"
    )
    enabled = bool(api_key) and settings.tracing_enabled

    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
    if project:
        os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ["LANGSMITH_TRACING"] = "true" if enabled else "false"

    if not enabled:
        return False

    return _get_client() is not None


def is_tracing_enabled() -> bool:
    if not LANGSMITH_AVAILABLE:
        return False
    override = os.getenv("INSIGHT_TRACING", "").strip().lower()
    if override in {"0", "false", "off", "no"}:
        return False
    settings = get_settings()
    api_key = (
        settings.langsmith_api_key
        or os.getenv("LANGSMITH_API_KEY")
        or os.getenv("LANGCHAIN_API_KEY")
    )
    return bool(api_key) and settings.tracing_enabled


def trace_span(name: str | None = None, run_type: str = "chain") -> Callable:
    """Decorator that records a LangSmith run when tracing is configured.

    Degrades to a no-op when the langsmith package is missing, no API key is
    set, or tracing is disabled in settings.
    """
    def decorator(func: Callable) -> Callable:
        if LANGSMITH_AVAILABLE and is_tracing_enabled():
            return _ls_traceable(
                name=name or func.__name__,
                run_type=run_type,
            )(func)
        return func

    return decorator


def attach_metadata(**fields: Any) -> None:
    """Attach metadata to the currently open LangSmith run, if any."""
    if not (LANGSMITH_AVAILABLE and is_tracing_enabled()):
        return
    try:
        from langsmith import run_helpers

        run_tree = run_helpers.get_current_run_tree()
        if run_tree is not None:
            run_tree.metadata.update({k: v for k, v in fields.items() if v is not None})
    except Exception:
        pass


def current_trace_url() -> str | None:
    if not (LANGSMITH_AVAILABLE and is_tracing_enabled()):
        return None
    try:
        from langsmith import run_helpers

        run_tree = run_helpers.get_current_run_tree()
        if run_tree is not None:
            return f"{run_tree.base_url}/r?trace_id={run_tree.trace_id}"
    except Exception:
        pass
    return None


def start_root_run(name: str, inputs: dict) -> tuple[Any, Any]:
    """Create a root run so streaming and sync calls both appear in the tree."""
    if not is_tracing_enabled():
        return None, None
    try:
        client = _get_client()
        if client is None:
            return None, None
        run = client.create_run(
            name=name,
            run_type="chain",
            inputs={k: v for k, v in inputs.items() if v is not None},
            start_time=_utcnow(),
        )
        return client, run
    except Exception:
        return None, None


def finish_root_run(
    client: Any,
    run: Any,
    outputs: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    if client is None or run is None:
        return
    try:
        client.update_run(
            run.id,
            outputs={"output": outputs} if outputs else None,
            error=error,
            end_time=_utcnow(),
        )
    except Exception:
        pass


@contextlib.contextmanager
def trace_phase(name: str, run_type: str = "chain", **inputs: Any) -> Iterator[Any]:
    """Context manager recording a phase (plan/execute/validate/render/explain)."""
    if not is_tracing_enabled():
        yield None
        return
    try:
        from langsmith import run_helpers
    except Exception:
        yield None
        return
    with run_helpers.trace(
        name=name, run_type=run_type, inputs=inputs or None
    ) as run_tree:
        yield run_tree


def _utcnow():
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc)
