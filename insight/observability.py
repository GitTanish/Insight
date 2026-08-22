from __future__ import annotations

import functools
import os
from typing import Any, Callable

from insight.settings import get_settings

try:
    from langsmith import traceable as _ls_traceable
    from langsmith import Client as _LSClient

    LANGSMITH_AVAILABLE = True
except ImportError:
    _ls_traceable = None
    _LSClient = None
    LANGSMITH_AVAILABLE = False


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

    try:
        _LSClient()
    except Exception:
        return False
    return True


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
