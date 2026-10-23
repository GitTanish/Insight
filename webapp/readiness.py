"""Liveness vs readiness: can this instance actually serve an analysis?"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from insight import __version__
from insight.settings import Settings, get_settings


def _artifacts_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".readyz-", dir=str(directory), delete=True
        ):
            pass
        return True
    except Exception:
        return False


def _provider_configured(settings: Settings) -> bool:
    if settings.require_user_key:
        return True
    try:
        from insight.llm import registry

        return any(
            registry.provider_enabled(spec, settings)
            for spec in registry.PROVIDERS.values()
        )
    except Exception:
        return False


def readiness_report(settings: Optional[Settings] = None) -> tuple[bool, dict]:
    settings = settings or get_settings()
    checks = {
        "artifacts_writable": _artifacts_writable(settings.artifacts_dir),
        "provider_configured": _provider_configured(settings),
    }
    ready = all(checks.values())
    return ready, {"status": "ready" if ready else "not_ready", **checks, "version": __version__}
