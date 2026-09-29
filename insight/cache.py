from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Optional

from insight.conversation.state import AnalysisSessionState
from insight.domain.dataset import DatasetProfile
from insight.domain.query import AnalysisRequest
from insight.domain.visualization import AnalysisResponse
from insight.settings import get_settings


def normalize_question(question: str) -> str:
    return " ".join(question.lower().split())


def _scope_signature(state: Optional[AnalysisSessionState]) -> str:
    """Only active filters change which rows an answer covers.

    dims/metrics/result digests change prompt context but not the data scope,
    so they are deliberately excluded: asking the same question twice must hit
    the cache even though the session learned something in between.
    """
    if state is None or not state.active_filters:
        return "no-filter"
    parts = sorted(
        f"{f.column.lower()}:{f.op}:{f.value}" for f in state.active_filters
    )
    return "|".join(parts)


def cache_key_for(
    profile: DatasetProfile,
    request: AnalysisRequest,
    state: Optional[AnalysisSessionState] = None,
) -> Optional[str]:
    settings = get_settings()
    if not settings.query_cache_enabled:
        return None
    temperature = (
        request.temperature if request.temperature is not None else settings.temperature
    )
    raw = "|".join([
        profile.fingerprint.content_hash,
        profile.fingerprint.schema_hash,
        str(profile.row_count),
        normalize_question(request.question),
        request.model_id or settings.default_model_id,
        f"{temperature:.3f}",
        _scope_signature(state),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _cache_root() -> Path:
    return get_settings().artifacts_dir / "cache"


def _entry_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _remove(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def sweep_cache() -> int:
    """Enforce TTL and size budget over .artifacts/cache; returns entries removed."""
    settings = get_settings()
    root = _cache_root()
    if not root.exists():
        return 0
    now = time.time()
    max_age_s = max(1, settings.query_cache_ttl_days) * 86_400
    max_bytes = max(1, settings.query_cache_max_mb) * 1024 * 1024

    entries: list[tuple[float, int, Path]] = []
    for path in root.iterdir():
        try:
            entries.append((path.stat().st_mtime, _entry_size(path), path))
        except OSError:
            continue

    removed = 0
    kept: list[tuple[float, int, Path]] = []
    for mtime, size, path in entries:
        if now - mtime > max_age_s:
            _remove(path)
            removed += 1
        else:
            kept.append((mtime, size, path))

    total = sum(size for _, size, _ in kept)
    kept.sort(key=lambda e: e[0])
    for mtime, size, path in kept:
        if total <= max_bytes:
            break
        _remove(path)
        total -= size
        removed += 1
    return removed


def load_cached_response(
    key: str, artifacts_dir: Path | str
) -> Optional[AnalysisResponse]:
    try:
        payload_path = _cache_root() / f"{key}.json"
        blob_dir = _cache_root() / key
        if not payload_path.exists():
            return None
        response = AnalysisResponse.model_validate(
            json.loads(payload_path.read_text(encoding="utf-8"))
        )
        dest_dir = Path(artifacts_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for plot in response.plots:
            src = blob_dir / Path(plot.path).name
            if not src.exists():
                return None
            dest = dest_dir / src.name
            shutil.copyfile(src, dest)
            plot.path = str(dest)
        return response
    except Exception:
        return None


def store_cached_response(key: str, response: AnalysisResponse) -> None:
    try:
        sweep_cache()
        root = get_settings().artifacts_dir
        blob_dir = root / "cache" / key
        blob_dir.mkdir(parents=True, exist_ok=True)
        stored = response.model_copy(deep=True)
        for plot in stored.plots:
            src = Path(plot.path)
            if src.exists():
                shutil.copyfile(src, blob_dir / src.name)
        payload_path = root / "cache" / f"{key}.json"
        payload_path.write_text(
            json.dumps(stored.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass
