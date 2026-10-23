"""Artifact retention: delete generated report/chart directories older than N days.

Only ``<root>/web/<data_id>/`` directories are considered — the query cache has
its own TTL/eviction and evaluation artifacts are left untouched. Symlinks are
never followed or removed.
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def _newest_mtime(path: Path) -> float | None:
    if path.is_symlink():
        return None
    if path.is_file():
        try:
            return path.stat().st_mtime
        except OSError:
            return None
    newest: float | None = None
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                continue
            try:
                mtime = candidate.stat().st_mtime
            except OSError:
                continue
            newest = mtime if newest is None else max(newest, mtime)
    if newest is None:
        try:
            newest = path.stat().st_mtime
        except OSError:
            return None
    return newest


def _size(path: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            candidate = Path(dirpath) / name
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def sweep_artifacts(
    root: str | Path,
    max_age_days: float,
    *,
    now: float | None = None,
    dry_run: bool = False,
) -> dict:
    if max_age_days < 0:
        raise ValueError("max_age_days must be >= 0")
    root_path = Path(root)
    base = root_path / "web"
    cutoff = (time.time() if now is None else now) - max_age_days * 86400
    removed: list[str] = []
    failed: list[dict] = []
    bytes_freed = 0
    scanned = 0

    if not base.exists():
        return {
            "root": str(root_path),
            "scanned": 0,
            "removed": 0,
            "bytes_freed": 0,
            "dry_run": dry_run,
            "removed_paths": [],
            "failed": [],
        }

    for child in sorted(base.iterdir()):
        if child.is_symlink() or not child.is_dir():
            continue
        scanned += 1
        newest = _newest_mtime(child)
        if newest is None or newest >= cutoff:
            continue
        size = _size(child)
        if not dry_run:
            try:
                shutil.rmtree(child)
            except OSError as exc:
                failed.append({"path": str(child), "error": str(exc)})
                continue
        removed.append(str(child))
        bytes_freed += size

    return {
        "root": str(root_path),
        "scanned": scanned,
        "removed": len(removed),
        "bytes_freed": bytes_freed,
        "dry_run": dry_run,
        "removed_paths": removed[:50],
        "failed": failed,
    }
