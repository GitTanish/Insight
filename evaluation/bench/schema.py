"""Shared schema, manifest, and statistics helpers for benchmark evidence.

The evidence contract is intentionally narrow: every benchmark record has a
stable trial id, the mutation that was applied, the verdict from the component
under test, and the final validation verdict. Aggregates are derived from those
records; the raw JSONL remains the source of truth.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation.ab_stats import wilson_ci

SCHEMA_VERSION = "1.0"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas scalars and nested containers to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(jsonable(record), sort_keys=True) + "\n")


def git_info(root: Path) -> dict:
    def run(*args: str, strip: bool = True) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        output = result.stdout
        return output.strip() if strip else output.rstrip("\n")

    status = run("status", "--porcelain", "--untracked-files=all", strip=False)
    lines = [line for line in status.splitlines() if line.strip()]
    dirty_files = [line[3:] for line in lines if not line.startswith("??")]
    untracked_files = [line[3:] for line in lines if line.startswith("??")]
    return {
        "commit": run("rev-parse", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "dirty": bool(dirty_files),
        "dirty_files": dirty_files[:50],
        "dirty_file_count": len(dirty_files),
        "untracked_files": untracked_files[:50],
        "untracked_file_count": len(untracked_files),
    }


def _total_memory_bytes() -> int | None:
    try:
        if sys.platform == "win32":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return int(page_size * pages)
    except (AttributeError, OSError, ValueError):
        return None


def environment_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "total_memory_bytes": _total_memory_bytes(),
        "packages": _package_versions(),
    }


def _package_versions() -> dict[str, str | None]:
    from importlib import metadata

    versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "pydantic", "hypothesis"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def rate_summary(records: list[dict], z: float = 1.96) -> dict:
    trials = len(records)
    detected = sum(1 for record in records if record.get("detected"))
    warning_only = sum(
        1
        for record in records
        if not record.get("detected") and record.get("verifier_warning_count", 0) > 0
    )
    validator_rejected = sum(
        1 for record in records if record.get("validator_valid") is False
    )
    low, high = wilson_ci(detected, trials, z=z) if trials else (0.0, 0.0)
    return {
        "trials": trials,
        "detected": detected,
        "missed": trials - detected,
        "rate": (detected / trials) if trials else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
        "warning_only": warning_only,
        "validator_rejected": validator_rejected,
    }


def group_rate(
    records: list[dict],
    key_fn,
    *,
    sort_key=None,
) -> list[dict]:
    groups: dict[Any, list[dict]] = {}
    for record in records:
        key = key_fn(record)
        groups.setdefault(key, []).append(record)
    output = []
    for key, subset in groups.items():
        entry = {"key": key if not isinstance(key, tuple) else list(key)}
        entry.update(rate_summary(subset))
        output.append(entry)
    if sort_key is not None:
        output.sort(key=sort_key)
    else:
        output.sort(key=lambda item: str(item["key"]))
    return output


def clean_false_positive_summary(records: list[dict]) -> dict:
    """Summarize clean-control failures (error verdicts and warnings)."""
    clean = [r for r in records if r.get("trial_kind") == "clean_control"]
    false_positives = [
        r for r in clean if r.get("verifier_error_count", 0) > 0
    ]
    warning_controls = [
        r for r in clean if r.get("verifier_warning_count", 0) > 0
    ]
    invalid = [r for r in clean if r.get("validator_valid") is False]
    trials = len(clean)
    fp_low, fp_high = wilson_ci(len(false_positives), trials) if trials else (0.0, 0.0)
    return {
        "trials": trials,
        "false_positive_trials": len(false_positives),
        "false_positive_rate": (len(false_positives) / trials) if trials else None,
        "false_positive_wilson_95_low": fp_low,
        "false_positive_wilson_95_high": fp_high,
        "warning_control_trials": len(warning_controls),
        "invalid_clean_controls": len(invalid),
    }


def command_line() -> str:
    return " ".join([Path(sys.executable).name, *sys.argv])


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
