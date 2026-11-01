"""Load, concurrency, and soak evidence for the trust core.

This harness exercises three surfaces that a successful deployment depends on:

* the deterministic analysis executor under concurrent calls,
* the HTTP upload path under concurrent clients, and
* the result cache under write/read churn.

It also runs a bounded-duration upload soak while sampling the server process's
resident set size (RSS). The harness starts a real single-worker uvicorn process
and talks to it over localhost HTTP; it never calls an LLM. Raw JSONL is the
source of truth; the summary and Markdown report are derived views.

The harness is internal evidence tooling. It disables rate limiting and model
discovery for the child server, restores the process settings it mutates, and
never changes shipped defaults.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import httpx

from evaluation.ab_stats import wilson_ci
from evaluation.bench.fixtures import Fixture, build_fixtures
from insight import cache
from insight.domain.visualization import AnalysisResponse
from insight.execution.executor import execute_plan_sync
from insight.settings import get_settings

ROOT = Path(__file__).resolve().parent.parent.parent

_CACHE_SETTINGS = (
    "artifacts_dir",
    "query_cache_enabled",
    "query_cache_ttl_days",
    "query_cache_max_mb",
)


@dataclass(frozen=True)
class LoadSoakConfig:
    analysis_rounds: int = 2
    analysis_repeats: int = 6
    analysis_workers: int = 4
    analysis_fixture_limit: int = 0
    http_uploads: int = 48
    http_workers: int = 8
    http_rows: int = 400
    http_timeout_s: float = 120.0
    soak_seconds: float = 60.0
    soak_workers: int = 4
    soak_rows: int = 200
    soak_sample_interval_s: float = 1.0
    soak_max_growth_mb: float = 96.0
    soak_warmup_requests: int = 2
    cache_writes: int = 48
    cache_writers: int = 2
    cache_readers: int = 4
    cache_reads: int = 120
    cache_max_mb: int = 1
    cache_payload_kb: int = 96


def config_for_mode(mode: str) -> LoadSoakConfig:
    if mode == "full":
        return LoadSoakConfig()
    return LoadSoakConfig(
        analysis_rounds=1,
        analysis_repeats=3,
        analysis_workers=2,
        analysis_fixture_limit=8,
        http_uploads=12,
        http_workers=4,
        http_rows=120,
        soak_seconds=3.0,
        soak_workers=2,
        soak_rows=100,
        soak_sample_interval_s=0.5,
        cache_writes=24,
        cache_writers=2,
        cache_readers=2,
        cache_reads=30,
        cache_payload_kb=64,
    )


def _record(
    run_id: str,
    claim_id: str,
    component: str,
    scenario: str,
    *,
    passed: bool,
    expected: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": "load-soak-1",
        "run_id": run_id,
        "suite": "load_soak",
        "trial_id": f"{claim_id}-{scenario}",
        "claim_id": claim_id,
        "component": component,
        "scenario": scenario,
        "expectation": "scored",
        "expected": expected,
        "passed": bool(passed),
        "details": details or {},
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _latency_summary(values: list[float]) -> dict[str, float | None]:
    return {
        "p50": round(_percentile(values, 0.50), 2) if values else None,
        "p95": round(_percentile(values, 0.95), 2) if values else None,
        "max": round(max(values), 2) if values else None,
        "mean": round(statistics.fmean(values), 2) if values else None,
    }


def _strip_timings(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_timings(item)
            for key, item in value.items()
            if key != "duration_ms"
        }
    if isinstance(value, list):
        return [_strip_timings(item) for item in value]
    return value


def _execution_outcome(fixture: Fixture) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        result = execute_plan_sync(fixture.frame.copy(), fixture.plan)
        payload = _strip_timings(result.model_dump(mode="json"))
        digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8")
        ).hexdigest()
        return {
            "success": bool(result.success),
            "signature": digest,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception as exc:  # noqa: BLE001 - a failure is evidence
        return {
            "success": False,
            "signature": None,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def _analysis_trials(run_id: str, config: LoadSoakConfig) -> list[dict]:
    fixtures = build_fixtures()
    if config.analysis_fixture_limit > 0:
        fixtures = fixtures[: config.analysis_fixture_limit]
    records: list[dict] = []
    workers = max(1, min(config.analysis_workers, config.analysis_repeats))

    for round_index in range(config.analysis_rounds):
        for fixture in fixtures:
            baseline = _execution_outcome(fixture)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as pool:
                outcomes = list(
                    pool.map(
                        lambda _index: _execution_outcome(fixture),
                        range(config.analysis_repeats),
                    )
                )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            signatures = {
                outcome.get("signature")
                for outcome in outcomes
                if outcome.get("success")
            }
            failures = [
                outcome
                for outcome in outcomes
                if not outcome.get("success")
                or outcome.get("signature") != baseline.get("signature")
            ]
            passed = (
                baseline.get("success") is True
                and not failures
                and len(signatures) == 1
                and baseline.get("signature") in signatures
            )
            records.append(
                _record(
                    run_id,
                    "L-01",
                    "analysis_executor",
                    f"{fixture.fixture_id}_round_{round_index}",
                    passed=passed,
                    expected=(
                        f"{config.analysis_repeats} concurrent executions match "
                        "the serial signature"
                    ),
                    details={
                        "fixture_id": fixture.fixture_id,
                        "operation": fixture.operation,
                        "round": round_index,
                        "workers": workers,
                        "repeats": config.analysis_repeats,
                        "elapsed_ms": elapsed_ms,
                        "baseline_elapsed_ms": baseline.get("elapsed_ms"),
                        "baseline_signature": baseline.get("signature"),
                        "unique_signatures": len(signatures),
                        "failures": failures[:3],
                    },
                )
            )
    return records


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(path: Path, limit: int = 1200) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-limit:]


def _server_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "INSIGHT_TRACING": "off",
            "INSIGHT_QUERY_CACHE": "off",
            "INSIGHT_RATE_LIMIT": "off",
            "INSIGHT_CUSTOM_DISCOVERY": "off",
            "INSIGHT_BRIEFING_LLM_POLISH": "off",
            "INSIGHT_AUDIT_LEDGER": "off",
            "INSIGHT_SESSION_MAX": "64",
        }
    )
    return env


@dataclass
class _Server:
    base_url: str
    process: subprocess.Popen
    log_path: Path


@contextmanager
def _server(work_root: Path, timeout_s: float = 60.0) -> Iterator[_Server]:
    port = _free_port()
    work_root.mkdir(parents=True, exist_ok=True)
    log_path = work_root / "uvicorn.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "webapp.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
            "--no-access-log",
        ],
        cwd=ROOT,
        env=_server_env(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"uvicorn exited with {process.returncode}: {_tail(log_path)}"
                )
            try:
                response = httpx.get(
                    f"{base_url}/healthz", timeout=1.0, trust_env=False
                )
                if response.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.2)
        else:
            raise RuntimeError(f"uvicorn did not become healthy: {_tail(log_path)}")
        yield _Server(base_url=base_url, process=process, log_path=log_path)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        log_handle.close()


def _csv_bytes(task_id: int, rows: int) -> bytes:
    regions = ("North", "South", "East", "West")
    categories = ("Alpha", "Bravo", "Charlie", "Delta")
    lines = ["region,category,revenue,units"]
    for index in range(rows):
        region = regions[(task_id + index) % len(regions)]
        category = categories[(task_id * 3 + index) % len(categories)]
        revenue = round(100 + ((task_id * 37 + index * 7) % 900) + (index % 13) / 10, 1)
        units = 1 + (index * 5 + task_id) % 25
        lines.append(f"{region},{category},{revenue},{units}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _upload_once(
    base_url: str,
    task_id: int,
    rows: int,
    timeout_s: float,
    *,
    verify_name: bool = True,
) -> dict[str, Any]:
    name = f"load_{task_id}.csv"
    started = time.perf_counter()
    with httpx.Client(
        base_url=base_url, timeout=timeout_s, trust_env=False
    ) as client:
        upload = client.post(
            "/upload",
            files={"file": (name, _csv_bytes(task_id, rows), "text/csv")},
        )
        page_status = None
        body_ok = None
        if upload.status_code == 303:
            page = client.get("/")
            page_status = page.status_code
            body_ok = (
                page.status_code == 200
                and (name in page.text if verify_name else True)
            )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "task_id": task_id,
        "filename": name,
        "rows": rows,
        "status": upload.status_code,
        "page_status": page_status,
        "body_ok": body_ok,
        "elapsed_ms": elapsed_ms,
        "passed": upload.status_code == 303 and body_ok is True,
    }


def _http_trials(run_id: str, base_url: str, config: LoadSoakConfig) -> tuple[list[dict], dict]:
    def task(task_id: int) -> dict[str, Any]:
        try:
            return _upload_once(
                base_url, task_id, config.http_rows, config.http_timeout_s
            )
        except Exception as exc:  # noqa: BLE001 - report every failure kind
            return {
                "task_id": task_id,
                "status": 0,
                "page_status": None,
                "body_ok": None,
                "elapsed_ms": None,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=config.http_workers) as pool:
        results = list(pool.map(task, range(config.http_uploads)))
    wall_s = time.perf_counter() - started

    records = []
    for result in results:
        records.append(
            _record(
                run_id,
                "L-02",
                "http_upload",
                f"upload_{result['task_id']:03d}",
                passed=result["passed"],
                expected=(
                    "303 redirect and the session page lists the uploaded file"
                ),
                details=result,
            )
        )

    latencies = [
        float(result["elapsed_ms"])
        for result in results
        if result.get("elapsed_ms") is not None
    ]
    statuses = Counter(str(result.get("status")) for result in results)
    metrics = {
        "requests": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "failed": sum(1 for result in results if not result["passed"]),
        "workers": config.http_workers,
        "rows": config.http_rows,
        "wall_s": round(wall_s, 3),
        "throughput_rps": round(len(results) / wall_s, 2) if wall_s else None,
        "status_counts": dict(sorted(statuses.items())),
        "latency_ms": _latency_summary(latencies),
    }
    return records, metrics


def _windows_process_tree(pid: int) -> list[int]:
    """Return ``pid`` plus its descendants using the Toolhelp snapshot."""
    import ctypes
    from ctypes import wintypes

    class ProcessEntry32(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.CHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32First.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32),
    ]
    kernel32.Process32First.restype = wintypes.BOOL
    kernel32.Process32Next.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessEntry32),
    ]
    kernel32.Process32Next.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if not snapshot or snapshot == wintypes.HANDLE(-1).value:
        return [pid]
    parents: dict[int, int] = {}
    try:
        entry = ProcessEntry32()
        entry.dwSize = ctypes.sizeof(ProcessEntry32)
        if kernel32.Process32First(snapshot, ctypes.byref(entry)):
            while True:
                parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
                if not kernel32.Process32Next(snapshot, ctypes.byref(entry)):
                    break
    finally:
        kernel32.CloseHandle(snapshot)

    tree = [pid]
    changed = True
    while changed:
        changed = False
        known = set(tree)
        for child, parent in parents.items():
            if parent in known and child not in known:
                tree.append(child)
                changed = True
    return tree


def _process_tree_rss_bytes(pid: int) -> int | None:
    """Sum resident memory for a process and any descendants it spawned."""
    if sys.platform == "win32":
        values = [
            _process_rss_bytes(item)
            for item in _windows_process_tree(pid)
        ]
        known = [value for value in values if value is not None]
        return sum(known) if known else None
    values = [_process_rss_bytes(pid)]
    children = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        child_ids = [int(item) for item in children.read_text().split()]
    except (OSError, ValueError):
        child_ids = []
    for child in child_ids:
        value = _process_tree_rss_bytes(child)
        if value is not None:
            values.append(value)
    known = [value for value in values if value is not None]
    return sum(known) if known else None


def _process_rss_bytes(pid: int) -> int | None:
    """Return one process's working set without requiring psutil."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.OpenProcess.argtypes = [
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            ]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

            handle = kernel32.OpenProcess(0x1000 | 0x0400 | 0x0010, False, pid)
            if not handle:
                return None
            try:
                counters = ProcessMemoryCounters()
                counters.cb = ctypes.sizeof(ProcessMemoryCounters)
                if not psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                ):
                    return None
                return int(counters.WorkingSetSize)
            finally:
                kernel32.CloseHandle(handle)

        status = Path(f"/proc/{pid}/status")
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except Exception:
        return None
    return None


def _soak_trials(
    run_id: str,
    base_url: str,
    pid: int,
    config: LoadSoakConfig,
) -> tuple[list[dict], dict]:
    warmup_failures: list[dict] = []
    warmup_requests = 0
    for index in range(max(0, config.soak_warmup_requests)):
        try:
            outcome = _upload_once(
                base_url,
                900_000 + index,
                config.soak_rows,
                config.http_timeout_s,
                verify_name=False,
            )
            warmup_requests += 1
            if not outcome["passed"]:
                warmup_failures.append(outcome)
        except Exception as exc:  # noqa: BLE001
            warmup_failures.append(
                {"task_id": 900_000 + index, "error": f"{type(exc).__name__}: {exc}"}
            )

    baseline = _process_tree_rss_bytes(pid)
    samples: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + max(0.1, config.soak_seconds)

    def sample() -> None:
        samples.append(
            {
                "elapsed_s": round(time.monotonic() - started, 2),
                "rss_bytes": _process_tree_rss_bytes(pid),
            }
        )

    sample()

    def worker(worker_id: int) -> dict[str, Any]:
        cycles = 0
        failures: list[dict] = []
        durations: list[float] = []
        while time.monotonic() < deadline:
            task_id = 5_000_000 + worker_id * 1_000_000 + cycles
            try:
                outcome = _upload_once(
                    base_url,
                    task_id,
                    config.soak_rows,
                    config.http_timeout_s,
                    verify_name=False,
                )
                durations.append(float(outcome["elapsed_ms"]))
                if not outcome["passed"]:
                    failures.append(outcome)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "task_id": task_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            cycles += 1
        return {
            "worker": worker_id,
            "cycles": cycles,
            "failures": failures,
            "durations": durations,
        }

    with ThreadPoolExecutor(max_workers=config.soak_workers) as pool:
        futures = [
            pool.submit(worker, worker_id)
            for worker_id in range(config.soak_workers)
        ]
        while True:
            finished, _pending = wait(
                futures, timeout=max(0.05, config.soak_sample_interval_s)
            )
            sample()
            if len(finished) == len(futures):
                break

    results = []
    for future in futures:
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001 - a worker failure is evidence
            results.append(
                {
                    "worker": len(results),
                    "cycles": 0,
                    "failures": [
                        {"error": f"{type(exc).__name__}: {exc}"}
                    ],
                    "durations": [],
                }
            )
    elapsed_s = time.monotonic() - started

    durations: list[float] = []
    failure_count = len(warmup_failures)
    request_count = warmup_requests
    for result in results:
        durations.extend(result["durations"])
        failure_count += len(result["failures"])
        request_count += result["cycles"]

    rss_values = [
        item["rss_bytes"] for item in samples if item["rss_bytes"] is not None
    ]
    rss_max = max(rss_values) if rss_values else None
    rss_final = samples[-1]["rss_bytes"] if samples else None
    growth = (
        rss_max - baseline
        if baseline is not None and rss_max is not None
        else None
    )
    growth_limit = int(config.soak_max_growth_mb * 1024 * 1024)
    no_failures = request_count > 0 and failure_count == 0
    rss_bounded = (
        baseline is not None
        and rss_max is not None
        and len(rss_values) >= 2
        and growth is not None
        and growth <= growth_limit
    )
    latency = _latency_summary(durations)
    details = {
        "duration_s": round(elapsed_s, 2),
        "requested_duration_s": config.soak_seconds,
        "workers": config.soak_workers,
        "rows": config.soak_rows,
        "requests": request_count,
        "warmup_requests": warmup_requests,
        "failure_count": failure_count,
        "failures": (warmup_failures + [
            failure
            for result in results
            for failure in result["failures"]
        ])[:5],
        "throughput_rps": round(request_count / elapsed_s, 2) if elapsed_s else None,
        "latency_ms": latency,
    }
    rss_details = {
        **details,
        "rss_baseline_bytes": baseline,
        "rss_max_bytes": rss_max,
        "rss_final_bytes": rss_final,
        "rss_growth_bytes": growth,
        "rss_growth_mb": round(growth / 1048576, 2) if growth is not None else None,
        "rss_limit_mb": config.soak_max_growth_mb,
        "samples": [
            {
                "elapsed_s": item["elapsed_s"],
                "rss_mb": (
                    round(item["rss_bytes"] / 1048576, 2)
                    if item["rss_bytes"] is not None
                    else None
                ),
            }
            for item in samples
        ],
    }
    records = [
        _record(
            run_id,
            "L-03",
            "sustained_soak",
            "soak_no_request_failures",
            passed=no_failures,
            expected="every upload and page load during the soak succeeds",
            details=details,
        ),
        _record(
            run_id,
            "L-03",
            "sustained_soak",
            "soak_rss_growth_bounded",
            passed=rss_bounded,
            expected=(
                f"peak RSS growth stays at or below {config.soak_max_growth_mb:g} MB"
            ),
            details=rss_details,
        ),
    ]
    metrics = {
        "requests": request_count,
        "failures": failure_count,
        "workers": config.soak_workers,
        "rows": config.soak_rows,
        "duration_s": round(elapsed_s, 2),
        "throughput_rps": details["throughput_rps"],
        "latency_ms": latency,
        "rss_baseline_mb": (
            round(baseline / 1048576, 2) if baseline is not None else None
        ),
        "rss_max_mb": round(rss_max / 1048576, 2) if rss_max is not None else None,
        "rss_final_mb": (
            round(rss_final / 1048576, 2) if rss_final is not None else None
        ),
        "rss_growth_mb": rss_details["rss_growth_mb"],
        "rss_limit_mb": config.soak_max_growth_mb,
        "sample_count": len(samples),
        "no_failures": no_failures,
        "rss_bounded": rss_bounded,
    }
    return records, metrics


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _cache_answer(key: str, payload_kb: int) -> str:
    return f"{key}:" + ("x" * (payload_kb * 1024))


def _cache_trials(
    run_id: str, root: Path, config: LoadSoakConfig
) -> tuple[list[dict], dict]:
    settings = get_settings()
    snapshot = {name: getattr(settings, name) for name in _CACHE_SETTINGS}
    artifacts = root / "cache-churn" / "artifacts"
    cache_root = artifacts / "cache"
    settings.artifacts_dir = artifacts
    settings.query_cache_enabled = True
    settings.query_cache_ttl_days = 14
    settings.query_cache_max_mb = config.cache_max_mb

    counters = {"hits": 0, "misses": 0}
    corrupt: list[str] = []
    counter_lock = threading.Lock()

    try:
        seed_keys = [f"seed-{index}" for index in range(2)]
        for key in seed_keys:
            cache.store_cached_response(
                key,
                AnalysisResponse(
                    question="cache churn",
                    answer=_cache_answer(key, config.cache_payload_kb),
                ),
            )

        keys = [f"churn-{index:03d}" for index in range(config.cache_writes)]
        # Seed keys are already published, so readers see at least one hit even
        # in a very small smoke configuration.
        reader_keys = seed_keys + keys

        def writer(worker_id: int) -> None:
            for index in range(worker_id, config.cache_writes, config.cache_writers):
                key = keys[index]
                cache.store_cached_response(
                    key,
                    AnalysisResponse(
                        question="cache churn",
                        answer=_cache_answer(key, config.cache_payload_kb),
                    ),
                )
                time.sleep(0.001)

        def reader(reader_id: int) -> None:
            for index in range(config.cache_reads):
                key = reader_keys[
                    (reader_id * config.cache_reads + index) % len(reader_keys)
                ]
                loaded = cache.load_cached_response(key, root / "cache-out")
                with counter_lock:
                    if loaded is None:
                        counters["misses"] += 1
                    elif loaded.answer.startswith(f"{key}:"):
                        counters["hits"] += 1
                    else:
                        corrupt.append(key)
                time.sleep(0.0005)

        threads = [
            threading.Thread(target=writer, args=(worker_id,))
            for worker_id in range(config.cache_writers)
        ] + [
            threading.Thread(target=reader, args=(reader_id,))
            for reader_id in range(config.cache_readers)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        size_before_sweep = _tree_size(cache_root)
        cache.sweep_cache()
        size_after_sweep = _tree_size(cache_root)
        temps = [
            str(path.relative_to(cache_root))
            for path in cache_root.rglob("*.tmp")
        ] if cache_root.exists() else []
        limit_bytes = max(1, config.cache_max_mb) * 1024 * 1024
        payload_bytes = max(
            len(_cache_answer(key, config.cache_payload_kb).encode("utf-8"))
            for key in reader_keys
        )
        # store_cached_response sweeps before publishing, so the cache may
        # briefly contain the budget plus the one payload being written.
        size_bound = limit_bytes + payload_bytes + 16_384
        reads_attempted = config.cache_readers * config.cache_reads

        size_ok = (
            size_before_sweep <= size_bound
            and size_after_sweep <= limit_bytes
            and not temps
        )
        access_ok = (
            not corrupt
            and not temps
            and counters["hits"] > 0
            and counters["hits"] + counters["misses"] == reads_attempted
        )
        details = {
            "writes": config.cache_writes,
            "writers": config.cache_writers,
            "readers": config.cache_readers,
            "reads": reads_attempted,
            "hits": counters["hits"],
            "misses": counters["misses"],
            "corrupt_reads": len(corrupt),
            "size_before_sweep_bytes": size_before_sweep,
            "size_after_sweep_bytes": size_after_sweep,
            "limit_bytes": limit_bytes,
            "size_bound_bytes": size_bound,
            "payload_kb": config.cache_payload_kb,
            "temp_files": temps[:5],
        }
        metrics = {
            **details,
            "size_before_sweep_mb": round(size_before_sweep / 1048576, 2),
            "size_after_sweep_mb": round(size_after_sweep / 1048576, 2),
            "limit_mb": config.cache_max_mb,
            "size_ok": size_ok,
            "access_ok": access_ok,
        }
        records = [
            _record(
                run_id,
                "L-04",
                "result_cache",
                "cache_size_bounded",
                passed=size_ok,
                expected=(
                    "after churn the cache is within one payload of the budget; "
                    "an explicit sweep brings it within budget"
                ),
                details=details,
            ),
            _record(
                run_id,
                "L-04",
                "result_cache",
                "concurrent_access_safe",
                passed=access_ok,
                expected=(
                    "readers see a complete payload or a safe miss; no corrupt "
                    "reads or leftover temp files"
                ),
                details=details,
            ),
        ]
        return records, metrics
    finally:
        for name, value in snapshot.items():
            setattr(settings, name, value)


def _claim_summary(
    records: list[dict], claim_id: str, statement: str, scope: str
) -> dict[str, Any]:
    scored = [
        record
        for record in records
        if record["claim_id"] == claim_id and record["expectation"] == "scored"
    ]
    passed = sum(1 for record in scored if record["passed"])
    low, high = wilson_ci(passed, len(scored)) if scored else (0.0, 0.0)
    status = (
        "supported"
        if scored and passed == len(scored)
        else ("gap" if scored else "no-data")
    )
    return {
        "id": claim_id,
        "statement": statement,
        "scope": scope,
        "status": status,
        "trials": len(scored),
        "passed": passed,
        "failed": len(scored) - passed,
        "rate": (passed / len(scored)) if scored else None,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def run_load_soak(
    run_id: str,
    root: Path,
    config: LoadSoakConfig | None = None,
) -> tuple[list[dict], dict]:
    config = config or LoadSoakConfig()
    settings = get_settings()
    snapshot = {name: getattr(settings, name) for name in _CACHE_SETTINGS}
    work_root = root / "work"
    if work_root.exists():
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    harness_errors: list[str] = []
    metrics: dict[str, Any] = {}
    try:
        records.extend(_analysis_trials(run_id, config))
        cache_records, cache_metrics = _cache_trials(run_id, work_root, config)
        records.extend(cache_records)
        metrics["cache"] = cache_metrics

        try:
            with _server(work_root) as server:
                http_records, http_metrics = _http_trials(
                    run_id, server.base_url, config
                )
                records.extend(http_records)
                metrics["http"] = http_metrics
                soak_records, soak_metrics = _soak_trials(
                    run_id, server.base_url, server.process.pid, config
                )
                records.extend(soak_records)
                metrics["soak"] = soak_metrics
        except Exception as exc:  # noqa: BLE001 - a server failure is evidence
            harness_errors.append(f"http/soak harness: {type(exc).__name__}: {exc}")
    finally:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
        for name, value in snapshot.items():
            setattr(settings, name, value)

    analysis_records = [
        record for record in records if record["claim_id"] == "L-01"
    ]
    metrics["analysis"] = {
        "fixtures": len({
            record["details"]["fixture_id"] for record in analysis_records
        }),
        "rounds": config.analysis_rounds,
        "repeats": config.analysis_repeats,
        "workers": config.analysis_workers,
        "executions": len(analysis_records) * config.analysis_repeats,
        "trials": len(analysis_records),
        "passed": sum(1 for record in analysis_records if record["passed"]),
        "max_elapsed_ms": (
            max(
                (
                    record["details"].get("elapsed_ms") or 0.0
                    for record in analysis_records
                ),
                default=0.0,
            )
        ),
    }

    claims = [
        _claim_summary(
            records,
            "L-01",
            "Concurrent core analysis executions are deterministic and error-free.",
            (
                f"{config.analysis_rounds} round(s), "
                f"{config.analysis_repeats} concurrent repeats per fixture"
            ),
        ),
        _claim_summary(
            records,
            "L-02",
            "Concurrent HTTP uploads succeed and load the expected dataset.",
            (
                f"{config.http_uploads} uploads, {config.http_workers} clients, "
                f"{config.http_rows} rows per CSV"
            ),
        ),
        _claim_summary(
            records,
            "L-03",
            "A sustained upload soak remains error-free with bounded RSS growth.",
            (
                f"{config.soak_seconds:g}s, {config.soak_workers} clients, "
                f"{config.soak_max_growth_mb:g} MB growth limit"
            ),
        ),
        _claim_summary(
            records,
            "L-04",
            "Result-cache churn respects its size budget after sweeping and "
            "never serves corrupt reads.",
            (
                f"{config.cache_writes} writes, {config.cache_readers} readers, "
                f"{config.cache_max_mb} MB budget"
            ),
        ),
    ]
    scored = [record for record in records if record["expectation"] == "scored"]
    counts = {
        "records": len(records),
        "scored": len(scored),
        "passed": sum(1 for record in scored if record["passed"]),
        "failed": sum(1 for record in scored if not record["passed"]),
        "harness_errors": len(harness_errors),
    }
    summary = {
        "schema": "load-soak-1",
        "benchmark": "load_soak",
        "run_id": run_id,
        "config": {
            "analysis_rounds": config.analysis_rounds,
            "analysis_repeats": config.analysis_repeats,
            "analysis_workers": config.analysis_workers,
            "analysis_fixture_limit": config.analysis_fixture_limit,
            "http_uploads": config.http_uploads,
            "http_workers": config.http_workers,
            "http_rows": config.http_rows,
            "http_timeout_s": config.http_timeout_s,
            "soak_seconds": config.soak_seconds,
            "soak_workers": config.soak_workers,
            "soak_rows": config.soak_rows,
            "soak_sample_interval_s": config.soak_sample_interval_s,
            "soak_max_growth_mb": config.soak_max_growth_mb,
            "soak_warmup_requests": config.soak_warmup_requests,
            "cache_writes": config.cache_writes,
            "cache_writers": config.cache_writers,
            "cache_readers": config.cache_readers,
            "cache_reads": config.cache_reads,
            "cache_max_mb": config.cache_max_mb,
            "cache_payload_kb": config.cache_payload_kb,
        },
        "counts": counts,
        "metrics": metrics,
        "claims": claims,
        "harness_errors": harness_errors,
        "status": (
            "ok"
            if not harness_errors
            and all(claim["status"] == "supported" for claim in claims)
            else "attention"
        ),
    }
    return records, summary
