"""Durability and fault-injection evidence for the trust core.

This harness exercises the persistence boundaries that a successful analysis
depends on: the hash-chained audit ledger, the result cache, alert-runner
state, and readiness checks. It deliberately injects process-style faults
(concurrent appends, torn writes, corrupt payloads, failed webhooks) and records
what the implementation does. Raw JSONL is the source of truth; the summary and
Markdown report are derived views.

The harness is internal evidence tooling. It never changes shipped defaults and
it restores the process settings it mutates before returning.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.ab_stats import wilson_ci
from insight import audit
from insight.domain.visualization import AnalysisResponse, PlotArtifact
from insight.settings import Settings, get_settings
from tools.alert_runner import Monitor, Runner
from webapp.readiness import readiness_report

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class DurabilityConfig:
    ledger_roundtrips: int = 5
    ledger_records: int = 20
    tamper_records: int = 10
    concurrent_rounds: int = 3
    concurrent_writers: int = 4
    concurrent_records: int = 10
    cache_race_writes: int = 80
    cache_race_readers: int = 4
    cache_race_reads: int = 120


def config_for_mode(mode: str) -> DurabilityConfig:
    if mode == "full":
        return DurabilityConfig()
    return DurabilityConfig(
        ledger_roundtrips=3,
        ledger_records=10,
        tamper_records=8,
        concurrent_rounds=2,
        concurrent_writers=3,
        concurrent_records=5,
        cache_race_writes=40,
        cache_race_readers=2,
        cache_race_reads=60,
    )


_SETTINGS_FIELDS = (
    "audit_ledger",
    "audit_ledger_path",
    "artifacts_dir",
    "query_cache_enabled",
    "query_cache_ttl_days",
    "query_cache_max_mb",
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
    expectation: str = "scored",
) -> dict[str, Any]:
    return {
        "schema": "durability-1",
        "run_id": run_id,
        "suite": "durability",
        "trial_id": f"{claim_id}-{scenario}",
        "claim_id": claim_id,
        "component": component,
        "scenario": scenario,
        "expectation": expectation,
        "expected": expected,
        "passed": bool(passed),
        "details": details or {},
    }


def _write_raw(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _append_probes(path: Path, count: int) -> list[dict]:
    settings = get_settings()
    settings.audit_ledger = True
    settings.audit_ledger_path = path
    records = []
    for index in range(count):
        records.append(
            audit.append_event(
                {"event": "durability_probe", "index": index, "probe": True}
            )
        )
    return records


def _ledger_roundtrips(
    run_id: str, root: Path, config: DurabilityConfig
) -> list[dict]:
    records = []
    for round_index in range(config.ledger_roundtrips):
        path = root / f"roundtrip-{round_index}" / "ledger.jsonl"
        started = time.perf_counter()
        appended = _append_probes(path, config.ledger_records)
        result = audit.verify_ledger(path)
        stored = list(audit.read_events(path))
        expected_seqs = list(range(1, config.ledger_records + 1))
        passed = (
            result["ok"]
            and result["records"] == config.ledger_records
            and len(stored) == config.ledger_records
            and [record["seq"] for record in stored] == expected_seqs
            and appended[-1]["hash"] == result["head"]
        )
        records.append(
            _record(
                run_id,
                "D-01",
                "audit_ledger",
                f"roundtrip_{round_index}",
                passed=passed,
                expected="append, chain, and verify round-trip",
                details={
                    "records": result["records"],
                    "ok": result["ok"],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                    "head": result["head"],
                },
            )
        )
    return records


def _tamper_lines(lines: list[str], mode: str) -> tuple[list[str], int | None]:
    working = list(lines)
    if mode == "modify_field":
        record = json.loads(working[3])
        record["index"] = 999
        working[3] = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        )
        return working, 4
    if mode == "modify_hash":
        record = json.loads(working[6])
        record["hash"] = "0" * 64
        working[6] = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        )
        return working, 7
    if mode == "reorder":
        working[2], working[3] = working[3], working[2]
        return working, 3
    if mode == "delete_middle":
        working.pop(5)
        return working, 6
    if mode == "duplicate_line":
        working.insert(3, working[2])
        return working, 4
    if mode == "truncate_tail":
        working[-1] = working[-1][:8]
        return working, len(working)
    if mode == "append_garbage":
        working.append("{not valid json}")
        return working, len(working)
    raise ValueError(f"unknown tamper mode: {mode}")


def _ledger_tamper(run_id: str, root: Path, config: DurabilityConfig) -> list[dict]:
    modes = (
        "modify_field",
        "modify_hash",
        "reorder",
        "delete_middle",
        "duplicate_line",
        "truncate_tail",
        "append_garbage",
    )
    records = []
    for mode in modes:
        path = root / "tamper" / mode / "ledger.jsonl"
        _append_probes(path, config.tamper_records)
        lines = path.read_text(encoding="utf-8").splitlines()
        tampered, expected_broken = _tamper_lines(lines, mode)
        _write_raw(path, "\n".join(tampered) + "\n")
        result = audit.verify_ledger(path)
        passed = (
            result["ok"] is False
            and result["broken_at"] == expected_broken
            and result["reason"] is not None
        )
        records.append(
            _record(
                run_id,
                "D-02",
                "audit_ledger",
                f"tamper_{mode}",
                passed=passed,
                expected=f"detected at line {expected_broken}",
                details={
                    "broken_at": result["broken_at"],
                    "reason": result["reason"],
                    "records": result["records"],
                },
            )
        )

    boundary_path = root / "tamper" / "delete_complete_tail" / "ledger.jsonl"
    _append_probes(boundary_path, config.tamper_records)
    lines = boundary_path.read_text(encoding="utf-8").splitlines()
    _write_raw(boundary_path, "\n".join(lines[:-1]) + "\n")
    result = audit.verify_ledger(boundary_path)
    boundary_passed = result["ok"] and result["records"] == config.tamper_records - 1
    records.append(
        _record(
            run_id,
            "D-02",
            "audit_ledger",
            "delete_complete_tail",
            passed=boundary_passed,
            expected="undetectable by the chain alone; documented boundary",
            expectation="known_boundary",
            details={
                "ok": result["ok"],
                "records": result["records"],
                "why": (
                    "A complete tail deletion leaves a shorter internally valid "
                    "chain; detecting it requires an externally pinned head."
                ),
            },
        )
    )
    return records


def _concurrent_round(
    run_id: str,
    path: Path,
    round_index: int,
    config: DurabilityConfig,
) -> dict:
    env = dict(os.environ)
    env["INSIGHT_AUDIT_LEDGER"] = "on"
    env["INSIGHT_AUDIT_LEDGER_PATH"] = str(path)
    env["PYTHONPATH"] = str(ROOT)
    processes = []
    started = time.perf_counter()
    for writer in range(config.concurrent_writers):
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "evaluation.bench.ledger_worker",
                    "--count",
                    str(config.concurrent_records),
                    "--worker",
                    f"w{writer}",
                ],
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    failures = []
    for writer, process in enumerate(processes):
        stdout, stderr = process.communicate(timeout=120)
        if process.returncode != 0:
            failures.append(
                {
                    "worker": writer,
                    "returncode": process.returncode,
                    "stderr": stderr[-500:],
                    "stdout": stdout[-500:],
                }
            )
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    result = audit.verify_ledger(path)
    stored = list(audit.read_events(path)) if path.exists() else []
    total = config.concurrent_writers * config.concurrent_records
    seqs = [record.get("seq") for record in stored]
    event_ids = [record.get("event_id") for record in stored]
    worker_counts: dict[str, int] = {}
    for record in stored:
        worker = str(record.get("worker"))
        worker_counts[worker] = worker_counts.get(worker, 0) + 1
    complete_worker_counts = all(
        worker_counts.get(f"w{writer}") == config.concurrent_records
        for writer in range(config.concurrent_writers)
    )
    passed = (
        not failures
        and result["ok"]
        and len(stored) == total
        and seqs == list(range(1, total + 1))
        and len(set(event_ids)) == total
        and complete_worker_counts
    )
    return {
        "round": round_index,
        "passed": passed,
        "elapsed_ms": elapsed_ms,
        "records": len(stored),
        "expected_records": total,
        "ok": result["ok"],
        "broken_at": result["broken_at"],
        "worker_counts": worker_counts,
        "failures": failures,
    }


def _ledger_concurrency(
    run_id: str, root: Path, config: DurabilityConfig
) -> list[dict]:
    records = []
    for round_index in range(config.concurrent_rounds):
        path = root / "concurrent" / f"round-{round_index}" / "ledger.jsonl"
        details = _concurrent_round(run_id, path, round_index, config)
        records.append(
            _record(
                run_id,
                "D-03",
                "audit_ledger",
                f"concurrent_round_{round_index}",
                passed=details["passed"],
                expected=(
                    f"{config.concurrent_writers} writers x "
                    f"{config.concurrent_records} records, no loss"
                ),
                details=details,
            )
        )
    return records


def _ledger_torn_tail(run_id: str, root: Path) -> list[dict]:
    path = root / "torn-tail" / "ledger.jsonl"
    _append_probes(path, 5)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"event": "torn", "seq": 6')

    result = audit.verify_ledger(path)
    blocked = False
    error_type = None
    try:
        audit.append_event({"event": "after-torn"})
    except Exception as exc:  # noqa: BLE001 - we record the type
        blocked = True
        error_type = type(exc).__name__

    detected = (
        result["ok"] is False
        and result["broken_at"] == 6
        and blocked
        and error_type == "LedgerCorruptError"
    )
    records = [
        _record(
            run_id,
            "D-04",
            "audit_ledger",
            "torn_tail_detected_and_append_blocked",
            passed=detected,
            expected="verify fails at the torn line; append raises LedgerCorruptError",
            details={
                "broken_at": result["broken_at"],
                "reason": result["reason"],
                "append_blocked": blocked,
                "error_type": error_type,
            },
        )
    ]

    text = path.read_text(encoding="utf-8")
    _write_raw(path, text[: text.rfind("\n") + 1])
    recovered = audit.append_event({"event": "after-manual-truncate"})
    after = audit.verify_ledger(path)
    records.append(
        _record(
            run_id,
            "D-05",
            "audit_ledger",
            "torn_tail_manual_truncate_recovers",
            passed=after["ok"] and after["records"] == 6 and recovered["seq"] == 6,
            expected="truncate the partial line, then append resumes the chain",
            details={
                "ok": after["ok"],
                "records": after["records"],
                "head": after["head"],
            },
        )
    )
    return records


def _response(answer: str) -> AnalysisResponse:
    return AnalysisResponse(question="durability", answer=answer)


def _cache_trials(run_id: str, root: Path, config: DurabilityConfig) -> list[dict]:
    from insight import cache

    settings = get_settings()
    original_dir = settings.artifacts_dir
    settings.artifacts_dir = root / "cache-artifacts"
    settings.query_cache_enabled = True
    settings.query_cache_ttl_days = 14
    settings.query_cache_max_mb = 512
    records = []

    try:
        key = "atomic"
        cache.store_cached_response(key, _response("old-complete"))
        cache_root = settings.artifacts_dir / "cache"
        temp = cache_root / f".{key}.deadbeef.tmp"
        temp.write_text("{partial", encoding="utf-8")
        loaded = cache.load_cached_response(key, root / "cache-out")
        records.append(
            _record(
                run_id,
                "D-06",
                "result_cache",
                "crash_before_publish_preserves_old_entry",
                passed=loaded is not None and loaded.answer == "old-complete",
                expected="readers see the previous complete payload",
                details={
                    "loaded": loaded is not None,
                    "answer": getattr(loaded, "answer", None),
                    "stray_temp": temp.name,
                },
            )
        )
        temp.unlink(missing_ok=True)

        key = "corrupt"
        cache.store_cached_response(key, _response("before-corruption"))
        (cache_root / f"{key}.json").write_text("{not json", encoding="utf-8")
        try:
            loaded = cache.load_cached_response(key, root / "cache-out")
            raised = None
        except Exception as exc:  # noqa: BLE001
            loaded = None
            raised = type(exc).__name__
        records.append(
            _record(
                run_id,
                "D-06",
                "result_cache",
                "corrupt_payload_returns_miss",
                passed=loaded is None and raised is None,
                expected="return None, do not raise",
                details={"loaded": loaded, "raised": raised},
            )
        )

        source = root / "chart.png"
        source.write_bytes(b"png-bytes")
        key = "missing-blob"
        plotted = AnalysisResponse(
            question="durability",
            answer="with-plot",
            plots=[
                PlotArtifact(
                    path=str(source), title="chart", chart_type="bar", step_id=0
                )
            ],
        )
        cache.store_cached_response(key, plotted)
        blob_dir = cache_root / key
        for blob in blob_dir.glob("*"):
            blob.unlink()
        loaded = cache.load_cached_response(key, root / "cache-out")
        records.append(
            _record(
                run_id,
                "D-06",
                "result_cache",
                "missing_blob_returns_miss",
                passed=loaded is None,
                expected="return None when a referenced artifact is absent",
                details={"loaded": loaded is not None},
            )
        )

        key = "race"
        cache.store_cached_response(key, _response("seed"))
        counts = {"valid": 0, "misses": 0}
        corrupt: list[str] = []
        count_lock = threading.Lock()
        started = time.perf_counter()

        def writer() -> None:
            for index in range(config.cache_race_writes):
                cache.store_cached_response(key, _response(f"v{index}"))
                time.sleep(0.001)

        def reader() -> None:
            for _ in range(config.cache_race_reads):
                loaded = cache.load_cached_response(key, root / "cache-out")
                with count_lock:
                    if loaded is None:
                        # Publication uses os.replace. On Windows a reader can
                        # race the replacement and get a safe miss; a miss is a
                        # fallback path, not corrupt data.
                        counts["misses"] += 1
                    elif not loaded.answer.startswith(("seed", "v")):
                        corrupt.append(loaded.answer)
                    else:
                        counts["valid"] += 1
                time.sleep(0.0005)

        writer_thread = threading.Thread(target=writer)
        reader_threads = [
            threading.Thread(target=reader) for _ in range(config.cache_race_readers)
        ]
        writer_thread.start()
        for thread in reader_threads:
            thread.start()
        for thread in reader_threads:
            thread.join()
        writer_thread.join()
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        expected_reads = config.cache_race_readers * config.cache_race_reads
        valid_reads = counts["valid"]
        temps = list(cache_root.glob(f".{key}.*.tmp"))
        records.append(
            _record(
                run_id,
                "D-06",
                "result_cache",
                "concurrent_read_write_is_atomic",
                passed=not corrupt and valid_reads > 0 and not temps,
                expected=(
                    "readers see a complete old/new payload or a safe miss; "
                    "no corrupt reads"
                ),
                details={
                    "valid_reads": valid_reads,
                    "misses": counts["misses"],
                    "corrupt_reads": len(corrupt),
                    "hit_rate": (
                        valid_reads / expected_reads if expected_reads else None
                    ),
                    "expected_reads": expected_reads,
                    "temp_files": [path.name for path in temps],
                    "elapsed_ms": elapsed_ms,
                },
            )
        )

        blocker = root / "cache-blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        settings.artifacts_dir = blocker / "nested"
        raised = None
        try:
            cache.store_cached_response("blocked", _response("blocked"))
        except Exception as exc:  # noqa: BLE001
            raised = type(exc).__name__
        records.append(
            _record(
                run_id,
                "D-06",
                "result_cache",
                "write_failure_is_nonfatal",
                passed=raised is None,
                expected="swallow the write failure and keep serving",
                details={"raised": raised},
            )
        )
    finally:
        settings.artifacts_dir = original_dir
    return records


def _alert_runner_trials(run_id: str, root: Path) -> list[dict]:
    datasets = {"sales.csv": pd.DataFrame({"revenue": [100, 220, 150, 300]})}
    monitor = Monitor(
        name="durability_probe",
        query="SELECT * FROM sales",
        payload_template="rows={{ rows }}",
    )
    records = []

    state = root / "alert" / "restart.json"
    first = Runner(
        monitors=[monitor], datasets=datasets, dry_run=True, state_path=state
    )
    first.load_state()
    first.cycle()
    second = Runner(
        monitors=[monitor], datasets=datasets, dry_run=True, state_path=state
    )
    second.load_state()
    second.cycle()
    records.append(
        _record(
            run_id,
            "D-07",
            "alert_runner",
            "restart_does_not_repeat_delivered_alert",
            passed=first.posted == 1 and second.posted == 0 and monitor.name in second.state,
            expected="persist triggered state across a restart",
            details={
                "first_posted": first.posted,
                "second_posted": second.posted,
                "restored_state": sorted(second.state),
            },
        )
    )

    corrupt = root / "alert" / "corrupt.json"
    _write_raw(corrupt, "{not valid json")
    corrupted_runner = Runner(monitors=[], state_path=corrupt)
    raised = None
    try:
        corrupted_runner.load_state()
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    records.append(
        _record(
            run_id,
            "D-07",
            "alert_runner",
            "corrupt_state_falls_back_to_empty",
            passed=raised is None and corrupted_runner.state == set(),
            expected="warn and start empty; a triggered alert may repost",
            expectation="known_boundary",
            details={
                "raised": raised,
                "state": sorted(corrupted_runner.state),
                "why": (
                    "The state file is a delivery ledger, not a hash chain; "
                    "corruption is treated as first run so monitoring continues."
                ),
            },
        )
    )

    blocker = root / "alert-blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    blocked_runner = Runner(
        monitors=[monitor], datasets=datasets, state_path=blocker / "nested" / "state.json"
    )
    raised = None
    try:
        blocked_runner.save_state()
    except Exception as exc:  # noqa: BLE001
        raised = type(exc).__name__
    records.append(
        _record(
            run_id,
            "D-07",
            "alert_runner",
            "state_write_failure_is_nonfatal",
            passed=raised is None,
            expected="warn and continue when state cannot be written",
            details={"raised": raised},
        )
    )

    attempts: list[dict] = []

    def fail_post(url: str, payload: dict) -> None:
        attempts.append(payload)
        raise urllib.error.URLError("webhook down")

    retry_runner = Runner(
        monitors=[monitor],
        datasets=datasets,
        webhook_url="https://example.invalid/hook",
        state_path=root / "alert" / "retry.json",
    )
    retry_runner.post = fail_post
    retry_runner.cycle()
    retry_runner.cycle()
    records.append(
        _record(
            run_id,
            "D-07",
            "alert_runner",
            "failed_delivery_is_retried",
            passed=len(attempts) == 2 and monitor.name not in retry_runner.state,
            expected="do not advance state when the webhook post fails",
            details={
                "attempts": len(attempts),
                "state": sorted(retry_runner.state),
                "posted": retry_runner.posted,
            },
        )
    )

    recovery_monitor = Monitor(
        name="recovery_probe",
        query="SELECT * FROM sales",
        metric_column="revenue",
        op=">",
        threshold=99999,
    )
    recovery_attempts: list[dict] = []

    def fail_recovery(url: str, payload: dict) -> None:
        recovery_attempts.append(payload)
        raise urllib.error.URLError("webhook down")

    recovery_runner = Runner(
        monitors=[recovery_monitor],
        datasets=datasets,
        notify_recovery=True,
        webhook_url="https://example.invalid/hook",
        state_path=root / "alert" / "recovery-retry.json",
    )
    recovery_runner.post = fail_recovery
    recovery_runner.state.add(recovery_monitor.name)
    recovery_runner.cycle()
    recovery_runner.cycle()
    records.append(
        _record(
            run_id,
            "D-07",
            "alert_runner",
            "failed_recovery_is_retried",
            passed=(
                len(recovery_attempts) == 2
                and recovery_monitor.name in recovery_runner.state
            ),
            expected="keep triggered state until the recovery notice is delivered",
            details={
                "attempts": len(recovery_attempts),
                "state": sorted(recovery_runner.state),
            },
        )
    )
    return records


def _readiness_trials(run_id: str, root: Path) -> list[dict]:
    records = []
    writable = root / "ready-artifacts"
    ready_settings = Settings(
        require_user_key=True, artifacts_dir=writable, audit_ledger=False
    )
    ready, report = readiness_report(ready_settings)
    records.append(
        _record(
            run_id,
            "D-08",
            "readiness",
            "writable_artifacts_ready",
            passed=ready is True and report["artifacts_writable"] is True,
            expected="ready when the artifacts directory is writable",
            details=report,
        )
    )

    blocker = root / "ready-blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    blocked_settings = Settings(
        require_user_key=True,
        artifacts_dir=blocker / "nested",
        audit_ledger=False,
    )
    ready, report = readiness_report(blocked_settings)
    records.append(
        _record(
            run_id,
            "D-08",
            "readiness",
            "unwritable_artifacts_not_ready",
            passed=ready is False and report["artifacts_writable"] is False,
            expected="not ready when artifacts cannot be written",
            details=report,
        )
    )

    ledger_settings = Settings(
        require_user_key=True,
        artifacts_dir=writable,
        audit_ledger=True,
        audit_ledger_path=blocker / "nested" / "ledger.jsonl",
    )
    ready, report = readiness_report(ledger_settings)
    records.append(
        _record(
            run_id,
            "D-08",
            "readiness",
            "unwritable_ledger_not_ready",
            passed=ready is False and report["audit_ledger_writable"] is False,
            expected="not ready when the audit ledger cannot be written",
            details=report,
        )
    )
    return records


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


def run_durability(
    run_id: str,
    root: Path,
    config: DurabilityConfig | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    config = config or DurabilityConfig()
    settings = get_settings()
    snapshot = {name: getattr(settings, name) for name in _SETTINGS_FIELDS}
    work_root = root / "work"
    if work_root.exists():
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    harness_errors: list[str] = []
    try:
        records.extend(_ledger_roundtrips(run_id, work_root, config))
        records.extend(_ledger_tamper(run_id, work_root, config))
        records.extend(_ledger_concurrency(run_id, work_root, config))
        records.extend(_ledger_torn_tail(run_id, work_root))
        records.extend(_cache_trials(run_id, work_root, config))
        records.extend(_alert_runner_trials(run_id, work_root))
        records.extend(_readiness_trials(run_id, work_root))
    except Exception as exc:  # noqa: BLE001 - a harness failure is evidence
        harness_errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
        for name, value in snapshot.items():
            setattr(settings, name, value)

    claims = [
        _claim_summary(
            records,
            "D-01",
            "Ledger append/verify round-trips preserve every record and the chain head.",
            "sequential appends, fresh ledgers",
        ),
        _claim_summary(
            records,
            "D-02",
            "Ledger tampering is detected at the affected record.",
            "field/hash edits, reorder, delete, duplicate, torn and appended lines",
        ),
        _claim_summary(
            records,
            "D-03",
            "Concurrent OS processes append without lost updates or a broken chain.",
            f"{config.concurrent_writers} writers x "
            f"{config.concurrent_records} records per round",
        ),
        _claim_summary(
            records,
            "D-04",
            "A torn tail is detected and appends fail closed.",
            "partial JSON line at the ledger tail",
        ),
        _claim_summary(
            records,
            "D-05",
            "Manual truncation of a torn tail allows appends to resume.",
            "operator recovery procedure",
        ),
        _claim_summary(
            records,
            "D-06",
            "Cache publication is atomic and corruption degrades to a cache miss.",
            "crash before publish, corrupt payload, missing blob, "
            "concurrent readers (safe misses counted)",
        ),
        _claim_summary(
            records,
            "D-07",
            "Alert-runner state survives restart and delivery failures are retried.",
            "restart, corrupt state, write failure, webhook failure",
        ),
        _claim_summary(
            records,
            "D-08",
            "Readiness reports writable artifacts and ledger requirements.",
            "writable and blocked paths",
        ),
    ]
    scored = [record for record in records if record["expectation"] == "scored"]
    boundaries = [
        record for record in records if record["expectation"] == "known_boundary"
    ]
    counts = {
        "records": len(records),
        "scored": len(scored),
        "passed": sum(1 for record in scored if record["passed"]),
        "failed": sum(1 for record in scored if not record["passed"]),
        "known_boundaries": len(boundaries),
        "harness_errors": len(harness_errors),
    }
    summary = {
        "schema": "durability-1",
        "benchmark": "durability",
        "run_id": run_id,
        "config": {
            "ledger_roundtrips": config.ledger_roundtrips,
            "ledger_records": config.ledger_records,
            "tamper_records": config.tamper_records,
            "concurrent_rounds": config.concurrent_rounds,
            "concurrent_writers": config.concurrent_writers,
            "concurrent_records": config.concurrent_records,
            "cache_race_writes": config.cache_race_writes,
            "cache_race_readers": config.cache_race_readers,
            "cache_race_reads": config.cache_race_reads,
        },
        "counts": counts,
        "claims": claims,
        "known_boundaries": boundaries,
        "harness_errors": harness_errors,
        "status": (
            "ok"
            if not harness_errors and all(claim["status"] == "supported" for claim in claims)
            else "attention"
        ),
    }
    return records, summary
