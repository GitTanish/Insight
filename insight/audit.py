"""Append-only evidence ledger for plan reviews and completed analyses.

The ledger is a JSONL file: one self-contained record per line. Records are
hash-chained (``prev`` -> ``hash``) so edits, reordering, and mid-ledger
deletion are detectable with :func:`verify_ledger`. Writes are serialised with
a process lock plus an OS file lock, so multiple workers can append to the same
file.

The ledger is independent of the query cache and chart artifacts: nothing in
the application rewrites or evicts it. It is disabled unless
``INSIGHT_AUDIT_LEDGER=on``. A failed ledger write is logged but never blocks
the analysis response.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from insight.domain.dataset import DatasetProfile
from insight.domain.visualization import AnalysisResponse
from insight.settings import get_settings

SCHEMA_VERSION = 1
LOG = logging.getLogger("insight.audit")

try:  # POSIX
    import fcntl  # type: ignore
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:  # Windows
    import msvcrt  # type: ignore
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None


class LedgerCorruptError(RuntimeError):
    """Raised when the tail of the ledger cannot be parsed safely."""


_PROCESS_LOCK = threading.Lock()
_TIMING_KEYS = (
    "planning_ms",
    "repair_ms",
    "charts_ms",
    "grounding_retry_ms",
    "first_token_ms",
    "explaining_ms",
    "execution_ms",
    "total_ms",
)


def _digest(value: Any, length: int = 64) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _record_hash(record: dict[str, Any]) -> str:
    return _digest({k: v for k, v in record.items() if k != "hash"})


def _short(value: Any, limit: int = 300) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _session_ref(session_id: Optional[str]) -> Optional[str]:
    return _digest(session_id, 16) if session_id else None


@contextlib.contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    handle = open(lock_path, "a+b")  # noqa: SIM115 - closed in finally
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:  # pragma: no cover - exercised on Windows CI
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def _last_record(path: Path) -> Optional[dict[str, Any]]:
    """Read the last JSON record without loading the whole ledger."""
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            pos = handle.tell()
            if pos == 0:
                return None
            data = b""
            line = b""
            while pos > 0:
                step = min(8192, pos)
                pos -= step
                handle.seek(pos)
                data = handle.read(step) + data
                stripped = data.rstrip(b"\r\n")
                if b"\n" in stripped:
                    line = stripped.rsplit(b"\n", 1)[-1]
                    break
            else:
                line = data.strip()
    except FileNotFoundError:
        return None
    if not line:
        return None
    try:
        record = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerCorruptError(str(exc)) from exc
    if not isinstance(record, dict):
        raise LedgerCorruptError("last record is not a JSON object")
    return record


def append_event(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Append one hash-chained record, or return None when the ledger is off."""
    settings = get_settings()
    if not settings.audit_ledger:
        return None

    path = Path(settings.audit_ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_name(path.name + ".lock")

    with _PROCESS_LOCK:
        with _file_lock(lock_path):
            previous = _last_record(path) or {}
            record = dict(event)
            record.setdefault("schema", SCHEMA_VERSION)
            record.setdefault("event_id", uuid.uuid4().hex)
            record.setdefault("recorded_at", time.time())
            record["seq"] = int(previous.get("seq") or 0) + 1
            record["prev"] = previous.get("hash")
            record["hash"] = _record_hash(record)

            line = json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            )
            created = not path.exists()
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if created:
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
    return record


def read_events(path: Optional[str | Path] = None) -> Iterator[dict[str, Any]]:
    """Yield records in file order; raises on malformed JSON."""
    target = Path(path) if path is not None else Path(get_settings().audit_ledger_path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify_ledger(path: Optional[str | Path] = None) -> dict[str, Any]:
    """Recompute the hash chain and report the first broken record."""
    target = Path(path) if path is not None else Path(get_settings().audit_ledger_path)
    if not target.exists():
        return {
            "path": str(target),
            "ok": True,
            "records": 0,
            "broken_at": None,
            "reason": "ledger does not exist",
            "head": None,
        }

    expected_seq = 1
    expected_prev: Optional[str] = None
    count = 0
    with target.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": f"malformed JSON: {exc.msg}",
                    "head": expected_prev,
                }
            if not isinstance(record, dict):
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": "record is not a JSON object",
                    "head": expected_prev,
                }
            if record.get("schema") != SCHEMA_VERSION:
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": f"unsupported schema: {record.get('schema')!r}",
                    "head": expected_prev,
                }
            if record.get("seq") != expected_seq:
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": f"expected seq {expected_seq}, found {record.get('seq')!r}",
                    "head": expected_prev,
                }
            if record.get("prev") != expected_prev:
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": "prev hash does not match the previous record",
                    "head": expected_prev,
                }
            claimed = record.get("hash")
            if not claimed or claimed != _record_hash(record):
                return {
                    "path": str(target),
                    "ok": False,
                    "records": count,
                    "broken_at": lineno,
                    "reason": "record hash does not match its contents",
                    "head": expected_prev,
                }
            expected_prev = claimed
            expected_seq += 1
            count += 1

    return {
        "path": str(target),
        "ok": True,
        "records": count,
        "broken_at": None,
        "reason": None,
        "head": expected_prev,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {"value": str(value)}


def _dataset_payload(
    profile: Optional[DatasetProfile],
    generation: Optional[int],
    uploaded_name: Optional[str],
    meta: dict[str, Any],
) -> dict[str, Any]:
    fingerprint = getattr(profile, "fingerprint", None)
    return {
        "id": getattr(fingerprint, "dataset_id", None) or meta.get("dataset_id"),
        "content_hash": getattr(fingerprint, "content_hash", None),
        "schema_hash": getattr(fingerprint, "schema_hash", None),
        "rows": getattr(profile, "row_count", None),
        "columns": getattr(profile, "column_count", None),
        "generation": generation,
        "uploaded_name": uploaded_name,
    }


def _analysis_approval(meta: dict[str, Any], submitted: Any) -> dict[str, Any]:
    info = meta.get("approval")
    if isinstance(info, dict) and info:
        payload = dict(info)
        payload.setdefault(
            "stage", "executed" if payload.get("bound") else "unbound"
        )
        return payload
    if submitted is not None:
        payload = _as_dict(submitted)
        payload["bound"] = True
        payload["stage"] = "submitted_unverified"
        return payload
    return {
        "bound": False,
        "stage": "not_required",
        "reason": "no human approval; planner-generated plan",
    }


def _plan_payload(meta: dict[str, Any]) -> dict[str, Any]:
    steps = meta.get("steps") or []
    compact: list[dict[str, Any]] = []
    for step in steps[:50]:
        if not isinstance(step, dict):
            continue
        compact.append(
            {
                "step_id": step.get("step_id"),
                "operation": step.get("operation"),
                "success": step.get("success"),
                "error": _short(step.get("error"), 240),
            }
        )
    return {
        "hash": meta.get("plan_hash"),
        "objective": meta.get("objective"),
        "step_count": len(steps),
        "steps": compact,
    }


def _validation_payload(validation: Any) -> Optional[dict[str, Any]]:
    if validation is None:
        return None
    return {
        "valid": bool(validation.valid),
        "errors": [_short(message, 300) for message in validation.error_messages()[:8]],
        "warnings": [
            _short(message, 300) for message in validation.warning_messages()[:8]
        ],
        "checks_run": list(validation.checks_run[:20]),
    }


def _timings(meta: dict[str, Any]) -> dict[str, Any]:
    return {key: meta[key] for key in _TIMING_KEYS if meta.get(key) is not None}


def record_analysis(
    response: AnalysisResponse,
    *,
    question: str,
    profile: Optional[DatasetProfile] = None,
    route: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dataset_generation: Optional[int] = None,
    uploaded_name: Optional[str] = None,
    submitted_approval: Any = None,
) -> Optional[dict[str, Any]]:
    """Append the evidence record for a completed (or failed) analysis."""
    try:
        meta = response.meta or {}
        answer = response.answer or ""
        event = {
            "event": "analysis",
            "route": route,
            "request_id": request_id,
            "session_ref": _session_ref(session_id),
            "question": question,
            "question_hash": _digest(question, 16),
            "dataset": _dataset_payload(
                profile, dataset_generation, uploaded_name, meta
            ),
            "approval": _analysis_approval(meta, submitted_approval),
            "plan": _plan_payload(meta),
            "validation": _validation_payload(response.validation),
            "outcome": {
                "status": "ok" if response.error is None else "error",
                "error": response.error,
                "mode": "agentic" if meta.get("agentic") else "plan",
                "cache_hit": bool(meta.get("cache_hit")),
                "answer_chars": len(answer),
                "answer_hash": _digest(answer, 16),
                "unverified_figures": list(meta.get("unverified_figures") or []),
                "evidence_count": len(response.evidence or []),
                "table_count": len(response.tables or []),
                "plot_count": len(response.plots or []),
            },
            "model": meta.get("model"),
            "timings_ms": _timings(meta),
            "tokens": {
                "input": meta.get("tokens_in"),
                "output": meta.get("tokens_out"),
                "cached_prompt": meta.get("cached_prompt_tokens"),
                "llm_calls": meta.get("llm_calls"),
            },
            "trace_url": meta.get("trace_url"),
        }
        return append_event(event)
    except Exception:
        LOG.exception("could not append analysis to the audit ledger")
        return None


def record_plan_review(
    plan: dict[str, Any],
    approval: Any,
    *,
    question: str,
    profile: Optional[DatasetProfile] = None,
    route: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dataset_generation: Optional[int] = None,
    uploaded_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Append the record of a plan shown for human approval."""
    try:
        approval_payload = _as_dict(approval)
        steps = plan.get("steps") or []
        event = {
            "event": "plan_review",
            "route": route,
            "request_id": request_id,
            "session_ref": _session_ref(session_id),
            "question": question,
            "question_hash": _digest(question, 16),
            "dataset": _dataset_payload(
                profile, dataset_generation, uploaded_name, {}
            ),
            "approval": {
                **approval_payload,
                "bound": True,
                "stage": "issued",
            },
            "plan": {
                "hash": approval_payload.get("plan_hash"),
                "objective": plan.get("objective"),
                "step_count": len(steps),
                "steps": [
                    {
                        "step_id": step.get("step_id"),
                        "operation": step.get("operation"),
                    }
                    for step in steps[:50]
                    if isinstance(step, dict)
                ],
            },
            "validation": None,
            "outcome": {
                "status": "awaiting_approval",
                "error": None,
                "step_count": len(steps),
            },
        }
        return append_event(event)
    except Exception:
        LOG.exception("could not append plan review to the audit ledger")
        return None


def record_failure(
    error: Any,
    *,
    question: str,
    profile: Optional[DatasetProfile] = None,
    route: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    dataset_generation: Optional[int] = None,
    uploaded_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Append a record when the stream raised before producing a result."""
    try:
        event = {
            "event": "analysis",
            "route": route,
            "request_id": request_id,
            "session_ref": _session_ref(session_id),
            "question": question,
            "question_hash": _digest(question, 16),
            "dataset": _dataset_payload(
                profile, dataset_generation, uploaded_name, {}
            ),
            "approval": {
                "bound": False,
                "stage": "not_checked",
                "reason": "run failed before a result",
            },
            "plan": None,
            "validation": None,
            "outcome": {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": _short(error, 500),
            },
            "model": None,
            "timings_ms": {},
            "tokens": {},
            "trace_url": None,
        }
        return append_event(event)
    except Exception:
        LOG.exception("could not append failure to the audit ledger")
        return None


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m insight.audit [path] [--json]`` verifies the hash chain."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify the hash chain of the Insight audit ledger."
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="ledger path (default: INSIGHT_AUDIT_LEDGER_PATH)",
    )
    parser.add_argument("--json", action="store_true", help="print result as JSON")
    args = parser.parse_args(argv)

    result = verify_ledger(args.path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result["ok"]:
        print(f"OK  {result['records']} record(s)  {result['path']}")
    else:
        print(f"FAIL  record {result['broken_at']}  {result['reason']}")
        print(f"      {result['path']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
