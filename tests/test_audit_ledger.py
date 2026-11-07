"""The evidence ledger: append-only JSONL, hash-chained, and wired into queries."""
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from insight import audit
from insight.domain.visualization import (
    AnalysisResponse,
    Evidence,
    ValidationIssue,
    ValidationResult,
)
from insight.settings import get_settings

from conftest import EXPLANATION, VALID_PLAN, ScriptedLLM, StubChain


class _Fingerprint:
    dataset_id = "ds_test"
    content_hash = "a" * 64
    schema_hash = "b" * 64


class _Profile:
    fingerprint = _Fingerprint()
    row_count = 120
    column_count = 5


def _install(monkeypatch, tmp_path, enabled=True):
    settings = SimpleNamespace(
        audit_ledger=enabled,
        audit_ledger_path=tmp_path / "audit" / "analysis.jsonl",
    )
    monkeypatch.setattr(audit, "get_settings", lambda: settings)
    return settings


def _response(**meta):
    return AnalysisResponse(
        question="avg revenue by region",
        answer="West leads.",
        evidence=[Evidence(kind="calculation", label="mean", value=42)],
        validation=ValidationResult(
            valid=True,
            warnings=[ValidationIssue(severity="warning", message="small sample")],
            checks_run=["schema", "arithmetic"],
        ),
        meta={
            "model": "fake/model",
            "objective": "average revenue by region",
            "plan_hash": "0123456789abcdef",
            "dataset_id": "ds_test",
            "steps": [
                {
                    "step_id": 0,
                    "operation": "group_aggregate",
                    "success": True,
                    "error": None,
                    "params": {"column": "secret"},
                }
            ],
            "planning_ms": 12,
            "execution_ms": 4,
            "total_ms": 50,
            "tokens_in": 100,
            "tokens_out": 20,
            "cached_prompt_tokens": 60,
            "llm_calls": 2,
            "trace_url": "https://trace.example/1",
            **meta,
        },
    )


def test_disabled_ledger_is_a_noop(monkeypatch, tmp_path):
    settings = _install(monkeypatch, tmp_path, enabled=False)

    assert audit.record_analysis(_response(), question="q") is None
    assert not settings.audit_ledger_path.exists()


def test_analysis_record_captures_identity_approval_and_validation(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)
    response = _response(
        approval={
            "bound": True,
            "dataset_id": "ds_test",
            "generation": 3,
            "approved_plan_hash": "0123456789abcdef",
            "submitted_plan_hash": "0123456789abcdef",
            "edited": False,
            "issued_at": 1.0,
            "expires_at": 2.0,
        }
    )

    record = audit.record_analysis(
        response,
        question="avg revenue by region",
        profile=_Profile(),
        route="/query",
        request_id="req-1",
        session_id="session-1",
        dataset_generation=3,
        uploaded_name="sales.csv",
    )

    assert record is not None
    assert record["event"] == "analysis"
    assert record["route"] == "/query"
    assert record["dataset"]["id"] == "ds_test"
    assert record["dataset"]["content_hash"] == "a" * 64
    assert record["dataset"]["schema_hash"] == "b" * 64
    assert record["dataset"]["rows"] == 120
    assert record["dataset"]["generation"] == 3
    assert record["approval"]["stage"] == "executed"
    assert record["approval"]["edited"] is False
    assert record["plan"]["hash"] == "0123456789abcdef"
    assert record["plan"]["steps"] == [
        {
            "step_id": 0,
            "operation": "group_aggregate",
            "success": True,
            "error": None,
        }
    ]
    # Plan params may contain literals and are deliberately not copied.
    assert "secret" not in json.dumps(record)
    assert record["validation"]["valid"] is True
    assert record["validation"]["warnings"] == ["small sample"]
    assert record["outcome"]["status"] == "ok"
    assert record["timings_ms"]["planning_ms"] == 12
    assert record["tokens"]["input"] == 100
    assert record["trace_url"].endswith("/1")
    assert record["session_ref"] != "session-1"

    stored = list(audit.read_events())
    assert len(stored) == 1
    assert stored[0]["hash"] == record["hash"]


def test_plan_review_records_the_issued_binding(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    record = audit.record_plan_review(
        {
            "objective": "average revenue by region",
            "steps": [
                {"step_id": 0, "operation": "group_aggregate", "params": {"x": 1}}
            ],
        },
        {
            "dataset_id": "ds_test",
            "generation": 2,
            "plan_hash": "abcdef0123456789",
            "issued_at": 10.0,
            "expires_at": 20.0,
        },
        question="avg revenue",
        profile=_Profile(),
        route="/api/query",
        dataset_generation=2,
    )

    assert record["event"] == "plan_review"
    assert record["outcome"]["status"] == "awaiting_approval"
    assert record["approval"]["stage"] == "issued"
    assert record["approval"]["plan_hash"] == "abcdef0123456789"
    assert record["plan"]["steps"] == [
        {"step_id": 0, "operation": "group_aggregate"}
    ]
    assert record["dataset"]["generation"] == 2


def test_record_failure_marks_the_run_failed(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    record = audit.record_failure(
        RuntimeError("provider down"),
        question="avg revenue",
        profile=_Profile(),
        route="/query",
    )

    assert record["event"] == "analysis"
    assert record["outcome"]["status"] == "failed"
    assert record["outcome"]["error_type"] == "RuntimeError"
    assert "provider down" in record["outcome"]["error"]


def test_ledger_is_hash_chained_and_verifiable(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    first = audit.append_event({"event": "analysis", "outcome": {"status": "ok"}})
    second = audit.append_event({"event": "plan_review"})
    third = audit.append_event({"event": "analysis"})

    assert first["seq"] == 1 and first["prev"] is None
    assert second["prev"] == first["hash"]
    assert third["seq"] == 3

    result = audit.verify_ledger()
    assert result["ok"] is True
    assert result["records"] == 3
    assert result["head"] == third["hash"]


def test_tampering_is_detected(monkeypatch, tmp_path):
    settings = _install(monkeypatch, tmp_path)
    audit.append_event({"event": "analysis", "outcome": {"status": "ok"}})
    audit.append_event({"event": "analysis", "outcome": {"status": "ok"}})

    lines = settings.audit_ledger_path.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["outcome"]["status"] = "error"
    lines[0] = json.dumps(record, sort_keys=True, separators=(",", ":"))
    settings.audit_ledger_path.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    result = audit.verify_ledger()
    assert result["ok"] is False
    assert result["broken_at"] == 1
    assert "hash" in result["reason"]


def test_ledger_write_failure_does_not_break_analysis(monkeypatch, tmp_path):
    _install(monkeypatch, tmp_path)

    def _boom(event):
        raise OSError("disk full")

    monkeypatch.setattr(audit, "append_event", _boom)
    assert audit.record_analysis(_response(), question="q") is None


def test_torn_tail_is_detected_and_blocks_append(monkeypatch, tmp_path):
    settings = _install(monkeypatch, tmp_path)
    audit.append_event({"event": "analysis"})
    audit.append_event({"event": "analysis"})

    with settings.audit_ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"event": "torn", "seq": 3')

    result = audit.verify_ledger()
    assert result["ok"] is False
    assert result["broken_at"] == 3
    assert "malformed JSON" in result["reason"]

    with pytest.raises(audit.LedgerCorruptError):
        audit.append_event({"event": "analysis"})
    assert len(settings.audit_ledger_path.read_text(encoding="utf-8").splitlines()) == 3


def test_truncating_torn_tail_restores_append(monkeypatch, tmp_path):
    settings = _install(monkeypatch, tmp_path)
    audit.append_event({"event": "analysis"})
    audit.append_event({"event": "analysis"})

    with settings.audit_ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write('{"event": "torn", "seq": 3')

    complete = settings.audit_ledger_path.read_text(encoding="utf-8")
    settings.audit_ledger_path.write_text(
        complete[: complete.rfind("\n") + 1], encoding="utf-8", newline="\n"
    )

    recovered = audit.append_event({"event": "analysis"})
    assert recovered["seq"] == 3
    result = audit.verify_ledger()
    assert result["ok"] is True
    assert result["records"] == 3
    assert result["head"] == recovered["hash"]


def test_complete_tail_deletion_is_not_detectable(monkeypatch, tmp_path):
    settings = _install(monkeypatch, tmp_path)
    first = audit.append_event({"event": "analysis"})
    audit.append_event({"event": "analysis"})

    lines = settings.audit_ledger_path.read_text(encoding="utf-8").splitlines()
    settings.audit_ledger_path.write_text(
        "\n".join(lines[:-1]) + "\n", encoding="utf-8", newline="\n"
    )

    # A truncated tail is indistinguishable from a shorter legitimate ledger
    # without an externally persisted head. This is a documented boundary.
    result = audit.verify_ledger()
    assert result["ok"] is True
    assert result["records"] == 1
    assert result["head"] == first["hash"]


CSV_BYTES = b"region,revenue\n" + b"".join(
    f"r{i % 4},{100 + i}\n".encode() for i in range(80)
)


def _enable_webapp_ledger(monkeypatch, ledger):
    monkeypatch.setenv("INSIGHT_AUDIT_LEDGER", "on")
    monkeypatch.setenv("INSIGHT_AUDIT_LEDGER_PATH", str(ledger))
    get_settings.cache_clear()


def _upload(client):
    uploaded = client.post(
        "/upload",
        files={"file": ("sales.csv", CSV_BYTES, "text/csv")},
        follow_redirects=False,
    )
    assert uploaded.status_code == 303


def _records(ledger):
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_api_query_appends_a_ledger_record(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _enable_webapp_ledger(monkeypatch, ledger)

    import insight.orchestrator as orch
    from webapp.app import STORE, app

    llm = ScriptedLLM([VALID_PLAN, EXPLANATION])
    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: StubChain(llm))

    sid = "audit-ledger-session"
    STORE.drop(sid)
    try:
        with TestClient(app) as client:
            _upload(client)
            response = client.post(
                "/api/query",
                json={"question": "Which region has the highest revenue?"},
            )
            assert response.status_code == 200
            assert response.json()["error"] is None

        records = _records(ledger)
        assert len(records) == 1
        record = records[0]
        assert record["event"] == "analysis"
        assert record["route"] == "/api/query"
        assert record["outcome"]["status"] == "ok"
        assert record["dataset"]["id"]
        assert record["dataset"]["generation"] == 1
        assert record["validation"]["valid"] is True
        assert record["plan"]["step_count"] >= 1
        assert audit.verify_ledger()["ok"] is True
    finally:
        STORE.drop(sid)
        get_settings.cache_clear()


def test_api_plan_only_appends_a_plan_review_record(monkeypatch, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    _enable_webapp_ledger(monkeypatch, ledger)

    import insight.orchestrator as orch
    from webapp.app import STORE, app

    llm = ScriptedLLM([VALID_PLAN])
    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: StubChain(llm))

    sid = "audit-plan-review-session"
    STORE.drop(sid)
    try:
        with TestClient(app) as client:
            _upload(client)
            response = client.post(
                "/api/query",
                json={
                    "question": "Which region has the highest revenue?",
                    "plan_only": True,
                },
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["approval"]["plan_hash"]

        records = _records(ledger)
        assert len(records) == 1
        record = records[0]
        assert record["event"] == "plan_review"
        assert record["route"] == "/api/query"
        assert record["outcome"]["status"] == "awaiting_approval"
        assert record["approval"]["stage"] == "issued"
        assert record["approval"]["plan_hash"] == payload["approval"]["plan_hash"]
        assert record["plan"]["step_count"] >= 1
        assert audit.verify_ledger()["ok"] is True
    finally:
        STORE.drop(sid)
        get_settings.cache_clear()
