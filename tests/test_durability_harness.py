"""Fast harness tests for the durability evidence tooling.

These tests do not re-check every persistence mechanism; the existing ledger,
cache, and alert-runner tests do that. They assert that the evidence harness
exercises every claim, records the known boundaries, restores process settings,
and renders a report that keeps the limitations visible.
"""

from __future__ import annotations

import pytest

from evaluation.bench.durability import (
    DurabilityConfig,
    config_for_mode,
    run_durability,
)
from evaluation.bench.durability_report import render_durability_markdown
from evaluation.bench.schema import write_jsonl
from insight.settings import get_settings


def _config() -> DurabilityConfig:
    return DurabilityConfig(
        ledger_roundtrips=1,
        ledger_records=5,
        tamper_records=8,
        concurrent_rounds=1,
        concurrent_writers=2,
        concurrent_records=3,
        cache_race_writes=10,
        cache_race_readers=2,
        cache_race_reads=10,
    )


@pytest.fixture(scope="module")
def durability_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("durability")
    settings = get_settings()
    before = {
        "artifacts_dir": settings.artifacts_dir,
        "audit_ledger": settings.audit_ledger,
        "audit_ledger_path": settings.audit_ledger_path,
        "query_cache_enabled": settings.query_cache_enabled,
    }
    records, summary = run_durability("test-run", root, _config())
    return records, summary, before


def test_durability_suite_supports_every_claim(durability_run):
    records, summary, _before = durability_run
    failures = [record for record in records if not record["passed"]]
    assert summary["status"] == "ok", (
        summary["harness_errors"],
        [claim for claim in summary["claims"] if claim["status"] != "supported"],
        failures,
    )
    assert summary["counts"] == {
        "records": 25,
        "scored": 23,
        "passed": 23,
        "failed": 0,
        "known_boundaries": 2,
        "harness_errors": 0,
    }
    assert [claim["id"] for claim in summary["claims"]] == [
        f"D-0{index}" for index in range(1, 9)
    ]
    assert all(claim["status"] == "supported" for claim in summary["claims"])
    assert {record["expectation"] for record in records} == {
        "scored",
        "known_boundary",
    }
    assert all(record["passed"] for record in records)
    boundary_ids = {
        record["trial_id"]
        for record in summary["known_boundaries"]
    }
    assert boundary_ids == {
        "D-02-delete_complete_tail",
        "D-07-corrupt_state_falls_back_to_empty",
    }


def test_durability_run_restores_process_settings(durability_run):
    _records, _summary, before = durability_run
    settings = get_settings()
    assert settings.artifacts_dir == before["artifacts_dir"]
    assert settings.audit_ledger == before["audit_ledger"]
    assert settings.audit_ledger_path == before["audit_ledger_path"]
    assert settings.query_cache_enabled == before["query_cache_enabled"]


def test_quick_mode_is_smaller_than_full():
    quick = config_for_mode("quick")
    full = config_for_mode("full")
    assert quick.concurrent_writers < full.concurrent_writers
    assert quick.concurrent_records < full.concurrent_records
    assert quick.cache_race_writes < full.cache_race_writes


def test_durability_report_renders_claims_and_boundaries(tmp_path, durability_run):
    records, summary, _before = durability_run
    raw_path = tmp_path / "durability_trials.jsonl"
    write_jsonl(raw_path, records)
    manifest = {
        "run_id": "test-run",
        "started_at": "2026-11-01T00:00:00Z",
        "finished_at": "2026-11-01T00:01:00Z",
        "status": summary["status"],
        "git": {"commit": "a" * 40, "branch": "test", "dirty": False,
                "untracked_file_count": 0},
        "environment": {"python": "3.11.9", "platform": "test", "cpu_count": 4},
        "artifacts": {"manifest": {"path": "summary/manifest.json"}},
    }
    report_path = tmp_path / "DURABILITY.md"
    render_durability_markdown(
        summary,
        manifest,
        records,
        raw_path=str(raw_path),
        output_path=report_path,
        root=tmp_path,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "# Insight Durability Evidence" in text
    for claim in summary["claims"]:
        assert claim["id"] in text
    assert "## Known boundaries" in text
    assert "delete_complete_tail" in text
    assert "detectable without an externally pinned head" in text
    assert "not a probability of correctness" in text
