"""Fast harness tests for the load, concurrency, and soak evidence tooling.

The harness starts a real local uvicorn process, so these tests use a tiny
workload. They assert that every claim is exercised, the HTTP and RSS metrics
are populated, process settings are restored, and the generated report keeps
the limitations visible. The full-size run lives in
``evaluation/bench/run_load_soak.py``.
"""

from __future__ import annotations

import os

import pytest

from evaluation.bench.load_soak import (
    LoadSoakConfig,
    _process_rss_bytes,
    _process_tree_rss_bytes,
    config_for_mode,
    run_load_soak,
)
from evaluation.bench.load_soak_report import render_load_soak_markdown
from evaluation.bench.schema import write_jsonl
from insight.settings import get_settings


def _config() -> LoadSoakConfig:
    return LoadSoakConfig(
        analysis_rounds=1,
        analysis_repeats=2,
        analysis_workers=2,
        analysis_fixture_limit=2,
        http_uploads=2,
        http_workers=2,
        http_rows=30,
        soak_seconds=1.0,
        soak_workers=1,
        soak_rows=20,
        soak_sample_interval_s=0.25,
        soak_max_growth_mb=512,
        soak_warmup_requests=1,
        cache_writes=8,
        cache_writers=2,
        cache_readers=1,
        cache_reads=4,
        cache_max_mb=1,
        cache_payload_kb=16,
    )


@pytest.fixture(scope="module")
def load_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("load-soak")
    settings = get_settings()
    before = {
        "artifacts_dir": settings.artifacts_dir,
        "query_cache_enabled": settings.query_cache_enabled,
        "query_cache_ttl_days": settings.query_cache_ttl_days,
        "query_cache_max_mb": settings.query_cache_max_mb,
    }
    records, summary = run_load_soak("test-run", root, _config())
    return records, summary, before


def test_load_soak_supports_every_claim(load_run):
    records, summary, _before = load_run
    failures = [record for record in records if not record["passed"]]
    assert summary["status"] == "ok", (
        summary["harness_errors"],
        [claim for claim in summary["claims"] if claim["status"] != "supported"],
        failures,
    )
    assert summary["counts"] == {
        "records": 8,
        "scored": 8,
        "passed": 8,
        "failed": 0,
        "harness_errors": 0,
    }
    assert [claim["id"] for claim in summary["claims"]] == [
        "L-01",
        "L-02",
        "L-03",
        "L-04",
    ]
    assert all(claim["status"] == "supported" for claim in summary["claims"])
    assert {record["expectation"] for record in records} == {"scored"}
    assert all(record["passed"] for record in records)


def test_load_soak_metrics_are_populated(load_run):
    _records, summary, _before = load_run
    metrics = summary["metrics"]
    assert metrics["analysis"]["fixtures"] == 2
    assert metrics["analysis"]["passed"] == 2
    assert metrics["http"]["requests"] == 2
    assert metrics["http"]["passed"] == 2
    assert metrics["http"]["status_counts"] == {"303": 2}
    assert metrics["soak"]["requests"] > 0
    assert metrics["soak"]["failures"] == 0
    assert metrics["soak"]["rss_bounded"] is True
    assert metrics["soak"]["rss_baseline_mb"] > 0
    assert metrics["soak"]["sample_count"] >= 2
    assert metrics["cache"]["hits"] > 0
    assert metrics["cache"]["corrupt_reads"] == 0
    assert metrics["cache"]["size_after_sweep_bytes"] <= metrics["cache"]["limit_bytes"]


def test_load_soak_restores_process_settings(load_run):
    _records, _summary, before = load_run
    settings = get_settings()
    assert settings.artifacts_dir == before["artifacts_dir"]
    assert settings.query_cache_enabled == before["query_cache_enabled"]
    assert settings.query_cache_ttl_days == before["query_cache_ttl_days"]
    assert settings.query_cache_max_mb == before["query_cache_max_mb"]


def test_process_rss_helpers_return_positive_values():
    single = _process_rss_bytes(os.getpid())
    tree = _process_tree_rss_bytes(os.getpid())
    assert single is not None and single > 0
    assert tree is not None and tree >= single


def test_quick_mode_is_smaller_than_full():
    quick = config_for_mode("quick")
    full = config_for_mode("full")
    assert quick.http_uploads < full.http_uploads
    assert quick.soak_seconds < full.soak_seconds
    assert quick.cache_writes < full.cache_writes


def test_load_soak_report_renders_claims_and_limits(tmp_path, load_run):
    records, summary, _before = load_run
    raw_path = tmp_path / "load_trials.jsonl"
    write_jsonl(raw_path, records)
    manifest = {
        "run_id": "test-run",
        "started_at": "2026-11-01T00:00:00Z",
        "finished_at": "2026-11-01T00:01:00Z",
        "status": summary["status"],
        "git": {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
            "untracked_file_count": 0,
        },
        "environment": {
            "python": "3.11.9",
            "platform": "test",
            "cpu_count": 4,
        },
        "artifacts": {"manifest": {"path": "summary/manifest.json"}},
    }
    report_path = tmp_path / "LOAD_SOAK.md"
    render_load_soak_markdown(
        summary,
        manifest,
        records,
        raw_path=str(raw_path),
        output_path=report_path,
        root=tmp_path,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "# Insight Load and Soak Evidence" in text
    for claim in summary["claims"]:
        assert claim["id"] in text
    assert "## Workload metrics" in text
    assert "## Limitations" in text
    assert "single-worker" in text
    assert "not proof that a leak is impossible" in text
    assert "not an SLO" in text
