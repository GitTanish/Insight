"""Offline tests for the live A/B and ablation harness.

These tests never call a provider. They use a fake ``run_analysis_sync``-shaped
runner to exercise arm orchestration, setting restoration, resume, aggregation,
and report rendering.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.bench.live_ab import (
    DEFAULT_ARMS,
    Arm,
    LiveConfig,
    compare_arms,
    run_live_ab,
    select_cases,
    summarize_arm,
)
from evaluation.bench.live_ab_report import render_live_ab_markdown
from insight.domain.visualization import AnalysisResponse, Evidence, ValidationResult
from insight.settings import get_settings


CASE = {
    "id": "case_a",
    "dataset": "data.csv",
    "question": "Which region leads?",
    "expect_ops": ["group_aggregate"],
    "answer_contains": ["West"],
    "numeric_expect": [],
}


def _frames():
    return {"data.csv": (pd.DataFrame({"region": ["West"], "value": [1]}), None)}


def _response(request):
    return AnalysisResponse(
        question=request.question,
        answer="West leads.",
        evidence=[Evidence(kind="calculation", label="total", value=1)],
        validation=ValidationResult(valid=True),
        meta={
            "model": request.model_id,
            "validation_valid": True,
            "llm_calls": 2 if request.agentic else 1,
            "prompt_tokens": 10,
            "completion_tokens": 4,
            "grounding_retry_ms": 0,
            "unverified_figures": [],
            "agentic": bool(request.agentic),
            "agentic_throttled": False,
            "steps": [{"operation": "group_aggregate", "success": True}],
        },
    )


def test_live_ab_runs_all_arms_and_restores_settings(tmp_path):
    settings = get_settings()
    before = {
        "answer_grounded_check": settings.answer_grounded_check,
        "agentic_call_delay_s": settings.agentic_call_delay_s,
    }
    requests = []
    grounding_seen = []

    def runner(df, profile, request, artifacts_dir):
        requests.append(request)
        grounding_seen.append(settings.answer_grounded_check)
        return _response(request)

    records, summary = run_live_ab(
        run_id="test-run",
        config=LiveConfig(
            model_id="test/model",
            repetitions=1,
            delay_s=0,
            no_fallback=True,
        ),
        arms=DEFAULT_ARMS,
        out_dir=tmp_path,
        cases=[CASE],
        frames=_frames(),
        runner=runner,
    )

    assert len(records) == 4
    assert summary["counts"] == {
        "expected": 4,
        "records": 4,
        "passed": 4,
        "failed": 0,
        "throttled": 0,
        "errors": 0,
        "missing": 0,
    }
    assert summary["status"] == "ok"
    assert [request.agentic for request in requests] == [False, True, True, True]
    assert all(request.no_fallback for request in requests)
    assert grounding_seen == [True, True, True, False]
    assert settings.answer_grounded_check == before["answer_grounded_check"]
    assert settings.agentic_call_delay_s == before["agentic_call_delay_s"]
    assert all(arm["complete"] for arm in summary["arms"])
    assert {claim["id"] for claim in summary["claims"]} == {"A-01", "A-02", "A-03", "A-04"}


def test_live_ab_resume_skips_completed_trials(tmp_path):
    calls = []

    def runner(df, profile, request, artifacts_dir):
        calls.append(request.question)
        return _response(request)

    kwargs = dict(
        run_id="resume-run",
        config=LiveConfig(
            model_id="test/model",
            repetitions=1,
            delay_s=0,
            no_fallback=True,
        ),
        arms=DEFAULT_ARMS,
        out_dir=tmp_path,
        cases=[CASE],
        frames=_frames(),
        runner=runner,
    )
    first_records, first_summary = run_live_ab(**kwargs)
    assert len(calls) == 4
    assert len(first_records) == 4

    second_records, second_summary = run_live_ab(**{**kwargs, "resume": True})
    assert len(calls) == 4
    assert len(second_records) == 4
    assert second_summary["counts"] == first_summary["counts"]
    assert second_summary["run_id"] == first_summary["run_id"]


def test_grounding_ablation_marks_posthoc_unverified(tmp_path):
    def runner(df, profile, request, artifacts_dir):
        response = _response(request)
        response.answer = "West leads with 999."
        return response

    records, _summary = run_live_ab(
        run_id="grounding-run",
        config=LiveConfig(
            model_id="test/model",
            repetitions=1,
            delay_s=0,
            no_fallback=True,
        ),
        arms=(DEFAULT_ARMS[1], DEFAULT_ARMS[3]),
        out_dir=tmp_path,
        cases=[CASE],
        frames=_frames(),
        runner=runner,
    )
    by_arm = {record["arm_id"]: record for record in records}
    assert by_arm["agentic"]["grounding_enabled"] is True
    assert by_arm["agentic_no_grounding"]["grounding_enabled"] is False
    assert "999" in by_arm["agentic_no_grounding"]["posthoc_unverified"]


def test_verifier_ablation_is_patched_only_for_that_arm(tmp_path):
    from insight.validation import verification

    verifier_results = []

    def runner(df, profile, request, artifacts_dir):
        try:
            result = verification.independently_verify(None, None, None)
            verifier_results.append(("patched", result))
        except Exception:
            verifier_results.append(("real", None))
        return _response(request)

    run_live_ab(
        run_id="verifier-run",
        config=LiveConfig(
            model_id="test/model",
            repetitions=1,
            delay_s=0,
            no_fallback=True,
        ),
        arms=(DEFAULT_ARMS[1], DEFAULT_ARMS[2]),
        out_dir=tmp_path,
        cases=[CASE],
        frames=_frames(),
        runner=runner,
    )
    assert verifier_results[0][0] == "real"
    assert verifier_results[1][0] == "patched"


def test_select_cases_rejects_unknown_ids():
    with pytest.raises(ValueError, match="unknown case"):
        select_cases([CASE], ["missing"], 0)


def test_summarize_and_compare_paired_cases():
    arm = Arm("baseline", "Baseline", agentic=False)
    records = [
        {
            "arm_id": "baseline",
            "case_id": "a",
            "repetition": 1,
            "passed": True,
            "validation_valid": True,
            "unverified_figures": [],
            "posthoc_unverified": [],
            "grounding_retry_ms": 0,
            "llm_calls": 2,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "wall_s": 1.0,
            "throttled": False,
            "error": None,
            "reasons": [],
        },
        {
            "arm_id": "baseline",
            "case_id": "b",
            "repetition": 1,
            "passed": False,
            "validation_valid": True,
            "unverified_figures": [],
            "posthoc_unverified": [],
            "grounding_retry_ms": 0,
            "llm_calls": 2,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "wall_s": 2.0,
            "throttled": False,
            "error": None,
            "reasons": ["miss"],
        },
    ]
    summary = summarize_arm(arm, records, expected=2)
    assert summary["trials"] == 2
    assert summary["passed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["wilson_95"][0] < 0.5 < summary["wilson_95"][1]

    other = [
        {**records[0], "arm_id": "agentic", "passed": False},
        {**records[1], "arm_id": "agentic", "passed": True},
    ]
    comparison = compare_arms(records + other, "baseline", "agentic")
    assert comparison["paired_trials"] == 2
    assert comparison["both_pass"] == 0
    assert comparison["both_fail"] == 0
    assert comparison["left_only"] == 1
    assert comparison["right_only"] == 1
    assert comparison["delta_pass_rate"] == 0.0


def test_report_renders_arms_comparisons_and_limits(tmp_path):
    records, summary = run_live_ab(
        run_id="report-run",
        config=LiveConfig(
            model_id="test/model",
            repetitions=1,
            delay_s=0,
            no_fallback=True,
        ),
        arms=DEFAULT_ARMS,
        out_dir=tmp_path / "raw",
        cases=[CASE],
        frames=_frames(),
        runner=lambda df, profile, request, artifacts_dir: _response(request),
    )
    manifest = {
        "run_id": "report-run",
        "started_at": "2026-11-03T00:00:00Z",
        "finished_at": "2026-11-03T00:01:00Z",
        "git": {
            "commit": "a" * 40,
            "branch": "test",
            "dirty": False,
        },
        "environment": {"python": "3.11.9", "platform": "test"},
        "artifacts": {"manifest": {"path": "raw/manifest.json"}},
    }
    report_path = tmp_path / "LIVE_AB.md"
    render_live_ab_markdown(
        summary,
        manifest,
        records,
        raw_path="raw/live_trials.jsonl",
        output_path=report_path,
        root=tmp_path,
    )
    text = report_path.read_text(encoding="utf-8")
    assert "# Insight Live A/B and Ablation Evidence" in text
    assert "## Arm summary" in text
    assert "## Paired comparisons" in text
    assert "## Per-case matrix" in text
    assert "## Limitations" in text
    assert "not a universal accuracy claim" in text
    assert "`agentic_no_verifier`" in text
    assert "`agentic_no_grounding`" in text
