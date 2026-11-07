"""Fast harness tests for the benchmark evidence tooling.

These tests do not assert that the verifier detects every mutation; that is the
benchmark's job. They assert that fixtures are deterministic, clean controls are
clean, and the artifact schema stays stable.
"""

from __future__ import annotations

import json

from evaluation.bench.adversarial_verifier import (
    VERIFIED_OPERATIONS,
    VerifierConfig,
    run_adversarial_verifier,
)
from evaluation.bench.fixtures import build_fixtures, quick_fixture_ids
from evaluation.bench.mutations import MutationSpec, apply_mutation
from evaluation.bench.schema import write_jsonl
from insight.execution.executor import execute_plan_sync
from insight.validation.verification import independently_verify


def _fixture(fixture_id: str):
    return next(f for f in build_fixtures() if f.fixture_id == fixture_id)


def test_fixtures_cover_all_verified_operations():
    fixtures = build_fixtures()
    assert {fixture.operation for fixture in fixtures} == set(VERIFIED_OPERATIONS)
    quick = {fixture.fixture_id for fixture in fixtures if fixture.fixture_id in quick_fixture_ids()}
    assert len(quick) == len(VERIFIED_OPERATIONS)


def test_quick_fixtures_are_clean_controls():
    for fixture in build_fixtures():
        if fixture.fixture_id not in quick_fixture_ids():
            continue
        execution = execute_plan_sync(fixture.frame, fixture.plan)
        assert execution.success, execution.failed_messages()
        issues = independently_verify(fixture.frame, fixture.plan, execution)
        assert issues == [], (fixture.fixture_id, [issue.message for issue in issues])


def test_numeric_tamper_is_scored_by_runner():
    fixture = _fixture("group_aggregate_simple")
    config = VerifierConfig(
        mode="test",
        fixture_ids=(fixture.fixture_id,),
        magnitudes=(1e-2,),
        count_deltas=(1,),
        max_fixtures_per_op=1,
        max_numeric_targets=1,
        max_calculation_targets=0,
        include_structural=False,
        include_calculation=False,
        include_correlation=False,
        clean_variants=("full",),
    )
    records, summary = run_adversarial_verifier("test-run", config)
    clean = [record for record in records if record["trial_kind"] == "clean_control"]
    tamper = [record for record in records if record["trial_kind"] == "tamper"]

    assert len(clean) == 1
    assert clean[0]["detected"] is False
    assert clean[0]["validator_valid"] is True

    numeric = [record for record in tamper if record["mutation_class"] == "numeric_delta"]
    assert numeric
    assert all(record["detected"] for record in numeric)
    assert summary["numeric_ge_1e_4"]["trials"] == 1
    assert summary["numeric_ge_1e_4"]["detected"] == 1
    assert summary["claims"][0]["status"] == "supported"


def test_calculation_tamper_is_detected():
    fixture = _fixture("regression_single")
    execution = execute_plan_sync(fixture.frame, fixture.plan)
    spec = MutationSpec(
        mutation_class="calculation_delta",
        target={"kind": "calculation", "step_id": 0, "label": "r_squared"},
        magnitude=0.01,
        params={
            "step_id": 0,
            "label": "r_squared",
            "bounds": (0.0, 1.0),
            "operation": "delta",
        },
    )
    mutated = execution.model_copy(deep=True)
    applied, _before, _after, _metadata = apply_mutation(mutated, spec)
    assert applied
    issues = independently_verify(fixture.frame, fixture.plan, mutated)
    assert any("r_squared" in issue.message for issue in issues)


def test_jsonl_writer_round_trips(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl(path, [{"trial_id": "one", "detected": True}, {"trial_id": "two"}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["trial_id"] == "one"
