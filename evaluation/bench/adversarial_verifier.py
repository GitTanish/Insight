"""Adversarial matrix for the independent result verifier.

The benchmark executes seven table-producing operation families on deterministic
fixtures, applies controlled mutations to copies of the clean execution, and
records whether ``independently_verify`` raises an error-severity issue. It also
records the final ``validate_execution`` verdict so propagation can be audited.

No production code is modified by this module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from insight.domain.analysis import AnalysisPlan
from insight.domain.execution import ExecutionResult
from insight.execution.executor import execute_plan_sync
from insight.validation.result_validator import validate_execution
from insight.validation.verification import independently_verify

from evaluation.bench.fixtures import DEFAULT_ROWS, DEFAULT_SEED, Fixture, build_fixtures, quick_fixture_ids
from evaluation.bench.mutations import MutationSpec, apply_mutation, enumerate_mutations
from evaluation.bench.schema import (
    SCHEMA_VERSION,
    clean_false_positive_summary,
    group_rate,
    jsonable,
    rate_summary,
)

VERIFIED_OPERATIONS = (
    "group_aggregate",
    "top_n",
    "value_counts",
    "correlation",
    "compare_subsets",
    "anova_test",
    "regression",
)

MAGNITUDES_FULL = (1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
MAGNITUDES_QUICK = (1e-9, 1e-6, 1e-4, 1e-2, 1.0)


@dataclass(frozen=True)
class VerifierConfig:
    mode: str
    seed: int = DEFAULT_SEED
    rows: int = DEFAULT_ROWS
    magnitudes: tuple[float, ...] = MAGNITUDES_QUICK
    count_deltas: tuple[int, ...] = (1, 5)
    max_fixtures_per_op: int | None = 1
    max_numeric_targets: int | None = 1
    max_calculation_targets: int | None = 2
    include_structural: bool = True
    include_calculation: bool = True
    include_correlation: bool = True
    clean_variants: tuple[str, ...] = ("full", "sample_80")
    fixture_ids: tuple[str, ...] | None = None


def config_for_mode(mode: str) -> VerifierConfig:
    normalized = mode.lower()
    if normalized == "quick":
        return VerifierConfig(
            mode="quick",
            magnitudes=MAGNITUDES_QUICK,
            count_deltas=(1, 5),
            max_fixtures_per_op=1,
            max_numeric_targets=1,
            max_calculation_targets=2,
            clean_variants=("full", "sample_80"),
            fixture_ids=tuple(sorted(quick_fixture_ids())),
        )
    if normalized == "full":
        return VerifierConfig(
            mode="full",
            magnitudes=MAGNITUDES_FULL,
            count_deltas=(1, 2, 5),
            max_fixtures_per_op=None,
            max_numeric_targets=None,
            max_calculation_targets=None,
            clean_variants=("full", "head_75", "sample_80", "flag_1"),
            fixture_ids=None,
        )
    raise ValueError("mode must be 'quick' or 'full'")


def _selected_fixtures(config: VerifierConfig) -> list[Fixture]:
    fixtures = build_fixtures(seed=config.seed, n=config.rows)
    if config.fixture_ids is not None:
        wanted = set(config.fixture_ids)
        fixtures = [fixture for fixture in fixtures if fixture.fixture_id in wanted]
    if config.max_fixtures_per_op is not None:
        selected: list[Fixture] = []
        counts: dict[str, int] = {}
        for fixture in fixtures:
            used = counts.get(fixture.operation, 0)
            if used >= config.max_fixtures_per_op:
                continue
            selected.append(fixture)
            counts[fixture.operation] = used + 1
        fixtures = selected
    return fixtures


def _clean_variant_frames(fixture: Fixture, config: VerifierConfig) -> list[tuple[str, Any]]:
    frame = fixture.frame
    variants: list[tuple[str, Any]] = []
    for name in config.clean_variants:
        if name == "full":
            variants.append((name, frame))
        elif name == "head_75":
            variants.append((name, frame.head(max(40, int(len(frame) * 0.75)))))
        elif name == "sample_80":
            variants.append(
                (
                    name,
                    frame.sample(frac=0.8, random_state=config.seed % (2**32)).reset_index(
                        drop=True
                    ),
                )
            )
        elif name == "flag_1":
            filtered = frame[frame["flag"] == 1].reset_index(drop=True)
            if len(filtered) >= 40:
                variants.append((name, filtered))
    return variants


def _evaluate(
    frame,
    plan: AnalysisPlan,
    execution: ExecutionResult,
) -> tuple[list, Any, float, str | None]:
    started = time.perf_counter()
    error: str | None = None
    try:
        issues = independently_verify(frame, plan, execution)
    except Exception as exc:  # pragma: no cover - harness safety net
        issues = []
        error = f"independently_verify raised {type(exc).__name__}: {exc}"
    try:
        validation = validate_execution(plan, execution, df=frame)
    except Exception as exc:  # pragma: no cover - harness safety net
        validation = None
        error = (
            f"{error}; " if error else ""
        ) + f"validate_execution raised {type(exc).__name__}: {exc}"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return issues, validation, elapsed_ms, error


def _record_base(
    *,
    run_id: str,
    trial_id: str,
    fixture: Fixture,
    trial_kind: str,
    mutation_class: str,
    expectation: str,
    spec: MutationSpec | None = None,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "adversarial_verifier",
        "run_id": run_id,
        "trial_id": trial_id,
        "fixture_id": fixture.fixture_id,
        "operation": fixture.operation,
        "tags": list(fixture.tags),
        "trial_kind": trial_kind,
        "mutation_class": mutation_class,
        "expectation": expectation,
        "magnitude": spec.magnitude if spec else None,
    }


def _finish_record(
    record: dict,
    issues: list,
    validation,
    elapsed_ms: float,
    harness_error: str | None,
) -> dict:
    error_issues = [issue for issue in issues if issue.severity == "error"]
    warning_issues = [issue for issue in issues if issue.severity == "warning"]
    validator_errors = list(validation.errors) if validation is not None else []
    validator_warnings = list(validation.warnings) if validation is not None else []
    record.update(
        {
            "detected": bool(error_issues),
            "verifier_error_count": len(error_issues),
            "verifier_warning_count": len(warning_issues),
            "verifier_issue_total": len(issues),
            "verifier_issues": [
                {
                    "severity": issue.severity,
                    "message": issue.message,
                    "step_id": issue.step_id,
                }
                for issue in issues[:20]
            ],
            "validator_valid": bool(validation.valid) if validation is not None else None,
            "validator_error_count": len(validator_errors),
            "validator_warning_count": len(validator_warnings),
            "validator_errors": [
                {"message": issue.message, "step_id": issue.step_id}
                for issue in validator_errors[:20]
            ],
            "duration_ms": round(elapsed_ms, 3),
            "harness_error": harness_error,
        }
    )
    return record


def run_adversarial_verifier(run_id: str, config: VerifierConfig) -> tuple[list[dict], dict]:
    fixtures = _selected_fixtures(config)
    records: list[dict] = []
    skipped: list[dict] = []
    skipped_clean: list[dict] = []
    trial_counter = 0

    for fixture in fixtures:
        clean_execution = execute_plan_sync(fixture.frame, fixture.plan)
        if not clean_execution.success:
            raise RuntimeError(
                f"fixture {fixture.fixture_id} did not execute cleanly: "
                f"{clean_execution.failed_messages()}"
            )

        for variant_name, variant_frame in _clean_variant_frames(fixture, config):
            variant_execution = execute_plan_sync(variant_frame, fixture.plan)
            if not variant_execution.success:
                if variant_name == "full":
                    raise RuntimeError(
                        f"clean variant {fixture.fixture_id}/{variant_name} failed: "
                        f"{variant_execution.failed_messages()}"
                    )
                skipped_clean.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "variant": variant_name,
                        "error": "; ".join(variant_execution.failed_messages()),
                    }
                )
                continue
            issues, validation, elapsed_ms, harness_error = _evaluate(
                variant_frame, fixture.plan, variant_execution
            )
            trial_counter += 1
            record = _record_base(
                run_id=run_id,
                trial_id=f"{run_id}:{trial_counter:05d}",
                fixture=fixture,
                trial_kind="clean_control",
                mutation_class="clean",
                expectation="valid",
            )
            record["clean_variant"] = variant_name
            _finish_record(record, issues, validation, elapsed_ms, harness_error)
            records.append(record)

        specs = enumerate_mutations(
            fixture,
            clean_execution,
            magnitudes=config.magnitudes,
            count_deltas=config.count_deltas,
            include_structural=config.include_structural,
            include_calculation=config.include_calculation,
            include_correlation=config.include_correlation,
            max_numeric_targets=config.max_numeric_targets,
            max_calculation_targets=config.max_calculation_targets,
        )
        for spec in specs:
            mutated = clean_execution.model_copy(deep=True)
            applied, before, after, metadata = apply_mutation(mutated, spec)
            if not applied:
                skipped.append(
                    {
                        "fixture_id": fixture.fixture_id,
                        "mutation_class": spec.mutation_class,
                        "target": spec.target,
                        "params": spec.params,
                        "magnitude": spec.magnitude,
                    }
                )
                continue
            issues, validation, elapsed_ms, harness_error = _evaluate(
                fixture.frame, fixture.plan, mutated
            )
            trial_counter += 1
            record = _record_base(
                run_id=run_id,
                trial_id=f"{run_id}:{trial_counter:05d}",
                fixture=fixture,
                trial_kind="tamper",
                mutation_class=spec.mutation_class,
                expectation=spec.expectation,
                spec=spec,
            )
            record.update(
                {
                    "target": metadata.get("target", spec.target),
                    "mutation_params": {
                        key: value
                        for key, value in spec.params.items()
                        if key not in {"bounds"}
                    },
                    "before": jsonable(before),
                    "after": jsonable(after),
                }
            )
            _finish_record(record, issues, validation, elapsed_ms, harness_error)
            records.append(record)

    summary = summarize_adversarial(
        records, config=config, skipped=skipped, skipped_clean=skipped_clean
    )
    summary["artifact_reference"] = (
        f".artifacts/bench/{run_id}/verifier_matrix.jsonl"
    )
    return records, summary


def _claim(
    claim_id: str,
    statement: str,
    records: list[dict],
    *,
    scope: str,
    notes: str = "",
) -> dict:
    summary = rate_summary(records)
    if summary["trials"] == 0:
        status = "no-data"
    elif summary["missed"] == 0:
        status = "supported"
    elif summary["missed"] > summary["detected"]:
        status = "gap"
    else:
        status = "incomplete"
    return {
        "id": claim_id,
        "statement": statement,
        "scope": scope,
        "status": status,
        "notes": notes,
        **summary,
    }


def summarize_adversarial(
    records: list[dict],
    *,
    config: VerifierConfig,
    skipped: list[dict] | None = None,
    skipped_clean: list[dict] | None = None,
) -> dict:
    skipped = skipped or []
    skipped_clean = skipped_clean or []
    clean_records = [r for r in records if r["trial_kind"] == "clean_control"]
    detect_records = [r for r in records if r["expectation"] == "detect"]
    bypass_records = [r for r in records if r["expectation"] == "known_bypass"]
    known_gap_records = [r for r in records if r["expectation"] == "known_gap"]
    informational_records = [r for r in records if r["expectation"] == "informational"]

    numeric_records = [
        r
        for r in detect_records
        if r["mutation_class"] in {"numeric_delta", "correlation_delta"}
    ]
    numeric_sweep_records = [
        r for r in numeric_records if r.get("magnitude") is not None
    ]
    numeric_ge = [
        r
        for r in numeric_sweep_records
        if float(r["magnitude"]) >= 1e-4
    ]
    calculation_records = [
        r for r in detect_records if r["mutation_class"] == "calculation_delta"
    ]
    structural_records = [
        r for r in detect_records if r["mutation_class"] == "structural"
    ]

    by_operation = group_rate(detect_records, lambda record: record["operation"])
    by_operation_mutation = group_rate(
        detect_records, lambda record: (record["operation"], record["mutation_class"])
    )
    by_mutation_class = group_rate(detect_records, lambda record: record["mutation_class"])
    numeric_by_magnitude = group_rate(
        numeric_sweep_records, lambda record: float(record["magnitude"])
    )
    numeric_by_operation_magnitude = group_rate(
        numeric_sweep_records,
        lambda record: (record["operation"], float(record["magnitude"])),
    )
    calculation_by_operation = group_rate(
        calculation_records, lambda record: record["operation"]
    )
    structural_by_operation = group_rate(
        structural_records, lambda record: record["operation"]
    )
    structural_by_variant = group_rate(
        structural_records,
        lambda record: (record.get("mutation_params") or {}).get("operation", "unknown"),
    )
    structural_by_operation_variant = group_rate(
        structural_records,
        lambda record: (
            record["operation"],
            (record.get("mutation_params") or {}).get("operation", "unknown"),
        ),
    )
    clean_by_operation = group_rate(
        clean_records, lambda record: record["operation"]
    )
    bypass_summary = rate_summary(bypass_records)
    known_gap_summary = rate_summary(known_gap_records)
    informational_summary = rate_summary(informational_records)

    misses = [r for r in detect_records if not r["detected"]]
    miss_groups: dict = {}
    for record in misses:
        key = (record["fixture_id"], record["mutation_class"])
        miss_groups.setdefault(key, []).append(record)
    miss_groups_summary = [
        {
            "fixture_id": key[0],
            "operation": subset_records[0]["operation"],
            "mutation_class": key[1],
            "missed": len(subset_records),
            "examples": [
                {
                    "magnitude": example.get("magnitude"),
                    "target": example.get("target"),
                    "before": example.get("before"),
                    "after": example.get("after"),
                    "verifier_warning_count": example.get("verifier_warning_count"),
                }
                for example in subset_records[:3]
            ],
        }
        for key, subset_records in sorted(miss_groups.items(), key=lambda item: str(item[0]))
    ]

    propagated_detections = [r for r in detect_records if r["detected"]]
    propagated = [r for r in propagated_detections if r["validator_valid"] is False]
    harness_errors = [r for r in records if r.get("harness_error")]
    validation_rejected_without_detection = [
        r
        for r in detect_records
        if not r["detected"] and r["validator_valid"] is False
    ]

    clean_summary = clean_false_positive_summary(clean_records)
    claims = []
    claims.append(
        _claim(
            "V-01",
            "Numeric table-cell tampering at relative magnitude >= 1e-4 is "
            "rejected with an error-severity recomputation finding.",
            numeric_ge,
            scope=f"{len(VERIFIED_OPERATIONS)} verified operation families",
        )
    )
    for operation in VERIFIED_OPERATIONS:
        claims.append(
            _claim(
                f"V-01-{operation}",
                f"Numeric table-cell tampering at relative magnitude >= 1e-4 is "
                f"rejected for `{operation}`.",
                [r for r in numeric_ge if r["operation"] == operation],
                scope=f"operation={operation}",
            )
        )
    claims.append(
        _claim(
            "V-02",
            "Row deletion, extra rows, or changed keys are rejected with an "
            "error-severity finding.",
            [
                r
                for r in structural_records
                if (r.get("mutation_params") or {}).get("operation")
                in {"drop_row", "extra_row", "wrong_key"}
            ],
            scope="all verified operation families; order swaps in keyed tables "
            "are reported separately as informational",
        )
    )
    claims.append(
        _claim(
            "V-02b",
            "Duplicate rows and dropped columns are rejected with an "
            "error-severity finding.",
            [
                r
                for r in structural_records
                if (r.get("mutation_params") or {}).get("operation")
                in {"duplicate_row", "drop_column"}
            ],
            scope="all verified operation families",
            notes="Keyed tables collapse duplicate labels in a dict; several "
            "verifiers return early when a required column is absent.",
        )
    )
    claims.append(
        _claim(
            "V-02c",
            "Correlation matrix row/key structural tampering is rejected.",
            [r for r in structural_records if r["operation"] == "correlation"],
            scope="correlation",
            notes="The correlation verifier checks diagonal, symmetry, and "
            "matrix shape only; it does not compare the row set with the "
            "recomputed labels.",
        )
    )
    claims.append(
        _claim(
            "V-03",
            "Calculation tampering (test statistics, p-values, effect sizes, "
            "confidence intervals, and model-fit calculations) is rejected.",
            calculation_records,
            scope="verifiable calculation targets; correlation's "
            "strongest_correlation is a documented gap",
        )
    )
    claims.append(
        {
            "id": "V-04",
            "statement": "Clean executions produce zero verifier false-positive errors.",
            "scope": f"{clean_summary['trials']} clean controls",
            "status": "supported"
            if clean_summary["false_positive_trials"] == 0
            and clean_summary["invalid_clean_controls"] == 0
            else "gap",
            "notes": "",
            **clean_summary,
        }
    )
    claims.append(
        {
            "id": "V-05",
            "statement": "Every detected verifier error propagates to an invalid "
            "final validation verdict.",
            "scope": f"{len(propagated_detections)} detected tamper trials",
            "status": "supported"
            if propagated_detections
            and len(propagated) == len(propagated_detections)
            else ("gap" if propagated_detections else "no-data"),
            "notes": "",
            "trials": len(propagated_detections),
            "detected": len(propagated),
            "rate": (len(propagated) / len(propagated_detections))
            if propagated_detections
            else None,
        }
    )
    claims.append(
        {
            "id": "V-06",
            "statement": "Truncation-flag bypasses, symmetric correlation changes, "
            "and order-only changes are reported separately and are not counted "
            "as detections.",
            "scope": "documented trust boundaries",
            "status": "documented",
            "notes": (
                "A table can set truncated=true to suppress missing-row findings; "
                "keyed-table row order is treated as presentation-only; a symmetric "
                "correlation change cannot be checked without recomputing coefficients."
            ),
            "known_bypass_trials": bypass_summary["trials"],
            "known_bypass_detected": bypass_summary["detected"],
            "known_gap_trials": known_gap_summary["trials"],
            "known_gap_detected": known_gap_summary["detected"],
            "informational_trials": informational_summary["trials"],
            "informational_detected": informational_summary["detected"],
        }
    )

    return {
        "benchmark": "adversarial_verifier",
        "mode": config.mode,
        "config": {
            "seed": config.seed,
            "rows": config.rows,
            "magnitudes": list(config.magnitudes),
            "count_deltas": list(config.count_deltas),
            "clean_variants": list(config.clean_variants),
            "fixture_ids": list(config.fixture_ids) if config.fixture_ids else None,
        },
        "operations": list(VERIFIED_OPERATIONS),
        "counts": {
            "records": len(records),
            "clean_controls": len(clean_records),
            "tamper_trials": len(detect_records),
            "known_bypass_trials": len(bypass_records),
            "known_gap_trials": len(known_gap_records),
            "informational_trials": len(informational_records),
            "skipped_mutations": len(skipped),
            "skipped_clean_variants": len(skipped_clean),
            "detected": sum(1 for r in detect_records if r["detected"]),
            "missed": len(misses),
            "warning_only": rate_summary(detect_records)["warning_only"],
            "harness_errors": len(harness_errors),
        },
        "clean": clean_summary,
        "by_operation": by_operation,
        "by_operation_mutation": by_operation_mutation,
        "by_mutation_class": by_mutation_class,
        "numeric_by_magnitude": numeric_by_magnitude,
        "numeric_by_operation_magnitude": numeric_by_operation_magnitude,
        "numeric_ge_1e_4": rate_summary(numeric_ge),
        "calculation_by_operation": calculation_by_operation,
        "structural_by_operation": structural_by_operation,
        "structural_by_variant": structural_by_variant,
        "structural_by_operation_variant": structural_by_operation_variant,
        "clean_by_operation": clean_by_operation,
        "known_bypass": bypass_summary,
        "known_gap": known_gap_summary,
        "informational": informational_summary,
        "validation_propagation": {
            "detected_trials": len(propagated_detections),
            "validator_rejected": len(propagated),
            "validator_rejected_without_verifier_error": len(
                validation_rejected_without_detection
            ),
        },
        "miss_groups": miss_groups_summary,
        "misses": [
            {
                "trial_id": record["trial_id"],
                "fixture_id": record["fixture_id"],
                "operation": record["operation"],
                "mutation_class": record["mutation_class"],
                "magnitude": record.get("magnitude"),
                "target": record.get("target"),
                "before": record.get("before"),
                "after": record.get("after"),
                "verifier_warning_count": record.get("verifier_warning_count"),
            }
            for record in misses[:100]
        ],
        "harness_errors": [
            {
                "trial_id": record["trial_id"],
                "fixture_id": record["fixture_id"],
                "error": record["harness_error"],
            }
            for record in harness_errors
        ],
        "skipped": skipped[:50],
        "skipped_clean_variants": skipped_clean[:50],
        "claims": claims,
        "artifact_reference": None,
    }
