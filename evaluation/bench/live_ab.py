"""Live, quota-aware A/B and guardrail ablations on the production pipeline.

Unlike ``evaluation/run_eval_agentic.py`` (which scores the tool loop's final
answer directly), this harness drives ``run_analysis_sync`` for every arm. That
means both the planned and agentic paths run through the same validator,
independent recomputation, chart synthesis, explainer, and grounding guard.

The guardrail ablations are evaluation-only:

* ``agentic_no_verifier`` suppresses ``independently_verify`` while leaving the
  rest of ``validate_execution`` intact.
* ``agentic_no_grounding`` clears ``settings.answer_grounded_check`` for the
  arm, disabling the grounding retry guard.

No shipped default changes. The harness restores every setting it mutates,
writes each trial as it completes (for resume after throttling), and is meant
for small samples on a pinned model. It makes no universal-accuracy claim.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from unittest import mock

import pandas as pd

from evaluation.ab_stats import wilson_ci
from evaluation.run_eval import DEFAULT_CASES, load_cases, score_case
from insight.domain.query import AnalysisRequest
from insight.domain.visualization import AnalysisResponse
from insight.settings import get_settings

_MUTATED_SETTINGS = ("answer_grounded_check", "agentic_call_delay_s")


@dataclass(frozen=True)
class Arm:
    arm_id: str
    label: str
    agentic: bool
    verifier: bool = True
    grounding: bool = True


DEFAULT_ARMS: tuple[Arm, ...] = (
    Arm("baseline", "Single-shot plan, full tail", agentic=False),
    Arm("agentic", "Agentic loop, full tail", agentic=True),
    Arm(
        "agentic_no_verifier",
        "Agentic loop, independent verifier disabled",
        agentic=True,
        verifier=False,
    ),
    Arm(
        "agentic_no_grounding",
        "Agentic loop, grounding retry disabled",
        agentic=True,
        grounding=False,
    ),
)

ARM_BY_ID = {arm.arm_id: arm for arm in DEFAULT_ARMS}


@dataclass(frozen=True)
class LiveConfig:
    model_id: str
    repetitions: int = 1
    delay_s: float = 2.0
    no_fallback: bool = True
    case_ids: tuple[str, ...] = ()
    limit: int = 0


Runner = Callable[[pd.DataFrame, Any, AnalysisRequest, Path], AnalysisResponse]


def _default_runner(
    df: pd.DataFrame,
    profile: Any,
    request: AnalysisRequest,
    artifacts_dir: Path,
) -> AnalysisResponse:
    from insight.orchestrator import run_analysis_sync

    return run_analysis_sync(
        df,
        profile,
        request,
        artifacts_dir=artifacts_dir,
        use_cache=False,
    )


def select_cases(
    cases: Sequence[dict],
    case_ids: Sequence[str] = (),
    limit: int = 0,
) -> list[dict]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case["id"] in wanted]
        missing = sorted(wanted - {case["id"] for case in selected})
        if missing:
            raise ValueError(f"unknown case id(s): {', '.join(missing)}")
    else:
        selected = list(cases)
    if limit and limit > 0:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no cases selected")
    return selected


def run_signature(config: LiveConfig, arms: Sequence[Arm], cases: Sequence[dict]) -> str:
    payload = {
        "model_id": config.model_id,
        "repetitions": config.repetitions,
        "no_fallback": config.no_fallback,
        "case_ids": [case["id"] for case in cases],
        "arms": [
            {
                "id": arm.arm_id,
                "agentic": arm.agentic,
                "verifier": arm.verifier,
                "grounding": arm.grounding,
            }
            for arm in arms
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def trial_key(arm: Arm, repetition: int, case_id: str) -> str:
    return f"{arm.arm_id}|{repetition}|{case_id}"


def _posthoc_unverified(response: AnalysisResponse) -> list[str]:
    """Approximate residual grounding check independent of the guard setting.

    The production metric in ``meta["unverified_figures"]`` is authoritative
    when grounding is enabled. When it is disabled this post-hoc view still
    gives a comparable signal, but it only sees the response evidence (up to
    12 calculations) and tables; it is not used for pass/fail.
    """
    from insight.validation.grounding import find_ungrounded_numbers

    calcs = [item for item in response.evidence if item.kind == "calculation"]
    warnings = [item.label for item in response.evidence if item.kind == "warning"]
    return find_ungrounded_numbers(
        response.answer,
        calcs,
        response.tables,
        warnings=warnings,
    )


def _safe_trial(
    *,
    run_id: str,
    run_sig: str,
    arm: Arm,
    repetition: int,
    case: dict,
    response: AnalysisResponse | None,
    error: str | None,
    wall_s: float,
    artifacts_dir: Path,
) -> dict[str, Any]:
    meta = (response.meta if response is not None else {}) or {}
    validation = response.validation if response is not None else None
    validation_errors = (
        [issue.message for issue in validation.errors] if validation is not None else []
    )
    validation_valid = meta.get("validation_valid")
    if validation_valid is None and validation is not None:
        validation_valid = validation.valid
    passed = False
    reasons: list[str] = []
    if response is not None:
        try:
            passed, reasons = score_case(response, case)
        except Exception as exc:  # noqa: BLE001 - a scorer failure is evidence
            passed = False
            reasons = [f"scorer error: {type(exc).__name__}: {exc}"[:300]]
    if error:
        passed = False
        reasons = [error]

    unverified = list(meta.get("unverified_figures") or [])
    posthoc: list[str] = []
    if response is not None:
        try:
            posthoc = _posthoc_unverified(response)
        except Exception:  # noqa: BLE001 - an instrumentation miss is not a pass
            posthoc = []

    steps = meta.get("steps") or []
    operations = [
        item.get("operation")
        for item in steps
        if isinstance(item, dict) and item.get("success")
    ]
    llm_calls = meta.get("llm_calls")
    throttled = bool(meta.get("agentic_throttled")) or bool(
        error and "throttl" in error.lower()
    )
    return {
        "schema": "live-ab-1",
        "run_id": run_id,
        "run_signature": run_sig,
        "trial_key": trial_key(arm, repetition, case["id"]),
        "arm_id": arm.arm_id,
        "arm_label": arm.label,
        "agentic": arm.agentic,
        "verifier_enabled": arm.verifier,
        "grounding_enabled": arm.grounding,
        "repetition": repetition,
        "case_id": case["id"],
        "dataset": case["dataset"],
        "question": case["question"],
        "passed": bool(passed),
        "reasons": reasons,
        "error": error or (response.error if response is not None else None),
        "answer": response.answer if response is not None else None,
        "answer_chars": len(response.answer) if response is not None else None,
        "validation_valid": validation_valid,
        "validation_errors": validation_errors,
        "unverified_figures": unverified,
        "posthoc_unverified": posthoc,
        "grounding_retry_ms": meta.get("grounding_retry_ms"),
        "llm_calls": llm_calls,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": meta.get("completion_tokens"),
        "model_used": meta.get("model"),
        "agentic_mode": bool(meta.get("agentic")),
        "operations_used": operations,
        "throttled": throttled,
        "wall_s": round(wall_s, 2),
        "total_ms": meta.get("total_ms"),
        "case_artifacts": str(artifacts_dir),
    }


def run_arm(
    *,
    run_id: str,
    run_sig: str,
    arm: Arm,
    cases: Sequence[dict],
    frames: dict[str, tuple[pd.DataFrame, Any]],
    config: LiveConfig,
    out_dir: Path,
    runner: Runner | None = None,
    existing_keys: set[str] | None = None,
    on_trial: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run one arm sequentially and return the new trial records."""
    runner = runner or _default_runner
    existing_keys = existing_keys or set()
    settings = get_settings()
    snapshot = {name: getattr(settings, name) for name in _MUTATED_SETTINGS}
    records: list[dict] = []

    try:
        settings.answer_grounded_check = arm.grounding
        settings.agentic_call_delay_s = max(0.0, float(config.delay_s))
        verifier_patch = (
            nullcontext()
            if arm.verifier
            else mock.patch(
                "insight.validation.verification.independently_verify",
                return_value=[],
            )
        )
        with verifier_patch:
            for repetition in range(1, config.repetitions + 1):
                for case_index, case in enumerate(cases):
                    key = trial_key(arm, repetition, case["id"])
                    if key in existing_keys:
                        continue
                    df, profile = frames[case["dataset"]]
                    artifacts_dir = (
                        out_dir
                        / "case-artifacts"
                        / arm.arm_id
                        / f"rep-{repetition}"
                        / case["id"]
                    )
                    artifacts_dir.mkdir(parents=True, exist_ok=True)
                    request = AnalysisRequest(
                        question=case["question"],
                        model_id=config.model_id,
                        agentic=arm.agentic,
                        no_fallback=config.no_fallback,
                    )
                    started = time.monotonic()
                    response = None
                    error = None
                    try:
                        response = runner(df, profile, request, artifacts_dir)
                    except Exception as exc:  # noqa: BLE001 - a live failure is evidence
                        error = f"{type(exc).__name__}: {exc}"[:500]
                    if response is None and error is None:
                        error = "no response produced"
                    wall_s = time.monotonic() - started
                    record = _safe_trial(
                        run_id=run_id,
                        run_sig=run_sig,
                        arm=arm,
                        repetition=repetition,
                        case=case,
                        response=response,
                        error=error,
                        wall_s=wall_s,
                        artifacts_dir=artifacts_dir,
                    )
                    records.append(record)
                    if on_trial is not None:
                        on_trial(record)
                    if config.delay_s > 0 and not (
                        repetition == config.repetitions
                        and case_index == len(cases) - 1
                    ):
                        time.sleep(config.delay_s)
    finally:
        for name, value in snapshot.items():
            setattr(settings, name, value)
    return records


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return round(sum(numbers) / len(numbers), 2) if numbers else None


def summarize_arm(arm: Arm, records: Sequence[dict], expected: int) -> dict[str, Any]:
    arm_records = [record for record in records if record["arm_id"] == arm.arm_id]
    trials = len(arm_records)
    passed = sum(1 for record in arm_records if record["passed"])
    low, high = wilson_ci(passed, trials) if trials else (0.0, 0.0)
    validation_valid = sum(
        1 for record in arm_records if record.get("validation_valid") is True
    )
    unverified_cases = sum(
        1 for record in arm_records if record.get("unverified_figures")
    )
    posthoc_cases = sum(
        1 for record in arm_records if record.get("posthoc_unverified")
    )
    grounding_retries = sum(
        1
        for record in arm_records
        if isinstance(record.get("grounding_retry_ms"), (int, float))
        and record["grounding_retry_ms"] > 0
    )
    return {
        "arm_id": arm.arm_id,
        "label": arm.label,
        "agentic": arm.agentic,
        "verifier_enabled": arm.verifier,
        "grounding_enabled": arm.grounding,
        "expected": expected,
        "trials": trials,
        "passed": passed,
        "failed": trials - passed,
        "pass_rate": round(passed / trials, 3) if trials else None,
        "wilson_95": [round(low, 3), round(high, 3)],
        "validation_valid": validation_valid,
        "validation_invalid": trials - validation_valid,
        "unverified_cases": unverified_cases,
        "posthoc_unverified_cases": posthoc_cases,
        "grounding_retries": grounding_retries,
        "mean_llm_calls": _mean(record.get("llm_calls") for record in arm_records),
        "total_prompt_tokens": sum(
            record.get("prompt_tokens") or 0 for record in arm_records
        ),
        "total_completion_tokens": sum(
            record.get("completion_tokens") or 0 for record in arm_records
        ),
        "mean_wall_s": _mean(record.get("wall_s") for record in arm_records),
        "throttled": sum(1 for record in arm_records if record.get("throttled")),
        "errors": sum(1 for record in arm_records if record.get("error")),
        "failed_cases": sorted(
            record["case_id"] for record in arm_records if not record["passed"]
        ),
        "failure_reasons": sorted(
            {
                reason
                for record in arm_records
                if not record["passed"]
                for reason in (record.get("reasons") or [])
            }
        ),
        "complete": trials == expected,
    }


def compare_arms(
    records: Sequence[dict],
    left: str,
    right: str,
) -> dict[str, Any] | None:
    left_map = {
        (record["case_id"], record["repetition"]): record
        for record in records
        if record["arm_id"] == left
    }
    right_map = {
        (record["case_id"], record["repetition"]): record
        for record in records
        if record["arm_id"] == right
    }
    shared = sorted(set(left_map) & set(right_map))
    if not shared:
        return None
    left_passes = sum(1 for key in shared if left_map[key]["passed"])
    right_passes = sum(1 for key in shared if right_map[key]["passed"])
    both_pass = sum(
        1
        for key in shared
        if left_map[key]["passed"] and right_map[key]["passed"]
    )
    both_fail = sum(
        1
        for key in shared
        if not left_map[key]["passed"] and not right_map[key]["passed"]
    )
    left_only = sum(
        1
        for key in shared
        if left_map[key]["passed"] and not right_map[key]["passed"]
    )
    right_only = sum(
        1
        for key in shared
        if right_map[key]["passed"] and not left_map[key]["passed"]
    )
    return {
        "left": left,
        "right": right,
        "paired_trials": len(shared),
        "left_passes": left_passes,
        "right_passes": right_passes,
        "delta_pass_rate": round((left_passes - right_passes) / len(shared), 3),
        "both_pass": both_pass,
        "both_fail": both_fail,
        "left_only": left_only,
        "right_only": right_only,
        "improved_cases": sorted(
            key[0] for key in shared if left_map[key]["passed"] and not right_map[key]["passed"]
        ),
        "regressed_cases": sorted(
            key[0] for key in shared if right_map[key]["passed"] and not left_map[key]["passed"]
        ),
    }


def _load_existing(path: Path, run_sig: str) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("run_signature") == run_sig:
            records.append(record)
    return records


def run_live_ab(
    *,
    run_id: str,
    config: LiveConfig,
    arms: Sequence[Arm],
    out_dir: Path,
    cases: Sequence[dict] | None = None,
    frames: dict[str, tuple[pd.DataFrame, Any]] | None = None,
    data_dir: Path | None = None,
    runner: Runner | None = None,
    resume: bool = False,
) -> tuple[list[dict], dict]:
    import pandas as pd

    from insight.profiling.profiler import profile_dataframe

    arm_ids = [arm.arm_id for arm in arms]
    if len(set(arm_ids)) != len(arm_ids):
        raise ValueError("duplicate arm ids")
    if config.repetitions < 1:
        raise ValueError("repetitions must be >= 1")

    cases = (
        select_cases(cases, config.case_ids, config.limit)
        if cases is not None
        else select_cases(load_cases(DEFAULT_CASES), config.case_ids, config.limit)
    )
    if frames is None:
        if data_dir is None:
            from evaluation.run_eval import DEFAULT_DATA

            data_dir = DEFAULT_DATA
        frames = {}
        for case in cases:
            name = case["dataset"]
            if name in frames:
                continue
            frame = pd.read_csv(data_dir / name)
            profile = profile_dataframe(
                frame, name, content_hash=f"live{name}{len(frame)}"[:32]
            )
            frames[name] = (frame, profile)

    run_sig = run_signature(config, arms, cases)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "live_trials.jsonl"
    if resume:
        existing = _load_existing(records_path, run_sig)
    else:
        existing = []
        if records_path.exists():
            records_path.unlink()
    existing_keys = {record["trial_key"] for record in existing}
    if existing:
        # A resumed run keeps the identity of the run it is completing.
        run_id = existing[0].get("run_id") or run_id

    def append_trial(record: dict) -> None:
        with records_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        status = "PASS" if record["passed"] else "FAIL"
        print(
            f"[{record['arm_id']}] {record['case_id']} {status} "
            f"calls={record.get('llm_calls')} wall={record.get('wall_s')}s",
            flush=True,
        )

    all_records = list(existing)
    for arm_index, arm in enumerate(arms):
        expected_per_arm = len(cases) * config.repetitions
        print(f"[{arm.arm_id}] {arm.label} ({expected_per_arm} trials)")
        arm_records = run_arm(
            run_id=run_id,
            run_sig=run_sig,
            arm=arm,
            cases=cases,
            frames=frames,
            config=config,
            out_dir=out_dir,
            runner=runner,
            existing_keys=existing_keys,
            on_trial=append_trial,
        )
        all_records.extend(arm_records)
        if config.delay_s > 0 and arm_index < len(arms) - 1:
            time.sleep(config.delay_s)

    # Preserve a deterministic raw order for later aggregation.
    all_records.sort(key=lambda item: (item["arm_id"], item["repetition"], item["case_id"]))
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    expected_per_arm = len(cases) * config.repetitions
    arm_summaries = [
        summarize_arm(arm, all_records, expected_per_arm) for arm in arms
    ]
    comparisons = []
    if "agentic" in arm_ids and "baseline" in arm_ids:
        comparisons.append(compare_arms(all_records, "agentic", "baseline"))
    if "agentic" in arm_ids and "agentic_no_verifier" in arm_ids:
        comparisons.append(
            compare_arms(all_records, "agentic", "agentic_no_verifier")
        )
    if "agentic" in arm_ids and "agentic_no_grounding" in arm_ids:
        comparisons.append(
            compare_arms(all_records, "agentic", "agentic_no_grounding")
        )
    comparisons = [item for item in comparisons if item is not None]

    counts = {
        "expected": expected_per_arm * len(arms),
        "records": len(all_records),
        "passed": sum(1 for record in all_records if record["passed"]),
        "failed": sum(1 for record in all_records if not record["passed"]),
        "throttled": sum(1 for record in all_records if record.get("throttled")),
        "errors": sum(1 for record in all_records if record.get("error")),
    }
    counts["missing"] = counts["expected"] - counts["records"]
    harness_errors = [
        record["error"]
        for record in all_records
        if record.get("error")
    ][:20]

    claims = []
    for index, summary in enumerate(arm_summaries, start=1):
        claims.append(
            {
                "id": f"A-{index:02d}",
                "arm_id": summary["arm_id"],
                "statement": summary["label"],
                "scope": (
                    f"{summary['trials']} trials on {len(cases)} curated cases, "
                    f"model {config.model_id}, k={config.repetitions}"
                ),
                "trials": summary["trials"],
                "passed": summary["passed"],
                "rate": summary["pass_rate"],
                "wilson_95_low": summary["wilson_95"][0],
                "wilson_95_high": summary["wilson_95"][1],
                "status": "sampled",
            }
        )

    summary = {
        "schema": "live-ab-1",
        "benchmark": "live_ab",
        "run_id": run_id,
        "config": {
            "model_id": config.model_id,
            "repetitions": config.repetitions,
            "delay_s": config.delay_s,
            "no_fallback": config.no_fallback,
            "case_ids": [case["id"] for case in cases],
            "case_count": len(cases),
            "arms": [
                {
                    "id": arm.arm_id,
                    "label": arm.label,
                    "agentic": arm.agentic,
                    "verifier": arm.verifier,
                    "grounding": arm.grounding,
                }
                for arm in arms
            ],
        },
        "run_signature": run_sig,
        "counts": counts,
        "arms": arm_summaries,
        "comparisons": comparisons,
        "claims": claims,
        "harness_errors": harness_errors,
        "status": (
            "ok"
            if counts["missing"] == 0
            and counts["errors"] == 0
            and not harness_errors
            and counts["throttled"] == 0
            and all(summary["complete"] for summary in arm_summaries)
            else "attention"
        ),
        "trial_matrix": {
            case["id"]: {
                arm.arm_id: next(
                    (
                        "pass" if record["passed"] else "fail"
                        for record in all_records
                        if record["arm_id"] == arm.arm_id
                        and record["case_id"] == case["id"]
                        and record["repetition"] == 1
                    ),
                    "missing",
                )
                for arm in arms
            }
            for case in cases
        },
    }
    return all_records, summary
