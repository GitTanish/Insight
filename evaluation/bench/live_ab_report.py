"""Render the live A/B summary and manifest into ``LIVE_AB.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _cell(value: Any, limit: int = 160) -> str:
    text = "-" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _observed(arm: dict) -> str:
    if not arm.get("trials"):
        return "n/a"
    low, high = arm.get("wilson_95", [0.0, 0.0])
    return f"{arm.get('passed', 0)}/{arm['trials']} [{low:.1%}, {high:.1%}]"


def _case_status(value: str) -> str:
    return {"pass": "pass", "fail": "**FAIL**", "missing": "n/a"}.get(value, value)


def render_live_ab_markdown(
    summary: dict,
    manifest: dict,
    records: list[dict],
    *,
    raw_path: str,
    output_path: Path,
    root: Path,
) -> None:
    config = summary["config"]
    counts = summary["counts"]
    arms = summary["arms"]
    comparisons = summary.get("comparisons", [])
    case_ids = config.get("case_ids", [])

    lines: list[str] = []
    lines.append("# Insight Live A/B and Ablation Evidence")
    lines.append("")
    lines.append(
        "> **Generated artifact.** This file is produced by "
        "`evaluation/bench/run_live_ab.py`; do not edit it by hand. Raw "
        "records are written to the path below and are the source of truth "
        "for every number in this report."
    )
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Benchmark | `{summary['benchmark']}` |")
    lines.append(f"| Run ID | `{manifest['run_id']}` |")
    lines.append(f"| Git commit | `{manifest.get('git', {}).get('commit') or 'unknown'}` |")
    lines.append(f"| Git branch | `{manifest.get('git', {}).get('branch') or 'unknown'}` |")
    lines.append(f"| Working tree dirty | `{manifest.get('git', {}).get('dirty')}` |")
    lines.append(f"| Started (UTC) | `{manifest.get('started_at')}` |")
    lines.append(f"| Finished (UTC) | `{manifest.get('finished_at')}` |")
    lines.append(f"| Python | `{manifest.get('environment', {}).get('python')}` |")
    lines.append(f"| Platform | {_cell(manifest.get('environment', {}).get('platform'))} |")
    lines.append(f"| Model | `{config.get('model_id')}` |")
    if config.get("underlying_model") and config["underlying_model"] != config.get(
        "model_id"
    ):
        lines.append(f"| Underlying model | `{config['underlying_model']}` |")
    aliases = config.get("provider_aliases") or []
    if len(aliases) > 1:
        lines.append(f"| Provider aliases | `{', '.join(aliases)}` |")
    lines.append(f"| Pinned (no fallback) | `{config.get('no_fallback')}` |")
    lines.append(f"| Cases | `{config.get('case_count')}` |")
    lines.append(f"| Repetitions per arm | `{config.get('repetitions')}` |")
    delay_text = config.get("delay_note") or f"{config.get('delay_s')} s"
    lines.append(f"| Inter-case delay | `{delay_text}` |")
    if config.get("quota_note"):
        lines.append(f"| Provider quota | {_cell(config['quota_note'], 200)} |")
    lines.append(f"| Raw records | `{raw_path}` |")
    lines.append(f"| Manifest | `{manifest['artifacts']['manifest']['path']}` |")
    lines.append("")
    lines.append(
        "Every arm runs through `insight.orchestrator.run_analysis_sync` with "
        "the same cases, model pinning, and grader. The planned arm and the "
        "agentic arm therefore share the validator, independent recomputation, "
        "charts, explainer, and grounding tail. The ablations remove one "
        "evaluation-selected guardrail at a time; they do not change shipped "
        "defaults."
    )
    lines.append("")

    source_runs = summary.get("source_runs") or []
    if source_runs:
        lines.append("## Source runs")
        lines.append("")
        lines.append(
            "| Run ID | Commit | Cases | Trials | Status | Raw records |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for source in source_runs:
            commit = (source.get("commit") or "unknown")[:12]
            lines.append(
                f"| `{source.get('run_id')}` | `{commit}` | "
                f"{source.get('case_count')} | "
                f"{source.get('records')}/{source.get('expected')} | "
                f"`{source.get('status')}` | "
                f"`{_cell(source.get('records_path'), 100)}` |"
            )
        lines.append("")
        lines.append(
            "Arms are paired only within a source run; the pooled counts below "
            "reuse the same trial fields. Each source run's manifest and raw "
            "records remain the evidence for its own trials."
        )
        lines.append("")

    lines.append("## Arm summary")
    lines.append("")
    lines.append(
        "| Claim | Arm | Scope | Passed / trials | Wilson 95% | "
        "Validation valid | Guard-reported unverified | Post-hoc unverified | "
        "Grounding retries | Mean calls | Mean wall | Throttled |"
    )
    lines.append(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    claim_by_arm = {claim["arm_id"]: claim for claim in summary["claims"]}
    for arm in arms:
        claim = claim_by_arm.get(arm["arm_id"], {})
        lines.append(
            f"| `{claim.get('id', '-')}` | `{arm['arm_id']}` | "
            f"{_cell(claim.get('scope', ''), 80)} | {_observed(arm)} | "
            f"{_pct(arm.get('pass_rate'))} | {arm.get('validation_valid')}/{arm['trials']} | "
            f"{arm.get('unverified_cases')}/{arm['trials']} | "
            f"{arm.get('posthoc_unverified_cases')}/{arm['trials']} | "
            f"{arm.get('grounding_retries')}/{arm['trials']} | "
            f"{arm.get('mean_llm_calls')} | {arm.get('mean_wall_s')} s | "
            f"{arm.get('throttled')} |"
        )
    lines.append("")
    lines.append(
        "Wilson intervals cover this sample only. `sampled` is deliberate: "
        "these are observations on one endpoint and one curated case set, not "
        "a universal accuracy claim. Guard-reported unverified cases use the "
        "production `unverified_figures` metric, which is only populated when "
        "grounding is enabled. The post-hoc view is approximate and is never "
        "used for pass/fail (see limitations)."
    )
    lines.append("")

    lines.append("## Paired comparisons")
    lines.append("")
    if comparisons:
        lines.append(
            "| Contrast | Paired trials | Left - right pass rate | Both pass | "
            "Both fail | Left only | Right only |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for item in comparisons:
            lines.append(
                f"| `{item['left']}` - `{item['right']}` | "
                f"{item['paired_trials']} | {item['delta_pass_rate']:+.1%} | "
                f"{item['both_pass']} | {item['both_fail']} | "
                f"{item['left_only']} | {item['right_only']} |"
            )
        lines.append("")
        for item in comparisons:
            if item.get("improved_cases") or item.get("regressed_cases"):
                lines.append(
                    f"- `{item['left']}` vs `{item['right']}`: "
                    f"left-only passes {item.get('improved_cases')}; "
                    f"right-only passes {item.get('regressed_cases')}."
                )
        lines.append("")
    else:
        lines.append("No paired comparison was available.")
        lines.append("")

    lines.append("## Per-case matrix")
    lines.append("")
    header = "| Case | " + " | ".join(f"`{arm['arm_id']}`" for arm in arms) + " |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in arms) + " |")
    matrix = summary.get("trial_matrix", {})
    for case_id in case_ids:
        row = matrix.get(case_id, {})
        lines.append(
            f"| `{case_id}` | "
            + " | ".join(_case_status(row.get(arm["arm_id"], "missing")) for arm in arms)
            + " |"
        )
    lines.append("")

    lines.append("## Telemetry and quota")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Expected trials | `{counts.get('expected')}` |")
    lines.append(f"| Recorded trials | `{counts.get('records')}` |")
    lines.append(f"| Missing trials | `{counts.get('missing')}` |")
    lines.append(f"| Passed / failed | `{counts.get('passed')}` / `{counts.get('failed')}` |")
    lines.append(f"| Throttled trials | `{counts.get('throttled')}` |")
    lines.append(f"| Error trials | `{counts.get('errors')}` |")
    lines.append(f"| Trial errors | `{len(summary.get('trial_errors', []))}` |")
    lines.append(f"| Status | `{summary.get('status')}` |")
    total_calls = sum(
        record.get("llm_calls") or 0 for record in records
    )
    total_prompt = sum(
        record.get("prompt_tokens") or 0 for record in records
    )
    total_completion = sum(
        record.get("completion_tokens") or 0 for record in records
    )
    lines.append(f"| Total LLM calls | `{total_calls}` |")
    lines.append(f"| Total prompt tokens | `{total_prompt}` |")
    lines.append(f"| Total completion tokens | `{total_completion}` |")
    lines.append("")
    if summary.get("trial_errors"):
        lines.append("Trial errors:")
        for error in summary["trial_errors"][:10]:
            lines.append(f"- `{_cell(error, 240)}`")
        lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    commands = [
        "```powershell",
        ".venv\\Scripts\\python.exe evaluation\\bench\\run_live_ab.py "
        f"--model {config.get('model_id')} --reps {config.get('repetitions')} "
        f"--delay {config.get('delay_s')} --out-dir <run-dir>",
        "```",
    ]
    if config.get("case_count") and config["case_count"] < 20:
        case_arg = ",".join(config.get("case_ids", []))
        commands.insert(
            1,
            ".venv\\Scripts\\python.exe evaluation\\bench\\run_live_ab.py "
            f"--model {config.get('model_id')} --reps {config.get('repetitions')} "
            f"--delay {config.get('delay_s')} --case-ids {case_arg} --out-dir <run-dir>",
        )
    lines.extend(commands)
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        f"- This is a small, single-run sample on one pinned endpoint "
        f"(`{config.get('model_id')}`) and {config.get('case_count')} curated "
        "synthetic cases. Re-running the model can change the answers. The "
        "Wilson intervals describe this sample, not future runs or other models."
    )
    if config.get("quota_note"):
        lines.append(f"- Provider quota at run time: {config['quota_note']}.")
    lines.append(
        "- Repetitions are sequential and may be throttled by provider quota. "
        "`--delay` spaces cases, not provider calls inside a case. A throttled "
        "or error trial is kept as a failure; it is not retried away."
    )
    lines.append(
        "- The no-verifier arm disables only `independently_verify`. The rest of "
        "`validate_execution` (step success, table sanity, sample size, "
        "missingness, chart sources) still runs."
    )
    lines.append(
        "- The no-grounding arm disables the grounding retry guard. Its "
        "post-hoc unverified count is computed from the response evidence and "
        "tables, so it is approximate; pass/fail always uses the shared grader, "
        "never this diagnostic."
    )
    lines.append(
        "- The grader checks required operations, answer strings, and numeric "
        "ground truth. It does not prove general correctness, and it does not "
        "reward the verifier or grounding guard directly."
    )
    lines.append(
        "- These are end-to-end pipeline results, unlike the older loop-only "
        "`run_eval_agentic.py` telemetry runs. Raw answers and provider "
        "telemetry are internal evidence and stay under `.artifacts/`."
    )
    if source_runs:
        lines.append(
            f"- This report pools {len(source_runs)} independent quota-split "
            "runs. The pooled pass rate is not a single-run estimate, and the "
            "source runs may have used different provider accounts or days. "
            "Use the source manifests to audit each run separately."
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
