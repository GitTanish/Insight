"""Render a benchmark summary and manifest into ``BENCHMARKS.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.bench.adversarial_verifier import VERIFIED_OPERATIONS


def _cell(value: Any, limit: int = 120) -> str:
    text = "-" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _rate(entry: dict | None) -> str:
    if not entry or not entry.get("trials"):
        return "n/a"
    rate = entry.get("rate")
    if rate is None:
        return f"{entry.get('detected', 0)}/{entry['trials']}"
    return (
        f"{entry.get('detected', 0)}/{entry['trials']} ({rate:.1%}) "
        f"[{entry.get('wilson_95_low', 0):.1%}, {entry.get('wilson_95_high', 0):.1%}]"
    )


def _simple_rate(entry: dict | None) -> str:
    if not entry or not entry.get("trials"):
        return "n/a"
    return f"{entry.get('detected', 0)}/{entry['trials']}"


def _ci(entry: dict | None) -> str:
    if not entry or not entry.get("trials"):
        return "n/a"
    return (
        f"[{entry.get('wilson_95_low', 0):.1%}, "
        f"{entry.get('wilson_95_high', 0):.1%}]"
    )


def _target_label(target: dict | None) -> str:
    if not target:
        return "unknown target"
    kind = target.get("kind")
    if kind == "table_cell":
        return f"{target.get('column')} row {target.get('row')}"
    if kind == "calculation":
        return f"calculation `{target.get('label')}`"
    if kind == "correlation_cell":
        suffix = " (symmetric pair)" if target.get("symmetric_pair") else ""
        return f"{target.get('row')} x {target.get('column')}{suffix}"
    if kind == "table_rows":
        return f"rows: {target.get('operation')}"
    if kind == "table_column":
        return f"column `{target.get('column')}`"
    if kind == "table_flag":
        return f"flag `{target.get('flag')}`"
    return _cell(target, 80)


def _lookup(entries: list[dict], key) -> dict | None:
    wanted = list(key) if isinstance(key, tuple) else key
    for entry in entries:
        if entry.get("key") == wanted:
            return entry
    return None


def _claim_line(claim: dict) -> str:
    if claim["id"] == "V-04":
        observed = (
            f"{claim['false_positive_trials']}/{claim['trials']} false positives; "
            f"Wilson 95% upper bound {claim['false_positive_wilson_95_high']:.1%}"
        )
    elif claim["id"] == "V-05":
        observed = f"{claim.get('detected', 0)}/{claim.get('trials', 0)} propagated"
    elif claim["id"] == "V-06":
        observed = (
            f"{claim['known_bypass_detected']}/{claim['known_bypass_trials']} bypass "
            f"detections; {claim['known_gap_detected']}/{claim['known_gap_trials']} "
            f"known-gap detections; {claim['informational_detected']}/"
            f"{claim['informational_trials']} order-only changes"
        )
    else:
        observed = _rate(claim)
    return (
        f"| `{claim['id']}` | {_cell(claim['statement'], 160)} | {claim['status']} | "
        f"{observed} | {_cell(claim.get('scope', ''), 100)} |"
    )


def render_markdown(
    summary: dict,
    manifest: dict,
    *,
    raw_path: str,
    output_path: Path,
    root: Path,
) -> None:
    by_operation_mutation = summary["by_operation_mutation"]
    by_operation = summary["by_operation"]
    numeric_grid = summary["numeric_by_operation_magnitude"]
    magnitudes = [float(m) for m in summary["config"]["magnitudes"]]
    clean = summary["clean"]

    lines: list[str] = []
    lines.append("# Insight Benchmark Evidence")
    lines.append("")
    lines.append(
        "> **Generated artifact.** This file is produced by "
        "`evaluation/bench/run_all.py`; do not edit it by hand. Raw records are "
        "written to the path shown below and are the source of truth for every "
        "number in this report."
    )
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Benchmark | `{summary['benchmark']}` |")
    lines.append(f"| Mode | `{summary['mode']}` |")
    lines.append(f"| Run ID | `{manifest['run_id']}` |")
    lines.append(f"| Git commit | `{manifest['git'].get('commit') or 'unknown'}` |")
    lines.append(f"| Git branch | `{manifest['git'].get('branch') or 'unknown'}` |")
    lines.append(f"| Working tree dirty | `{manifest['git'].get('dirty')}` |")
    lines.append(
        f"| Untracked files | `{manifest['git'].get('untracked_file_count', 0)}` |"
    )
    lines.append(f"| Started (UTC) | `{manifest['started_at']}` |")
    lines.append(f"| Finished (UTC) | `{manifest['finished_at']}` |")
    lines.append(f"| Python | `{manifest['environment'].get('python')}` |")
    lines.append(f"| Platform | `{_cell(manifest['environment'].get('platform'))}` |")
    lines.append(f"| CPU count | `{manifest['environment'].get('cpu_count')}` |")
    lines.append(f"| Raw records | `{raw_path}` |")
    lines.append(f"| Manifest | `{manifest['artifacts']['manifest']['path']}` |")
    lines.append("")
    lines.append(
        "This is **internal evidence**. The fixtures are synthetic and the "
        "matrix is curated, not exhaustive. Detection numbers describe this "
        "matrix; they are not a universal correctness proof. See the limitations "
        "at the end."
    )
    lines.append("")

    lines.append("## Claim summary")
    lines.append("")
    lines.append(
        "| Claim | Statement | Status | Observed | Scope |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for claim in summary["claims"]:
        lines.append(_claim_line(claim))
    lines.append("")
    lines.append(
        "`supported` means zero misses on this matrix. `incomplete` means at "
        "least one miss; `gap` means misses outnumber detections. Inspect the gap "
        "section before making a claim. Confidence intervals are Wilson 95% "
        "intervals."
    )
    lines.append("")

    lines.append("## Clean controls (false-positive cost)")
    lines.append("")
    lines.append(
        f"**{clean['trials']}** clean executions across "
        f"{len(VERIFIED_OPERATIONS)} operations and "
        f"{len(summary['config']['clean_variants'])} data variants. "
        f"Verifier error false positives: **{clean['false_positive_trials']}/{clean['trials']}** "
        f"(Wilson 95% CI "
        f"[{clean['false_positive_wilson_95_low']:.1%}, "
        f"{clean['false_positive_wilson_95_high']:.1%}]). "
        f"Warning-only controls: {clean['warning_control_trials']}. "
        f"Final-validation-invalid clean controls: {clean['invalid_clean_controls']}."
    )
    lines.append("")
    lines.append("| Operation | Clean controls | Error false positives |")
    lines.append("| --- | --- | --- |")
    for entry in summary["clean_by_operation"]:
        lines.append(
            f"| `{_cell(entry['key'])}` | {entry['trials']} | {entry['detected']} |"
        )
    lines.append("")
    skipped_counts = summary["counts"]
    if skipped_counts.get("skipped_clean_variants") or skipped_counts.get("skipped_mutations"):
        lines.append(
            f"Skipped: {skipped_counts.get('skipped_clean_variants', 0)} clean "
            f"variant(s) that were not applicable to their fixture, and "
            f"{skipped_counts.get('skipped_mutations', 0)} mutation(s) that were "
            "no-ops. Details are in the summary JSON."
        )
        lines.append("")

    lines.append("## Detection by operation and tamper class")
    lines.append("")
    lines.append(
        "Only trials marked `expectation=detect` are in this denominator. "
        "Warning-only means the verifier emitted a warning but no error; those "
        "trials are counted as misses for the detection claim."
    )
    lines.append("")
    lines.append("| Operation | Mutation class | Detected / trials | Warning-only | Final validator rejected |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in by_operation_mutation:
        operation, mutation_class = entry["key"]
        lines.append(
            f"| `{_cell(operation)}` | `{_cell(mutation_class)}` | {_rate(entry)} | "
            f"{entry['warning_only']} | {entry['validator_rejected']} |"
        )
    lines.append("")
    lines.append("### Operation totals")
    lines.append("")
    lines.append("| Operation | Detected / trials |")
    lines.append("| --- | --- |")
    for entry in by_operation:
        lines.append(f"| `{_cell(entry['key'])}` | {_rate(entry)} |")
    lines.append("")

    lines.append("## Numeric detection curve")
    lines.append("")
    lines.append(
        "Cells show detected/trials for numeric table-cell tampering at each "
        "relative magnitude. Correlation mutations are single-cell mutations; "
        "symmetric-pair correlation changes are reported as a known gap."
    )
    lines.append("")
    header = "| Magnitude | " + " | ".join(f"`{op}`" for op in VERIFIED_OPERATIONS) + " | Total |"
    lines.append(header)
    lines.append("| --- | " + " | ".join("---" for _ in VERIFIED_OPERATIONS) + " | --- |")
    for magnitude in magnitudes:
        cells = []
        total_detected = 0
        total_trials = 0
        for operation in VERIFIED_OPERATIONS:
            entry = _lookup(numeric_grid, (operation, magnitude))
            cells.append(_simple_rate(entry))
            if entry:
                total_detected += entry["detected"]
                total_trials += entry["trials"]
        lines.append(
            f"| `{magnitude:g}` | " + " | ".join(cells) + f" | {total_detected}/{total_trials} |"
        )
    lines.append("")
    lines.append("### Numeric totals by magnitude")
    lines.append("")
    lines.append("| Magnitude | Detected / trials | Wilson 95% CI |")
    lines.append("| --- | --- | --- |")
    for entry in summary["numeric_by_magnitude"]:
        lines.append(
            f"| `{float(entry['key']):g}` | {_simple_rate(entry)} | {_ci(entry)} |"
        )
    lines.append("")

    lines.append("## Structural and calculation tampering")
    lines.append("")
    lines.append("### Structural variants")
    lines.append("")
    lines.append("| Variant | Detected / trials | 95% CI |")
    lines.append("| --- | --- | --- |")
    for entry in summary["structural_by_variant"]:
        lines.append(
            f"| `{_cell(entry['key'])}` | {_simple_rate(entry)} | {_ci(entry)} |"
        )
    lines.append("")
    lines.append("### By operation")
    lines.append("")
    lines.append("| Operation | Structural detected/trials | Calculation detected/trials |")
    lines.append("| --- | --- | --- |")
    for operation in VERIFIED_OPERATIONS:
        structural = _lookup(summary["structural_by_operation"], operation)
        calculation = _lookup(summary["calculation_by_operation"], operation)
        lines.append(
            f"| `{operation}` | {_simple_rate(structural)} | {_simple_rate(calculation)} |"
        )
    lines.append("")
    lines.append(
        "Keyed-table row-order swaps are `informational`: for group, ANOVA, "
        "regression, correlation, and value-count tables the row identity "
        "carries the meaning; for `top_n` and `compare_subsets` order is "
        "correctness-relevant and those swaps are scored."
    )
    lines.append("")

    lines.append("## Known gaps and trust boundaries")
    lines.append("")
    lines.append(
        f"Known-bypass trials (not counted as detections): "
        f"**{summary['known_bypass']['detected']}/{summary['known_bypass']['trials']}** "
        "detected. Known correlation gaps: "
        f"**{summary['known_gap']['detected']}/{summary['known_gap']['trials']}** "
        "detected by the verifier. Informational order-only trials: "
        f"**{summary['informational']['detected']}/{summary['informational']['trials']}** "
        "detected."
    )
    lines.append("")
    propagation = summary["validation_propagation"]
    lines.append(
        f"Final-validator-only rejections (verifier missed, another validation "
        f"check rejected): **{propagation['validator_rejected_without_verifier_error']}**."
    )
    lines.append("")
    for group in summary["miss_groups"][:30]:
        examples = ", ".join(
            f"{_target_label(example.get('target'))} at magnitude "
            f"`{example.get('magnitude')}`"
            for example in group["examples"]
        )
        lines.append(
            f"- `{group['fixture_id']}` / `{group['mutation_class']}`: "
            f"{group['missed']} miss(es). Examples: {examples}"
        )
    if not summary["miss_groups"]:
        lines.append("- No missed trials in the scored matrix.")
    lines.append("")
    if summary["known_bypass"]["trials"]:
        lines.append(
            "The truncation flag is trusted by design: a result can set "
            "`truncated=true` and suppress missing-row findings. This is a "
            "documented bypass, not a detection success. The correlation "
            "verifier checks diagonal, symmetry, and structure; a symmetric "
            "off-diagonal change and the `strongest_correlation` calculation "
            "are known gaps. The final validator's numeric-sanity check can still "
            "reject a non-finite value that the verifier misses."
        )
        lines.append("")
    if summary["harness_errors"]:
        lines.append("### Harness errors")
        lines.append("")
        for error in summary["harness_errors"]:
            lines.append(
                f"- `{error['trial_id']}` (`{error['fixture_id']}`): "
                f"{_cell(error['error'], 240)}"
            )
        lines.append("")

    lines.append("## Reproduction")
    lines.append("")
    lines.append("From the repository root on PowerShell:")
    lines.append("")
    lines.append("```powershell")
    lines.append(".venv\\Scripts\\python.exe evaluation\\bench\\run_all.py --quick")
    lines.append(".venv\\Scripts\\python.exe evaluation\\bench\\run_all.py --full")
    lines.append("```")
    lines.append("")
    lines.append("On Linux/macOS:")
    lines.append("")
    lines.append("```bash")
    lines.append("python evaluation/bench/run_all.py --quick")
    lines.append("python evaluation/bench/run_all.py --full")
    lines.append("```")
    lines.append("")
    lines.append(
        "The mode, seed, row count, fixture selection, and mutation matrix are "
        "recorded in the manifest. Re-running with the same commit and config "
        "should reproduce the same record counts; floating-point timings may "
        "differ."
    )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Coverage is limited to the seven table-producing operations that "
        "`independently_verify` implements. Other operations are not covered."
    )
    lines.append(
        "- The fixtures are synthetic and deterministic; they do not represent "
        "every real-world schema, dtype, or missingness pattern."
    )
    lines.append(
        "- Mutations are curated. They are not an exhaustive fault model, and "
        "the benchmark does not prove that an unmutated result is correct."
    )
    lines.append(
        "- Detection is measured on the independent verifier. The final "
        "validator also runs structural and numeric-sanity checks, so a trial "
        "can be rejected by the validator even when the verifier misses it."
    )
    lines.append(
        "- Crash durability, concurrent writers, cache publication, alerting, "
        "and live model accuracy are separate benchmarks and are not measured "
        "here."
    )
    lines.append(
        "- This report is internal evidence. Raw records remain under "
        "`.artifacts/` until a sanitized evidence pack is approved."
    )
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
