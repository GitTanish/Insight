"""Render durability summary and manifest into ``DURABILITY.md``."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _cell(value: Any, limit: int = 160) -> str:
    text = "-" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _observed(claim: dict) -> str:
    if not claim.get("trials"):
        return "n/a"
    return (
        f"{claim.get('passed', 0)}/{claim['trials']} "
        f"[{claim.get('wilson_95_low', 0):.1%}, "
        f"{claim.get('wilson_95_high', 0):.1%}]"
    )


def _result(record: dict) -> str:
    if record.get("expectation") == "known_boundary":
        return "documented boundary"
    return "pass" if record.get("passed") else "FAIL"


def render_durability_markdown(
    summary: dict,
    manifest: dict,
    records: list[dict],
    *,
    raw_path: str,
    output_path: Path,
    root: Path,
) -> None:
    counts = summary["counts"]
    boundaries = summary.get("known_boundaries", [])

    lines: list[str] = []
    lines.append("# Insight Durability Evidence")
    lines.append("")
    lines.append(
        "> **Generated artifact.** This file is produced by "
        "`evaluation/bench/run_durability.py`; do not edit it by hand. Raw "
        "records are written to the path shown below and are the source of "
        "truth for every number in this report."
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
    lines.append(
        f"| Working tree dirty | `{manifest.get('git', {}).get('dirty')}` |"
    )
    lines.append(
        f"| Untracked files | "
        f"`{manifest.get('git', {}).get('untracked_file_count', 0)}` |"
    )
    lines.append(f"| Started (UTC) | `{manifest.get('started_at')}` |")
    lines.append(f"| Finished (UTC) | `{manifest.get('finished_at')}` |")
    lines.append(f"| Python | `{manifest.get('environment', {}).get('python')}` |")
    lines.append(
        f"| Platform | "
        f"`{_cell(manifest.get('environment', {}).get('platform'))}` |"
    )
    lines.append(f"| CPU count | `{manifest.get('environment', {}).get('cpu_count')}` |")
    lines.append(f"| Raw records | `{raw_path}` |")
    lines.append(f"| Manifest | `{manifest['artifacts']['manifest']['path']}` |")
    lines.append("")
    lines.append(
        "These trials inject process-style faults on the local filesystem. "
        "They are internal evidence on one host, not a universal durability "
        "proof. See the limitations at the end."
    )
    lines.append("")

    lines.append("## Claim summary")
    lines.append("")
    lines.append("| Claim | Statement | Status | Passed / trials | Scope |")
    lines.append("| --- | --- | --- | --- | --- |")
    for claim in summary["claims"]:
        lines.append(
            f"| `{claim['id']}` | {_cell(claim['statement'], 140)} | "
            f"{claim['status']} | {_observed(claim)} | "
            f"{_cell(claim.get('scope', ''), 110)} |"
        )
    lines.append("")
    lines.append(
        "Wilson 95% intervals over the scored trials. "
        "`supported` means every scored trial passed on this host; it is not a "
        "probability of correctness."
    )
    lines.append("")

    lines.append("## Trials")
    lines.append("")
    lines.append(
        f"**{counts['records']}** records: **{counts['scored']}** scored "
        f"(**{counts['passed']}** passed, **{counts['failed']}** failed), "
        f"**{counts['known_boundaries']}** documented boundary trial(s), "
        f"**{counts['harness_errors']}** harness error(s)."
    )
    lines.append("")
    lines.append("| Claim | Component | Scenario | Expectation | Result | Expected |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for record in records:
        lines.append(
            f"| `{record['claim_id']}` | `{_cell(record['component'], 40)}` | "
            f"`{_cell(record['scenario'], 60)}` | {record['expectation']} | "
            f"{_result(record)} | {_cell(record.get('expected'), 90)} |"
        )
    lines.append("")

    lines.append("## Known boundaries")
    lines.append("")
    if boundaries:
        lines.append("| Claim | Scenario | Observed | Why it remains a boundary |")
        lines.append("| --- | --- | --- | --- |")
        for record in boundaries:
            lines.append(
                f"| `{record['claim_id']}` | `{_cell(record['scenario'], 60)}` | "
                f"{_result(record)} | "
                f"{_cell(record.get('details', {}).get('why'), 220)} |"
            )
    else:
        lines.append("No boundary trials were recorded.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The ledger chain detects edits, reordering, mid-ledger deletion, "
        "duplication, torn tails, and appended garbage. Deleting a complete "
        "tail leaves a shorter internally valid chain and is **not** detectable "
        "without an externally pinned head; the boundary trial records this."
    )
    lines.append(
        "- A corrupt alert-runner state file is treated as a first run so "
        "monitoring continues; an alert that was already delivered may repost. "
        "This is a documented boundary, not a delivery exactly-once guarantee."
    )
    lines.append(
        "- Concurrent readers can observe a safe cache miss while a payload is "
        "replaced; the trial counts misses but requires zero corrupt reads and "
        "no leftover temp files."
    )
    lines.append(
        "- Faults are process-style operations on the local filesystem. Power "
        "loss, fsync loss, disk-full `ENOSPC`, filesystem corruption, and "
        "network partitions beyond a failed webhook POST are not simulated."
    )
    lines.append(
        "- Concurrency is a small fixed workload on one host. It shows that "
        "appends are serialized and that no updates are lost in this run; it is "
        "not a throughput or scaling benchmark."
    )
    lines.append(
        "- Readiness trials check path writability, not a full analysis, model "
        "call, or end-to-end request."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
