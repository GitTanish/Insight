"""Render the load/soak summary and manifest into ``LOAD_SOAK.md``."""

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


def _mb(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{value / 1048576:.2f} MB"


def _latency(entry: dict | None) -> str:
    if not entry:
        return "n/a"
    values = [
        f"p50={entry.get('p50')} ms",
        f"p95={entry.get('p95')} ms",
        f"max={entry.get('max')} ms",
    ]
    return ", ".join(values)


def _detail(record: dict) -> str:
    details = record.get("details", {})
    claim = record.get("claim_id")
    if claim == "L-01":
        return (
            f"workers={details.get('workers')} repeats={details.get('repeats')} "
            f"elapsed={details.get('elapsed_ms')} ms "
            f"signatures={details.get('unique_signatures')}"
        )
    if claim == "L-02":
        return (
            f"status={details.get('status')} page={details.get('page_status')} "
            f"body_ok={details.get('body_ok')} "
            f"elapsed={details.get('elapsed_ms')} ms"
        )
    if claim == "L-03":
        if "rss_growth_mb" in details:
            return (
                f"growth={details.get('rss_growth_mb')} MB "
                f"limit={details.get('rss_limit_mb')} MB "
                f"samples={len(details.get('samples', []))}"
            )
        return (
            f"requests={details.get('requests')} "
            f"failures={details.get('failure_count')} "
            f"throughput={details.get('throughput_rps')} req/s "
            f"p95={details.get('latency_ms', {}).get('p95')} ms"
        )
    if claim == "L-04":
        return (
            f"before={_mb(details.get('size_before_sweep_bytes'))} "
            f"after={_mb(details.get('size_after_sweep_bytes'))} "
            f"limit={_mb(details.get('limit_bytes'))} "
            f"hits={details.get('hits')} misses={details.get('misses')} "
            f"corrupt={details.get('corrupt_reads')}"
        )
    return _cell(details, 100)


def _result(record: dict) -> str:
    return "pass" if record.get("passed") else "FAIL"


def render_load_soak_markdown(
    summary: dict,
    manifest: dict,
    records: list[dict],
    *,
    raw_path: str,
    output_path: Path,
    root: Path,
) -> None:
    counts = summary["counts"]
    metrics = summary.get("metrics", {})
    config = summary.get("config", {})

    lines: list[str] = []
    lines.append("# Insight Load and Soak Evidence")
    lines.append("")
    lines.append(
        "> **Generated artifact.** This file is produced by "
        "`evaluation/bench/run_load_soak.py`; do not edit it by hand. Raw "
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
        f"{_cell(manifest.get('environment', {}).get('platform'))} |"
    )
    lines.append(f"| CPU count | `{manifest.get('environment', {}).get('cpu_count')}` |")
    lines.append(f"| Raw records | `{raw_path}` |")
    lines.append(f"| Manifest | `{manifest['artifacts']['manifest']['path']}` |")
    lines.append("")
    lines.append(
        "These trials run locally against a single-worker uvicorn process and "
        "deterministic fixtures. They are internal evidence on one host, not a "
        "capacity or universal performance claim. See the limitations at the end."
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
        "Wilson 95% intervals cover the scored trials. `supported` means every "
        "scored trial passed on this host; it is not a probability of correctness."
    )
    lines.append("")

    lines.append("## Workload metrics")
    lines.append("")

    analysis = metrics.get("analysis", {})
    lines.append("### Core analysis concurrency")
    lines.append("")
    if analysis:
        lines.append(
            f"**{analysis.get('fixtures')}** fixtures x **{analysis.get('rounds')}** "
            f"round(s), **{analysis.get('repeats')}** concurrent repeats on "
            f"**{analysis.get('workers')}** workers = "
            f"**{analysis.get('executions')}** executions. "
            f"**{analysis.get('passed')}/{analysis.get('trials')}** trials matched "
            f"the serial signature; slowest concurrent batch "
            f"**{analysis.get('max_elapsed_ms')} ms**."
        )
    else:
        lines.append("Not recorded.")
    lines.append("")

    http = metrics.get("http", {})
    lines.append("### Concurrent HTTP uploads")
    lines.append("")
    if http:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Uploads | `{http.get('requests')}` |")
        lines.append(
            f"| Passed / failed | `{http.get('passed')}` / `{http.get('failed')}` |"
        )
        lines.append(f"| Clients | `{http.get('workers')}` |")
        lines.append(f"| Rows per CSV | `{http.get('rows')}` |")
        lines.append(f"| Wall time | `{http.get('wall_s')} s` |")
        lines.append(f"| Throughput | `{http.get('throughput_rps')} uploads/s` |")
        lines.append(f"| Status counts | `{http.get('status_counts')}` |")
        lines.append(f"| Latency | {_latency(http.get('latency_ms'))} |")
    else:
        lines.append("Not recorded (the HTTP harness did not complete).")
    lines.append("")

    soak = metrics.get("soak", {})
    lines.append("### Sustained upload soak")
    lines.append("")
    if soak:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Requested duration | `{config.get('soak_seconds')} s` |")
        lines.append(f"| Observed duration | `{soak.get('duration_s')} s` |")
        lines.append(f"| Clients | `{soak.get('workers')}` |")
        lines.append(f"| Rows per CSV | `{soak.get('rows')}` |")
        lines.append(f"| Upload + page cycles | `{soak.get('requests')}` |")
        lines.append(f"| Failures | `{soak.get('failures')}` |")
        lines.append(f"| Throughput | `{soak.get('throughput_rps')} cycles/s` |")
        lines.append(f"| Latency | {_latency(soak.get('latency_ms'))} |")
        lines.append(f"| Baseline RSS | `{soak.get('rss_baseline_mb')} MB` |")
        lines.append(f"| Peak RSS | `{soak.get('rss_max_mb')} MB` |")
        lines.append(f"| Final RSS | `{soak.get('rss_final_mb')} MB` |")
        lines.append(f"| Peak growth | `{soak.get('rss_growth_mb')} MB` |")
        lines.append(f"| Growth limit | `{soak.get('rss_limit_mb')} MB` |")
        lines.append(f"| RSS samples | `{soak.get('sample_count')}` |")
    else:
        lines.append("Not recorded (the HTTP harness did not complete).")
    lines.append("")

    cache_metrics = metrics.get("cache", {})
    lines.append("### Result-cache churn")
    lines.append("")
    if cache_metrics:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Writes / writers | `{cache_metrics.get('writes')}` / `{cache_metrics.get('writers')}` |")
        lines.append(f"| Readers / reads | `{cache_metrics.get('readers')}` / `{cache_metrics.get('reads')}` |")
        lines.append(
            f"| Hits / misses | `{cache_metrics.get('hits')}` / "
            f"`{cache_metrics.get('misses')}` |"
        )
        lines.append(
            f"| Corrupt reads | `{cache_metrics.get('corrupt_reads')}` |"
        )
        lines.append(
            f"| Size after last write | `{cache_metrics.get('size_before_sweep_mb')} MB` |"
        )
        lines.append(
            f"| Size after sweep | `{cache_metrics.get('size_after_sweep_mb')} MB` |"
        )
        lines.append(f"| Budget | `{cache_metrics.get('limit_mb')} MB` |")
        lines.append(f"| Temp files | `{cache_metrics.get('temp_files')}` |")
    else:
        lines.append("Not recorded.")
    lines.append("")

    lines.append("## Trials")
    lines.append("")
    lines.append(
        f"**{counts['records']}** records: **{counts['scored']}** scored "
        f"(**{counts['passed']}** passed, **{counts['failed']}** failed), "
        f"**{counts['harness_errors']}** harness error(s)."
    )
    lines.append("")
    lines.append("| Claim | Component | Scenario | Result | Expected | Detail |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for record in records:
        lines.append(
            f"| `{record['claim_id']}` | `{_cell(record['component'], 36)}` | "
            f"`{_cell(record['scenario'], 52)}` | {_result(record)} | "
            f"{_cell(record.get('expected'), 78)} | {_cell(_detail(record), 120)} |"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- The core-analysis trial uses deterministic in-process fixtures and "
        "threads, not separate OS processes. It checks result signatures and "
        "exceptions under concurrent calls; it is not a CPU-scaling benchmark."
    )
    lines.append(
        "- HTTP trials cover upload, profiling, briefing generation, and the "
        "session page. They do not call an LLM, planner, verifier, or query "
        "endpoint. Rate limiting is disabled in the child server so the batch "
        "can complete."
    )
    lines.append(
        "- The server runs one uvicorn worker by design because session state is "
        "in process memory. Results do not describe multi-worker or multi-host "
        "deployments."
    )
    lines.append(
        "- Latency numbers are wall-clock observations on the host under test "
        "with synthetic CSVs. They are not an SLO, capacity plan, or comparison "
        "against other hardware."
    )
    lines.append(
        "- RSS is the server process working set sampled at intervals. Allocator "
        "retention, garbage-collection timing, and OS accounting add noise. The "
        "growth limit is a regression guard, not proof that a leak is impossible."
    )
    lines.append(
        "- The soak is deliberately bounded-duration. It demonstrates stability "
        "for this run and workload, not multi-hour or multi-day endurance."
    )
    lines.append(
        "- `store_cached_response` sweeps before publishing, so the cache may "
        "briefly hold the budget plus the payload being written. On Windows a "
        "reader can also hold a file open long enough to refuse an eviction; "
        "failed deletes are not counted, and a later sweep retries them. The "
        "trial records the size after the last write and after an explicit "
        "sweep."
    )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
