"""Combine disjoint, quota-split live A/B runs into one paired report.

A 20-case x 4-arm run can exceed a provider's daily token budget. Splitting the
case set across accounts or days keeps each run small enough to finish, but the
result is several independent runs. This tool validates that the runs share the
same model class, arms, repetitions, and pinning; requires disjoint case sets;
pools the raw trials; and regenerates the same summary/report shape as a single
run. Arms are paired only within a source run, and the source manifests remain
the evidence for each run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bench.live_ab import (  # noqa: E402
    Arm,
    compare_arms,
    summarize_arm,
)
from evaluation.bench.live_ab_report import render_live_ab_markdown  # noqa: E402
from evaluation.bench.schema import (  # noqa: E402
    SCHEMA_VERSION,
    command_line,
    environment_info,
    git_info,
    relative_path,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


def load_source_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "live_trials.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "run_dir": run_dir,
        "summary": summary,
        "manifest": manifest,
        "records": records,
    }


def _arm_from_summary(arm: dict) -> Arm:
    return Arm(
        arm_id=arm["id"],
        label=arm["label"],
        agentic=bool(arm["agentic"]),
        verifier=bool(arm["verifier"]),
        grounding=bool(arm["grounding"]),
    )


def _arm_signature(arms: Sequence[dict]) -> list[tuple[str, str, bool, bool, bool]]:
    return [
        (
            arm["id"],
            arm["label"],
            bool(arm["agentic"]),
            bool(arm["verifier"]),
            bool(arm["grounding"]),
        )
        for arm in arms
    ]


def _model_suffix(model_id: str) -> str:
    return model_id.split("/", 1)[1] if "/" in model_id else model_id


def _mean(values: Sequence[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return round(sum(numbers) / len(numbers), 2) if numbers else None


def combine_runs(
    sources: Sequence[dict[str, Any]],
    *,
    run_id: str,
    model_label: str = "",
    quota_note: str = "",
) -> tuple[list[dict], dict[str, Any]]:
    """Validate and pool source runs. Returns pooled records and a summary."""
    if len(sources) < 2:
        raise ValueError("combine at least two source runs")
    for source in sources:
        summary = source["summary"]
        if summary.get("schema") != "live-ab-1":
            raise ValueError(f"unsupported summary schema in {source['run_dir']}")
        if summary.get("benchmark") != "live_ab":
            raise ValueError(f"not a live A/B run: {source['run_dir']}")
        if not source["records"]:
            raise ValueError(f"no trial records in {source['run_dir']}")

    summaries = [source["summary"] for source in sources]
    configs = [summary["config"] for summary in summaries]
    run_ids = [summary["run_id"] for summary in summaries]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("source runs share a run id")

    repetitions = {config["repetitions"] for config in configs}
    if len(repetitions) != 1:
        raise ValueError("source runs use different repetitions")
    no_fallback = {config["no_fallback"] for config in configs}
    if len(no_fallback) != 1:
        raise ValueError("source runs use different fallback settings")
    arm_signatures = {tuple(_arm_signature(config["arms"])) for config in configs}
    if len(arm_signatures) != 1:
        raise ValueError("source runs use different arms")
    arms = [_arm_from_summary(arm) for arm in configs[0]["arms"]]
    arm_by_id = {arm.arm_id: arm for arm in arms}

    model_ids = sorted({config["model_id"] for config in configs})
    model_suffixes = sorted({_model_suffix(model_id) for model_id in model_ids})
    if len(model_suffixes) != 1:
        raise ValueError(
            f"source runs use different models: {', '.join(model_ids)}"
        )
    underlying_model = model_suffixes[0]
    provider_aliases = sorted({model_id.split("/", 1)[0] for model_id in model_ids})
    if len(model_ids) > 1 and not model_label:
        raise ValueError(
            "source runs use different provider aliases "
            f"({', '.join(model_ids)}); pass model_label to name the shared model"
        )
    model_id = model_label or model_ids[0]

    case_ids: list[str] = []
    for config in configs:
        case_ids.extend(config["case_ids"])
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("source runs have overlapping case ids")

    records: list[dict] = []
    source_entries: list[dict[str, Any]] = []
    for source in sources:
        summary = source["summary"]
        config = summary["config"]
        expected_cases = set(config["case_ids"])
        source_records = source["records"]
        seen_keys: set[tuple[str, int, str]] = set()
        for record in source_records:
            case_id = record.get("case_id")
            arm_id = record.get("arm_id")
            repetition = record.get("repetition")
            if case_id not in expected_cases:
                raise ValueError(f"record case {case_id!r} is not in the summary config")
            if arm_id not in arm_by_id:
                raise ValueError(f"record arm {arm_id!r} is not in the summary config")
            if not isinstance(repetition, int) or not 1 <= repetition <= config["repetitions"]:
                raise ValueError(f"record repetition {repetition!r} is out of range")
            key = (arm_id, repetition, case_id)
            if key in seen_keys:
                raise ValueError(f"duplicate trial {key} in {source['run_dir']}")
            seen_keys.add(key)
        records.extend(source_records)

        manifest = source["manifest"]
        source_entries.append(
            {
                "run_id": summary["run_id"],
                "commit": manifest.get("git", {}).get("commit"),
                "branch": manifest.get("git", {}).get("branch"),
                "case_count": config["case_count"],
                "case_ids": list(config["case_ids"]),
                "expected": summary["counts"]["expected"],
                "records": len(source_records),
                "passed": sum(1 for record in source_records if record["passed"]),
                "status": summary.get("status"),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "model_id": config["model_id"],
                "quota_note": config.get("quota_note") or "",
                "run_dir": str(source["run_dir"]),
                "summary_path": str(source["run_dir"] / "summary.json"),
                "manifest_path": str(source["run_dir"] / "manifest.json"),
                "records_path": str(source["run_dir"] / "live_trials.jsonl"),
            }
        )

    reps = repetitions.pop()
    expected_per_arm = len(case_ids) * reps
    arm_summaries = [
        summarize_arm(arm, records, expected_per_arm) for arm in arms
    ]

    comparisons = []
    arm_ids = [arm.arm_id for arm in arms]
    if "agentic" in arm_ids and "baseline" in arm_ids:
        comparisons.append(compare_arms(records, "agentic", "baseline"))
    if "agentic" in arm_ids and "agentic_no_verifier" in arm_ids:
        comparisons.append(
            compare_arms(records, "agentic", "agentic_no_verifier")
        )
    if "agentic" in arm_ids and "agentic_no_grounding" in arm_ids:
        comparisons.append(
            compare_arms(records, "agentic", "agentic_no_grounding")
        )
    comparisons = [item for item in comparisons if item is not None]

    expected = expected_per_arm * len(arms)
    counts = {
        "expected": expected,
        "records": len(records),
        "passed": sum(1 for record in records if record["passed"]),
        "failed": sum(1 for record in records if not record["passed"]),
        "throttled": sum(1 for record in records if record.get("throttled")),
        "errors": sum(1 for record in records if record.get("error")),
    }
    counts["missing"] = counts["expected"] - counts["records"]
    trial_errors = [
        f"{record.get('run_id')}: {record['error']}"
        for record in records
        if record.get("error")
    ][:20]

    claims = []
    for index, arm_summary in enumerate(arm_summaries, start=1):
        claims.append(
            {
                "id": f"A-{index:02d}",
                "arm_id": arm_summary["arm_id"],
                "statement": arm_summary["label"],
                "scope": (
                    f"{arm_summary['trials']} trials on {len(case_ids)} curated "
                    f"cases across {len(sources)} quota-split runs, model "
                    f"{model_id}, k={reps}"
                ),
                "trials": arm_summary["trials"],
                "passed": arm_summary["passed"],
                "rate": arm_summary["pass_rate"],
                "wilson_95_low": arm_summary["wilson_95"][0],
                "wilson_95_high": arm_summary["wilson_95"][1],
                "status": "sampled",
            }
        )

    source_signatures = sorted(
        summary.get("run_signature", "") for summary in summaries
    )
    combined_signature = hashlib.sha256(
        json.dumps(
            {
                "sources": source_signatures,
                "case_ids": case_ids,
                "arms": _arm_signature(configs[0]["arms"]),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    delays = sorted({float(config["delay_s"]) for config in configs})
    if len(delays) == 1:
        delay_note = f"{delays[0]} s"
    else:
        delay_values = ", ".join(f"{value:g}" for value in delays)
        delay_note = f"{delays[-1]} s (max; source runs used {delay_values} s)"
    if not quota_note:
        notes = sorted({entry["quota_note"] for entry in source_entries if entry["quota_note"]})
        quota_note = "; ".join(notes)

    status = (
        "ok"
        if counts["missing"] == 0
        and counts["errors"] == 0
        and not trial_errors
        and counts["throttled"] == 0
        and all(entry["status"] == "ok" for entry in source_entries)
        and all(arm_summary["complete"] for arm_summary in arm_summaries)
        else "attention"
    )

    summary = {
        "schema": "live-ab-1",
        "benchmark": "live_ab_combined",
        "run_id": run_id,
        "config": {
            "model_id": model_id,
            "underlying_model": underlying_model,
            "provider_aliases": provider_aliases,
            "repetitions": reps,
            "delay_s": delays[-1],
            "delay_note": delay_note,
            "no_fallback": configs[0]["no_fallback"],
            "quota_note": quota_note,
            "case_ids": case_ids,
            "case_count": len(case_ids),
            "arms": configs[0]["arms"],
        },
        "run_signature": combined_signature,
        "counts": counts,
        "arms": arm_summaries,
        "comparisons": comparisons,
        "claims": claims,
        "trial_errors": trial_errors,
        "source_runs": source_entries,
        "status": status,
        "trial_matrix": {
            case_id: {
                arm.arm_id: next(
                    (
                        "pass" if record["passed"] else "fail"
                        for record in records
                        if record["arm_id"] == arm.arm_id
                        and record["case_id"] == case_id
                        and record["repetition"] == 1
                    ),
                    "missing",
                )
                for arm in arms
            }
            for case_id in case_ids
        },
    }
    return records, summary


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": relative_path(path, ROOT),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _new_run_id(git: dict) -> str:
    from datetime import datetime, timezone

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commit = (git.get("commit") or "nogit")[:7]
    return f"{stamp}-live-ab-combined-{commit}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        type=Path,
        help="source run directory (repeat for each run)",
    )
    parser.add_argument(
        "--model-label",
        default="",
        help="shared model name when source runs used different provider aliases",
    )
    parser.add_argument(
        "--quota-note",
        default="",
        help="quota note for the combined report (default: join source notes)",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "LIVE_AB.md",
        help="Markdown report path (default: LIVE_AB.md)",
    )
    parser.add_argument("--no-report", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    git = git_info(ROOT)
    run_id = _new_run_id(git)
    out_dir = args.out_dir or (ROOT / ".artifacts" / "bench" / "live-ab" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "live_trials.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"

    sources = [load_source_run(path) for path in args.run_dir]
    records, summary = combine_runs(
        sources,
        run_id=run_id,
        model_label=args.model_label,
        quota_note=args.quota_note,
    )
    summary["run_id"] = run_id

    source_entries = []
    for source in sources:
        entry = next(
            item
            for item in summary["source_runs"]
            if item["run_id"] == source["summary"]["run_id"]
        )
        source_entries.append(
            {
                **entry,
                "summary_path": relative_path(
                    Path(entry["summary_path"]), ROOT
                ),
                "manifest_path": relative_path(
                    Path(entry["manifest_path"]), ROOT
                ),
                "records_path": relative_path(
                    Path(entry["records_path"]), ROOT
                ),
                "summary_sha256": sha256_file(Path(entry["summary_path"])),
                "manifest_sha256": sha256_file(Path(entry["manifest_path"])),
                "records_sha256": sha256_file(Path(entry["records_path"])),
            }
        )

    summary["source_runs"] = source_entries
    write_jsonl(records_path, records)
    write_json(summary_path, summary)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "live_ab_combined",
        "run_id": run_id,
        "started_at": min(
            (entry["started_at"] for entry in source_entries if entry["started_at"]),
            default=None,
        ),
        "finished_at": max(
            (entry["finished_at"] for entry in source_entries if entry["finished_at"]),
            default=None,
        ),
        "combined_at": utc_now(),
        "status": summary["status"],
        "git": git,
        "environment": environment_info(),
        "command": command_line(),
        "config": summary["config"],
        "run_signature": summary["run_signature"],
        "counts": summary["counts"],
        "source_runs": source_entries,
        "artifacts": {
            "records": {"path": relative_path(records_path, ROOT)},
            "summary": {"path": relative_path(summary_path, ROOT)},
            "manifest": {"path": relative_path(manifest_path, ROOT)},
        },
        "notes": [
            "Internal evidence run; raw records and answers are not published.",
            "Pools disjoint quota-split runs; arms are paired within each source run.",
            "Every arm uses the production run_analysis_sync pipeline.",
            "The model is pinned unless a source run allowed fallback.",
            "The sample is curated and small; intervals are not universal claims.",
        ],
    }

    if not args.no_report:
        render_live_ab_markdown(
            summary,
            manifest,
            records,
            raw_path=relative_path(records_path, ROOT),
            output_path=args.report,
            root=ROOT,
        )
        manifest["artifacts"]["report"] = _artifact(args.report)

    manifest["artifacts"]["records"] = _artifact(records_path)
    manifest["artifacts"]["summary"] = _artifact(summary_path)
    write_json(manifest_path, manifest)

    counts = summary["counts"]
    print(
        "trials: {records}/{expected}; passed: {passed}; failed: {failed}; "
        "throttled: {throttled}; errors: {errors}; missing: {missing}".format(
            **counts
        )
    )
    print(f"status: {summary['status']}")
    for entry in source_entries:
        print(
            f"source: {entry['run_id']} [{entry['status']}] "
            f"{entry['records']}/{entry['expected']} trials, commit {entry['commit']}"
        )
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    if not args.no_report:
        print(f"report: {args.report}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
