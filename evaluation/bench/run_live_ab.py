"""Run the live, quota-aware A/B and guardrail ablation suite.

Usage:
    python evaluation/bench/run_live_ab.py --model nvidia/openai/gpt-oss-20b
    python evaluation/bench/run_live_ab.py --model ... --case-ids sales_top_region --arms agentic

The run is sequential by design. Each completed trial is appended to
``live_trials.jsonl`` so an interrupted or throttled run can be resumed with
``--resume`` instead of spending quota twice. The model is pinned by default;
pass ``--allow-fallback`` only when fallback is part of the question.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bench.live_ab import (  # noqa: E402
    ARM_BY_ID,
    DEFAULT_ARMS,
    LiveConfig,
    run_live_ab,
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


def _new_run_id(git: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commit = (git.get("commit") or "nogit")[:7]
    return f"{stamp}-live-ab-{commit}"


def _artifact(path: Path) -> dict:
    return {
        "path": relative_path(path, ROOT),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def _parse_arms(raw: str | None) -> tuple:
    if not raw:
        return DEFAULT_ARMS
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in ARM_BY_ID]
    if unknown:
        raise SystemExit(
            f"unknown arm(s): {', '.join(unknown)}; "
            f"choose from {', '.join(ARM_BY_ID)}"
        )
    return tuple(ARM_BY_ID[item] for item in wanted)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="provider/model id to pin")
    parser.add_argument(
        "--arms",
        default=None,
        help=f"comma-separated arms (default: {','.join(ARM_BY_ID)})",
    )
    parser.add_argument(
        "--case-ids",
        default="",
        help="comma-separated case ids (default: the full golden set)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="take the first N selected cases (0 = all)",
    )
    parser.add_argument("--reps", type=int, default=1, help="sequential repetitions")
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="seconds between cases and between agentic provider calls",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="allow cross-provider fallback (default: pin exactly --model)",
    )
    parser.add_argument(
        "--quota-note",
        default="",
        help="provider quota observed at run time (recorded in the manifest/report)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="exact output directory (default: .artifacts/bench/live-ab/<run_id>)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "LIVE_AB.md",
        help="Markdown report path (default: LIVE_AB.md)",
    )
    parser.add_argument("--no-report", action="store_true", help="skip LIVE_AB.md")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed trials in the same output directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arms = _parse_arms(args.arms)
    case_ids = tuple(
        item.strip() for item in args.case_ids.split(",") if item.strip()
    )
    git = git_info(ROOT)
    run_id = _new_run_id(git)
    out_dir = args.out_dir or (ROOT / ".artifacts" / "bench" / "live-ab" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "live_trials.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"

    config = LiveConfig(
        model_id=args.model,
        repetitions=max(1, args.reps),
        delay_s=max(0.0, args.delay),
        no_fallback=not args.allow_fallback,
        case_ids=case_ids,
        limit=args.limit,
        quota_note=args.quota_note,
    )

    started_at = utc_now()
    print(f"run_id: {run_id}")
    print(f"model: {config.model_id} (no_fallback={config.no_fallback})")
    print(f"arms: {', '.join(arm.arm_id for arm in arms)}")
    print(f"output: {out_dir}")

    records, summary = run_live_ab(
        run_id=run_id,
        config=config,
        arms=arms,
        out_dir=out_dir,
        resume=args.resume,
    )
    run_id = summary["run_id"]
    finished_at = utc_now()

    records_rel = relative_path(records_path, ROOT)
    summary["artifact_reference"] = records_rel
    write_jsonl(records_path, records)
    write_json(summary_path, summary)

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "live_ab",
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": summary["status"],
        "git": git,
        "environment": environment_info(),
        "command": command_line(),
        "config": summary["config"],
        "run_signature": summary["run_signature"],
        "counts": summary["counts"],
        "artifacts": {
            "records": {"path": records_rel},
            "summary": {"path": relative_path(summary_path, ROOT)},
            "manifest": {"path": relative_path(manifest_path, ROOT)},
        },
        "notes": [
            "Internal evidence run; raw records and answers are not published.",
            "Every arm uses the production run_analysis_sync pipeline.",
            "The model is pinned unless --allow-fallback was set.",
            "A throttled, errored, or failed trial is recorded, not retried away.",
            "The sample is curated and small; intervals are not universal claims.",
        ],
    }

    if not args.no_report:
        render_live_ab_markdown(
            summary,
            manifest,
            records,
            raw_path=records_rel,
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
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    if not args.no_report:
        print(f"report: {args.report}")
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
