"""Run the benchmark suite that supports Insight's trust claims.

Usage:
    python evaluation/bench/run_all.py --quick
    python evaluation/bench/run_all.py --full [--seed N] [--rows N]

The default is ``--quick``: one fixture per verified operation, a reduced
magnitude sweep, and two clean-data variants. ``--full`` runs every fixture and
the complete mutation matrix. Both modes write raw JSONL, a summary, a manifest,
and regenerate ``BENCHMARKS.md``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bench.adversarial_verifier import (  # noqa: E402
    config_for_mode,
    run_adversarial_verifier,
)
from evaluation.bench.report import render_markdown  # noqa: E402
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
from evaluation.bench.fixtures import DEFAULT_ROWS, DEFAULT_SEED  # noqa: E402


def _new_run_id(mode: str, git: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commit = (git.get("commit") or "nogit")[:7]
    return f"{stamp}-{mode}-{commit}"


def _status(summary: dict) -> str:
    if summary["counts"]["harness_errors"]:
        return "attention"
    if summary["clean"]["false_positive_trials"]:
        return "attention"
    if summary["clean"]["warning_control_trials"]:
        return "attention"
    if summary["clean"]["invalid_clean_controls"]:
        return "attention"
    propagation = summary["validation_propagation"]
    if propagation["detected_trials"] != propagation["validator_rejected"]:
        return "attention"
    return "ok"


def _artifact(path: Path) -> dict:
    return {
        "path": relative_path(path, ROOT),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true", help="reduced matrix (default)")
    modes.add_argument("--full", action="store_true", help="full fixture and mutation matrix")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="exact output directory (default: .artifacts/bench/<run_id>)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "BENCHMARKS.md",
        help="Markdown report path (default: BENCHMARKS.md)",
    )
    parser.add_argument("--no-report", action="store_true", help="skip BENCHMARKS.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "full" if args.full else "quick"
    config = replace(
        config_for_mode(mode),
        seed=args.seed,
        rows=args.rows,
    )

    git = git_info(ROOT)
    run_id = _new_run_id(mode, git)
    out_dir = args.out_dir or (ROOT / ".artifacts" / "bench" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "verifier_matrix.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"

    started_at = utc_now()
    print(f"run_id: {run_id}")
    print(f"mode: {mode}; seed: {config.seed}; rows: {config.rows}")
    print(f"output: {out_dir}")

    records, summary = run_adversarial_verifier(run_id, config)
    finished_at = utc_now()

    records_rel = relative_path(records_path, ROOT)
    summary["artifact_reference"] = records_rel
    write_jsonl(records_path, records)
    write_json(summary_path, summary)

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "adversarial_verifier",
        "run_id": run_id,
        "mode": mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": _status(summary),
        "git": git,
        "environment": environment_info(),
        "command": command_line(),
        "config": summary["config"],
        "counts": summary["counts"],
        "artifacts": {
            "records": {"path": records_rel},
            "summary": {"path": relative_path(summary_path, ROOT)},
            "manifest": {"path": relative_path(manifest_path, ROOT)},
        },
        "notes": [
            "Internal evidence run; raw records are not published.",
            "Git state was captured before benchmark outputs were written.",
            "Detection means an error-severity independently_verify finding on a "
            "mutated execution; it is not a proof of correctness.",
        ],
    }

    if not args.no_report:
        render_markdown(
            summary,
            manifest,
            raw_path=records_rel,
            output_path=args.report,
            root=ROOT,
        )
        manifest["artifacts"]["report"] = _artifact(args.report)

    manifest["artifacts"]["records"] = _artifact(records_path)
    manifest["artifacts"]["summary"] = _artifact(summary_path)
    write_json(manifest_path, manifest)

    summary_counts = summary["counts"]
    print(
        "records: {records}; clean: {clean}; tamper: {tamper}; detected: {detected}; "
        "missed: {missed}; warning-only: {warning_only}".format(
            records=summary_counts["records"],
            clean=summary_counts["clean_controls"],
            tamper=summary_counts["tamper_trials"],
            detected=summary_counts["detected"],
            missed=summary_counts["missed"],
            warning_only=summary_counts["warning_only"],
        )
    )
    print(f"status: {manifest['status']}")
    print(f"summary: {summary_path}")
    print(f"manifest: {manifest_path}")
    if not args.no_report:
        print(f"report: {args.report}")

    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
