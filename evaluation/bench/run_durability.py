"""Run the durability and fault-injection suite.

Usage:
    python evaluation/bench/run_durability.py --quick
    python evaluation/bench/run_durability.py --full

Both modes write raw JSONL, a summary, a manifest, and regenerate
``DURABILITY.md``. ``--quick`` reduces ledger record counts, concurrency
workers, and cache race iterations; ``--full`` uses the harness defaults.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.bench.durability import (  # noqa: E402
    config_for_mode,
    run_durability,
)
from evaluation.bench.durability_report import (  # noqa: E402
    render_durability_markdown,
)
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


def _new_run_id(mode: str, git: dict) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    commit = (git.get("commit") or "nogit")[:7]
    return f"{stamp}-durability-{mode}-{commit}"


def _artifact(path: Path) -> dict:
    return {
        "path": relative_path(path, ROOT),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--quick", action="store_true", help="reduced concurrency and record counts (default)"
    )
    modes.add_argument(
        "--full", action="store_true", help="use the harness's full record counts"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="exact output directory (default: .artifacts/bench/durability/<run_id>)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "DURABILITY.md",
        help="Markdown report path (default: DURABILITY.md)",
    )
    parser.add_argument("--no-report", action="store_true", help="skip DURABILITY.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mode = "full" if args.full else "quick"
    config = config_for_mode(mode)

    git = git_info(ROOT)
    run_id = _new_run_id(mode, git)
    out_dir = args.out_dir or (ROOT / ".artifacts" / "bench" / "durability" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "durability_trials.jsonl"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "manifest.json"

    started_at = utc_now()
    print(f"run_id: {run_id}")
    print(f"mode: {mode}")
    print(f"output: {out_dir}")

    records, summary = run_durability(run_id, out_dir, config)
    finished_at = utc_now()

    records_rel = relative_path(records_path, ROOT)
    summary["artifact_reference"] = records_rel
    write_jsonl(records_path, records)
    write_json(summary_path, summary)

    manifest: dict = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "durability",
        "run_id": run_id,
        "mode": mode,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": summary["status"],
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
            "A failed scored trial is recorded as a gap; the harness does not "
            "retry failures away.",
            "Known-boundary trials are recorded but excluded from scored claims.",
        ],
    }

    if not args.no_report:
        render_durability_markdown(
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
        "records: {records}; scored: {scored}; passed: {passed}; failed: {failed}; "
        "boundaries: {boundaries}; harness errors: {errors}".format(
            records=counts["records"],
            scored=counts["scored"],
            passed=counts["passed"],
            failed=counts["failed"],
            boundaries=counts["known_boundaries"],
            errors=counts["harness_errors"],
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
