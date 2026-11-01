"""Subprocess worker used by the durability harness.

Each worker appends a fixed number of probe records to the ledger selected by
``INSIGHT_AUDIT_LEDGER_PATH``. Running several of these concurrently exercises
the process lock and OS file lock in :mod:`insight.audit`; the parent verifies
the resulting chain.

Usage (normally launched by ``evaluation/bench/durability.py``):
    python -m evaluation.bench.ledger_worker --count 10 --worker w0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.audit import append_event  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--worker", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seqs: list[int] = []
    for index in range(args.count):
        record = append_event(
            {
                "event": "durability_probe",
                "worker": args.worker,
                "index": index,
            }
        )
        if record is None:
            print("audit ledger is disabled", file=sys.stderr)
            return 2
        seqs.append(int(record["seq"]))

    print(
        json.dumps(
            {
                "worker": args.worker,
                "count": args.count,
                "seqs": seqs,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
