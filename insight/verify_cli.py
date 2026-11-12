"""Command-line entry point for standalone claim verification.

Examples::

    # Evidence supplied directly (no execution)
    python -m insight.verify_cli --answer "Revenue was 77." \
        --evidence evidence.json

    # Run an operation and verify the narrative against its evidence
    python -m insight.verify_cli --answer "North leads with 150." \
        --data sales.csv --operation group_aggregate \
        --params '{"group_by": "region", "metrics": [...]}'

    # Verify a text-to-SQL narrative against the query result
    python -m insight.verify_cli --answer-file draft.md \
        --data sales.csv --sql "SELECT region, SUM(revenue) AS total ..."

    # Verify a signed certificate (HMAC-SHA256)
    python -m insight.verify_cli --verify-certificate certificate.json \
        --signing-key-file signing-key.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from insight.toolkit import AnalysisToolkit, ToolkitError
from insight.verification.signing import verify_certificate_signature


def _load_answer(args: argparse.Namespace) -> str:
    if args.answer_file:
        return Path(args.answer_file).read_text(encoding="utf-8")
    if args.answer is not None:
        return args.answer
    if not sys.stdin.isatty():
        return sys.stdin.read()
    raise ValueError("provide --answer, --answer-file, or pipe text on stdin")


def _load_evidence(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence JSON must be an object")
    return payload


def _load_signing_key(path: Optional[str]) -> str:
    if path:
        key = Path(path).read_text(encoding="utf-8").strip()
    else:
        key = (os.getenv("INSIGHT_SIGNING_KEY") or "").strip()
    if not key:
        raise ValueError("provide --signing-key-file or set INSIGHT_SIGNING_KEY")
    return key


def _evidence_args(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_numbers": payload.get("numbers")
        or payload.get("evidence_numbers")
        or [],
        "evidence_tables": payload.get("tables")
        or payload.get("evidence_tables")
        or [],
        "evidence_notes": payload.get("notes") or payload.get("evidence_notes"),
    }


def _load_tables(toolkit: AnalysisToolkit, paths: list[str], name: Optional[str]) -> list[str]:
    if name and len(paths) != 1:
        raise ValueError("--name can only be used with a single --data file")
    names: list[str] = []
    for path in paths:
        table_name = name or Path(path).stem
        toolkit.load_table(table_name, path=path)
        names.append(table_name)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="insight-verify",
        description=(
            "Verify numbers in a narrative against computed evidence and emit "
            "a versioned verification certificate."
        ),
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--answer", help="Narrative text to verify")
    source.add_argument("--answer-file", help="Path to a narrative file")

    parser.add_argument(
        "--data",
        action="append",
        default=[],
        help="CSV/Excel file to load (repeatable; table name defaults to file stem)",
    )
    parser.add_argument("--name", help="Table name for a single --data file")
    parser.add_argument(
        "--operation",
        help="Registered operation to run on the first loaded table",
    )
    parser.add_argument(
        "--params",
        default="{}",
        help="JSON object of operation parameters (default: {})",
    )
    parser.add_argument(
        "--params-file",
        help="Path to a JSON object of operation parameters",
    )
    parser.add_argument(
        "--sql",
        help="Read-only SELECT/WITH query to run against the loaded tables",
    )
    parser.add_argument(
        "--evidence",
        help="JSON file with numbers/tables/notes for evidence-only verification",
    )
    parser.add_argument(
        "--verify-certificate",
        help="Verify the signature on an existing certificate JSON file",
    )
    parser.add_argument(
        "--signing-key-file",
        help="Read the HMAC signing key from this file (or set INSIGHT_SIGNING_KEY)",
    )
    parser.add_argument(
        "--out",
        help="Write the certificate to this path instead of stdout",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Emit compact JSON instead of indented JSON",
    )
    parser.add_argument(
        "--fail-on-warn",
        action="store_true",
        help="Exit non-zero on pass_with_warnings as well as fail",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verify_certificate:
        try:
            if (
                args.answer
                or args.answer_file
                or args.data
                or args.evidence
                or args.operation
                or args.sql
            ):
                raise ValueError(
                    "--verify-certificate cannot be combined with answer, data, "
                    "operation, SQL, or evidence inputs"
                )
            certificate = json.loads(
                Path(args.verify_certificate).read_text(encoding="utf-8")
            )
            if not isinstance(certificate, dict):
                raise ValueError("certificate JSON must be an object")
            key = _load_signing_key(args.signing_key_file)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"error": str(exc)}), file=sys.stderr)
            return 2

        valid = verify_certificate_signature(certificate, key)
        print(
            json.dumps(
                {
                    "valid": valid,
                    "certificate_id": certificate.get("certificate_id"),
                },
                indent=None if args.compact else 2,
                ensure_ascii=False,
            )
        )
        return 0 if valid else 1

    try:
        answer = _load_answer(args)
        toolkit = AnalysisToolkit()
        evidence = _evidence_args(_load_evidence(args.evidence)) if args.evidence else {}

        if args.data:
            table_names = _load_tables(toolkit, args.data, args.name)
            if args.sql and args.operation:
                raise ValueError("use either --sql or --operation, not both")
            if args.sql:
                toolkit.run_sql(args.sql, tables=table_names)
            elif args.operation:
                if args.params_file:
                    if args.params != "{}":
                        raise ValueError("use either --params or --params-file")
                    params = json.loads(
                        Path(args.params_file).read_text(encoding="utf-8")
                    )
                else:
                    params = json.loads(args.params or "{}")
                if not isinstance(params, dict):
                    raise ValueError("operation parameters must be a JSON object")
                toolkit.execute_operation(table_names[0], args.operation, params)
            else:
                raise ValueError("--data requires --operation or --sql")
        elif not args.evidence:
            raise ValueError("provide --evidence or --data with --operation/--sql")

        result = toolkit.verify_claims(answer, **evidence)
    except (ToolkitError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2

    certificate = result["certificate"]
    text = json.dumps(
        certificate,
        indent=None if args.compact else 2,
        ensure_ascii=False,
    )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    verdict = certificate.get("verdict", {}).get("overall")
    if verdict == "fail":
        return 1
    if args.fail_on_warn and verdict == "pass_with_warnings":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
