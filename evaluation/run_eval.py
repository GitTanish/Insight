"""Golden-question evaluation runner for Insight.

Usage:
    python evaluation/run_eval.py [--model MODEL_ID] [--limit N] [--dataset sales.csv]
                                  [--cases PATH] [--report PATH]

Runs each case through run_analysis_sync (cache bypassed) and scores:
  - ops superset match against successful plan steps
  - substring facts in the answer
  - numeric proximity (+/- tol_pct, default 0.5%) against calculation labels,
    wide-table columns, or long-format summarize tables

Writes a markdown report under .artifacts/eval/ and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CASES = Path(__file__).resolve().parent / "cases.json"
DEFAULT_DATA = Path(__file__).resolve().parent / "data"


def load_cases(path: Path) -> list[dict]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    required = {"id", "dataset", "question"}
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"case missing keys {missing}: {case}")
    return cases


def _ops_satisfied(case_ops: list[Any], executed: list[str]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for token in case_ops:
        if isinstance(token, list):
            if not any(op in executed for op in token):
                failures.append(f"expected any of {token}; executed={sorted(set(executed))}")
        elif token not in executed:
            failures.append(f"missing operation '{token}'; executed={sorted(set(executed))}")
    return not failures, failures


def _contains_satisfied(needles: list[str], answer: str) -> tuple[bool, list[str]]:
    haystack = answer.lower()
    missing = [n for n in needles if n.lower() not in haystack]
    return not missing, [f"answer missing fact {n!r}" for n in missing]


def _numeric_close(value: Any, approx: float, tol_pct: float) -> bool:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return False
    tolerance = max(abs(approx) * tol_pct / 100.0, 1e-9)
    return abs(number - approx) <= tolerance


def _numeric_found(response, spec: dict) -> tuple[bool, Optional[str]]:
    label = spec["label"]
    labels = [str(x).lower() for x in (label if isinstance(label, list) else [label])]
    column = str(spec.get("column") or "").lower()
    tol_pct = float(spec.get("tol_pct", 0.5))
    approx = float(spec["approx"])

    for item in response.evidence:
        if item.kind == "calculation" and str(item.label).lower() in labels:
            if _numeric_close(item.value, approx, tol_pct):
                return True, None
    for table in response.tables:
        lowered = [str(c).lower() for c in table.columns]
        for wanted in labels:
            if wanted in lowered:
                idx = lowered.index(wanted)
                if any(_numeric_close(row[idx], approx, tol_pct) for row in table.rows):
                    return True, None
        if lowered == ["column", "metric", "value"]:
            for row in table.rows:
                metric_name = str(row[1]).lower()
                row_column = str(row[0]).lower()
                if metric_name in labels and (not column or row_column == column):
                    if _numeric_close(row[2], approx, tol_pct):
                        return True, None
    expected = f"{spec['approx']} (±{tol_pct}% as {labels})"
    return False, f"numeric expectation not met: {expected}"


def score_case(response, case: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if response is None:
        return False, ["no response produced"]
    if response.error:
        return False, [f"pipeline error: {response.error}"]

    executed = [
        s.get("operation") for s in response.meta.get("steps", []) if s.get("success")
    ]
    ok, fails = _ops_satisfied(case.get("expect_ops", []), executed)
    reasons.extend(fails)

    ok2, fails2 = _contains_satisfied(case.get("answer_contains", []), response.answer)
    reasons.extend(fails2)

    for spec in case.get("numeric_expect", []):
        found, fail = _numeric_found(response, spec)
        if not found:
            reasons.append(fail or "numeric expectation failed")

    return not reasons, reasons


def run_eval(
    model_id: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_filter: Optional[str] = None,
    cases_path: Path = DEFAULT_CASES,
    data_dir: Path = DEFAULT_DATA,
    artifacts_root: Optional[Path] = None,
) -> tuple[int, int, Path]:
    from insight.domain.query import AnalysisRequest
    from insight.orchestrator import run_analysis_sync
    from insight.profiling.profiler import profile_dataframe
    from insight.settings import get_settings

    settings = get_settings()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = artifacts_root or (settings.artifacts_dir / "eval" / f"run_{stamp}")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    if dataset_filter:
        cases = [c for c in cases if c["dataset"] == dataset_filter]
    if limit:
        cases = cases[:limit]

    frames: dict[str, tuple[pd.DataFrame, Any]] = {}
    results: list[dict] = []

    for case in cases:
        name = case["dataset"]
        if name not in frames:
            df = pd.read_csv(data_dir / name)
            profile = profile_dataframe(
                df, name, content_hash=f"eval{name}{len(df)}"[:32]
            )
            frames[name] = (df, profile)

        df, profile = frames[name]
        request = AnalysisRequest(question=case["question"], model_id=model_id)
        print(f"[{case['id']}] asking: {case['question'][:70]}...")
        try:
            response = run_analysis_sync(
                df,
                profile,
                request,
                artifacts_dir=artifacts_root / name.replace(".csv", ""),
                use_cache=False,
            )
        except Exception as exc:
            response = None
            results.append({
                **case,
                "passed": False,
                "reasons": [f"exception: {type(exc).__name__}: {exc}"[:300]],
                "model_used": None,
            })
            print(f"[{case['id']}] EXCEPTION {exc}")
            continue

        passed, reasons = score_case(response, case)
        results.append({
            **case,
            "passed": passed,
            "reasons": reasons,
            "model_used": response.meta.get("model"),
            "total_ms": response.meta.get("total_ms"),
        })
        print(f"[{case['id']}] {'PASS' if passed else 'FAIL'} {reasons if reasons else ''}")

    passed_count = sum(1 for r in results if r["passed"])
    report_path = write_report(results, passed_count, len(results), model_id, artifacts_root)
    return passed_count, len(results), report_path


def write_report(
    results: list[dict],
    passed: int,
    total: int,
    model_id: Optional[str],
    artifacts_root: Path,
) -> Path:
    lines = [
        "# Insight eval report",
        "",
        f"- when: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- model: `{model_id or '(default)'}`",
        f"- result: **{passed}/{total} passed**",
        "",
        "| id | dataset | status | notes |",
        "|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        notes = "; ".join(r.get("reasons", [])) or "—"
        lines.append(f"| {r['id']} | {r['dataset']} | {status} | {notes} |")

    failed = [r for r in results if not r["passed"]]
    if failed:
        lines += ["", "## Failure detail", ""]
        for r in failed:
            lines.append(f"### {r['id']} — {r['question']}")
            lines.append("")
            for reason in r.get("reasons", []):
                lines.append(f"- {reason}")
            lines.append("")

    report_path = artifacts_root / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport → {report_path}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="provider/model id, e.g. groq/openai/gpt-oss-120b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", default=None, help="filter to one dataset filename")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA))
    args = parser.parse_args()

    passed, total, _ = run_eval(
        model_id=args.model,
        limit=args.limit,
        dataset_filter=args.dataset,
        cases_path=Path(args.cases),
        data_dir=Path(args.data_dir),
    )
    print(f"\n{passed}/{total} cases passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
