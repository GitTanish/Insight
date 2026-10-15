"""A/B evaluation: agentic tool-loop vs. the single-shot planner.

Usage:
    python evaluation/run_eval_agentic.py [--model MODEL_ID] [--limit N]
                                          [--dataset sales.csv] [--delay S]

Runs the SAME golden cases (evaluation/cases.json) through the agentic tool
loop instead of the single-shot planner, and scores them with the SAME grader
(evaluation.run_eval.score_case) so the two designs are directly comparable.
The point of the harness is the side-by-side telemetry it prints — not just
pass/fail, but the *cost* of agentic behaviour:

  - llm_calls per question (agentic makes more than single-shot's ~2),
  - whether any run was throttled (Groq free-tier TPM/TPD),
  - per-case pass/fail with the identical reason strings.

Throttle-aware: --delay spaces provider calls to stay under a per-minute
ceiling, and each case's result is written to disk as it completes so a
throttled run can be resumed instead of re-spending quota.

Opt-in live harness — like run_eval.py, it needs a real provider key.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.run_eval import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_DATA,
    load_cases,
    score_case,
)


def _response_from_outcome(question: str, outcome) -> Any:
    """Rebuild an AnalysisResponse-shaped object from the loop outcome.

    score_case only reads .answer, .meta["steps"], .evidence, .tables, .error,
    so a light stand-in is enough and keeps the grader shared verbatim.
    """
    from insight.domain.visualization import AnalysisResponse, Evidence

    evidence = [
        Evidence(kind="calculation", label=str(c.label), value=c.value, detail=c.detail)
        for r in outcome.tool_results
        if r.ok
        for c in (r.calculations or [])
    ]
    tables = [r.table for r in outcome.tool_results if r.ok and r.table is not None]
    steps = [{"operation": r.name, "success": r.ok} for r in outcome.tool_results]
    error = None
    if outcome.throttled:
        error = "throttled"
    elif outcome.error:
        error = outcome.error
    elif not any(r.ok for r in outcome.tool_results):
        error = "no_successful_tool_calls"
    return AnalysisResponse(
        question=question,
        answer=outcome.final_answer,
        evidence=evidence,
        tables=tables,
        meta={"steps": steps, "agentic_llm_calls": outcome.llm_calls,
              "agentic_iterations": outcome.iterations, "throttled": outcome.throttled},
        error=error,
    )


async def _run_chain(chain, request):
    """Call the first provider in the fallback chain that succeeds."""
    last_exc: Optional[Exception] = None
    for provider, model, info in chain:
        updates = {"model": model, "temperature": 0.0}
        if info.supports_json_mode is False:
            updates["allow_server_json"] = False
        effective = request.model_copy(update=updates)
        try:
            return await provider.generate(effective)
        except Exception as exc:  # noqa: BLE001 - fall through to next provider
            last_exc = exc
            continue
    raise last_exc or RuntimeError("no provider in chain")



def run_agentic_eval(
    model_id: Optional[str] = None,
    limit: Optional[int] = None,
    dataset_filter: Optional[str] = None,
    cases_path: Path = DEFAULT_CASES,
    data_dir: Path = DEFAULT_DATA,
    delay_s: float = 0.0,
    max_iterations: int = 3,
    max_llm_calls: int = 6,
    artifacts_root: Optional[Path] = None,
    record: bool = False,
) -> tuple[int, int, Path]:
    from insight.analytics.operations import operation_catalog
    from insight.analytics.sql_engine import DuckSession
    from insight.llm.registry import build_llm_chain
    from insight.planning.prompts import AGENTIC_SYSTEM, build_agentic_messages
    from insight.profiling.profiler import profile_dataframe
    from insight.settings import get_settings
    from insight.tools.loop import LoopBudget, run_loop

    settings = get_settings()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = artifacts_root or (settings.artifacts_dir / "eval" / f"agentic_{stamp}")
    artifacts_root.mkdir(parents=True, exist_ok=True)

    cases = load_cases(cases_path)
    if dataset_filter:
        cases = [c for c in cases if c["dataset"] == dataset_filter]
    if limit:
        cases = cases[:limit]

    primary_info, chain = build_llm_chain(model_id, settings)
    model_name = primary_info.model

    frames: dict[str, tuple[pd.DataFrame, Any]] = {}
    results: list[dict] = []

    transcript = (artifacts_root / "transcript.jsonl") if record else None

    for case in cases:
        name = case["dataset"]
        if name not in frames:
            df = pd.read_csv(data_dir / name)
            profile = profile_dataframe(df, name, content_hash=f"eval{name}{len(df)}"[:32])
            frames[name] = (df, profile)
        df, profile = frames[name]

        print(f"[{case['id']}] agentic ask: {case['question'][:64]}...")

        usage = {"seq": 0, "prompt_tokens": 0, "completion_tokens": 0, "models": set()}

        async def _generate(request, _chain=chain):
            response = await _run_chain(_chain, request)
            usage["seq"] += 1
            usage["prompt_tokens"] += response.prompt_tokens or 0
            usage["completion_tokens"] += response.completion_tokens or 0
            if response.model:
                usage["models"].add(response.model)
            if transcript is not None:
                _append_transcript(transcript, case["id"], usage["seq"], request, response)
            return response

        budget = LoopBudget(
            max_iterations=max_iterations,
            max_llm_calls=max_llm_calls,
            call_delay_s=delay_s,
        )
        # Mirror the production wiring: the agent sees the same system prompt,
        # operation catalog and DuckDB schema the app gives it, and SQL tool
        # calls run against a real read-only session instead of failing on a
        # missing context.
        table_datasets = {profile.name: df}
        session = DuckSession(table_datasets) if settings.sql_enabled else None
        initial_messages = build_agentic_messages(
            case["question"],
            profile,
            operation_catalog(),
            sql_schema=session.describe() if session is not None else None,
        )
        started = time.monotonic()
        try:
            outcome = asyncio.run(
                run_loop(
                    case["question"], df, profile, _generate, model_name,
                    duck_ctx=session,
                    budget=budget, system_prompt=AGENTIC_SYSTEM,
                    initial_messages=initial_messages,
                )
            )
        finally:
            if session is not None:
                session.close()
        wall_s = round(time.monotonic() - started, 2)
        response = _response_from_outcome(case["question"], outcome)
        passed, reasons = score_case(response, case)
        record = {
            **case,
            "passed": passed,
            "reasons": reasons,
            "llm_calls": outcome.llm_calls,
            "iterations": outcome.iterations,
            "operations_used": outcome.operations_used,
            "throttled": outcome.throttled,
            "error": outcome.error,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "models_used": sorted(usage["models"]),
            "wall_s": wall_s,
        }
        results.append(record)
        status = "PASS" if passed else "FAIL"
        extra = " [THROTTLED]" if outcome.throttled else ""
        err = f" error={outcome.error[:80]}" if outcome.error else ""
        print(f"[{case['id']}] {status}{extra} calls={outcome.llm_calls} "
              f"ops={outcome.operations_used}{err} {reasons if reasons else ''}")
        _write_incremental(artifacts_root, results, model_id)

    passed_count = sum(1 for r in results if r["passed"])
    report = _write_report(results, passed_count, len(results), model_id, artifacts_root)
    total_calls = sum(r.get("llm_calls", 0) for r in results)
    throttled = sum(1 for r in results if r.get("throttled"))
    print(f"\n{passed_count}/{len(results)} passed | total llm_calls={total_calls} "
          f"| throttled={throttled}")
    print(f"report -> {report}")
    return passed_count, len(results), report


def _append_transcript(path: Path, case_id: str, seq: int, request, response) -> None:
    """Append one raw provider exchange to the record/replay transcript."""
    line = {
        "case": case_id,
        "seq": seq,
        "request": request.model_dump(),
        "response": response.model_dump(),
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")


def _write_incremental(root: Path, results: list[dict], model_id: Optional[str]) -> None:
    (root / "results.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def _write_report(results, passed, total, model_id, artifacts_root) -> Path:
    lines = [
        "# Insight agentic eval report",
        "",
        f"- when: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- model: `{model_id or '(default)'}`",
        f"- result: **{passed}/{total} passed**",
        f"- total llm_calls: {sum(r.get('llm_calls', 0) for r in results)}",
        f"- total tokens: in={sum(r.get('prompt_tokens', 0) for r in results)} "
        f"out={sum(r.get('completion_tokens', 0) for r in results)}",
        "",
        "| id | dataset | status | llm_calls | tokens in/out | wall_s | ops | notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        if r.get("throttled"):
            status = "THROTTLED"
        notes = "; ".join(r.get("reasons", [])) or "—"
        lines.append(
            f"| {r['id']} | {r['dataset']} | {status} | {r.get('llm_calls')} "
            f"| {r.get('prompt_tokens', '?')}/{r.get('completion_tokens', '?')} "
            f"| {r.get('wall_s', '?')} "
            f"| {','.join(r.get('operations_used', []))} | {notes} |"
        )
    path = artifacts_root / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds between provider calls (throttle safety)")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=6)
    parser.add_argument("--record", action="store_true",
                        help="also write every raw provider exchange to transcript.jsonl")
    args = parser.parse_args()
    passed, total, _ = run_agentic_eval(
        model_id=args.model, limit=args.limit, dataset_filter=args.dataset,
        delay_s=args.delay, max_iterations=args.max_iterations,
        max_llm_calls=args.max_llm_calls, record=args.record,
    )
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
