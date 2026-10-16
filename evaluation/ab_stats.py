"""Aggregate A/B evaluation runs (k repetitions per arm) into one summary.

Usage:
    python evaluation/ab_stats.py --agentic DIR1,DIR2,DIR3
                                 [--baseline DIR4,DIR5,DIR6]
                                 [--out stats.json]

Each DIR must be an artifacts directory written by run_eval.py or
run_eval_agentic.py (contains results.json). Dirs listed for one arm are
repetitions of the same configuration; pass rates are aggregated per case
across repetitions, overall pass rate gets a 95% Wilson confidence interval,
and cost telemetry (llm_calls / tokens / wall time) is averaged.

The two runner schemas differ slightly (single-shot records total_ms and
model_used, agentic records wall_s and models_used); both are read
tolerantly so the same script aggregates either arm.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ARM_KEYS = ("agentic", "baseline")


def wilson_ci(passes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = passes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _load_dirs(spec: str) -> list[tuple[Path, list[dict]]]:
    runs: list[tuple[Path, list[dict]]] = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        path = Path(raw)
        results_path = path if path.name == "results.json" else path / "results.json"
        if not results_path.exists():
            raise SystemExit(f"missing results.json under {path}")
        runs.append((path, json.loads(results_path.read_text(encoding="utf-8"))))
    if not runs:
        raise SystemExit("no directories given")
    return runs


def _mean(values: list[float]) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 2) if nums else None


def _wall_s(row: dict) -> float | None:
    if isinstance(row.get("wall_s"), (int, float)):
        return float(row["wall_s"])
    if isinstance(row.get("total_ms"), (int, float)):
        return float(row["total_ms"]) / 1000.0
    return None


def aggregate(runs: list[tuple[Path, list[dict]]]) -> dict:
    per_case: dict[str, dict] = {}
    models: set[str] = set()
    for _, rows in runs:
        for row in rows:
            entry = per_case.setdefault(row["id"], {"passes": 0, "trials": 0})
            entry["passes"] += 1 if row.get("passed") else 0
            entry["trials"] += 1
            models.update(row.get("models_used") or [])
            if row.get("model_used"):
                models.add(row["model_used"])

    all_rows = [row for _, rows in runs for row in rows]
    passes = sum(1 for r in all_rows if r.get("passed"))
    trials = len(all_rows)
    low, high = wilson_ci(passes, trials)
    return {
        "runs": len(runs),
        "cases": len(per_case),
        "trials": trials,
        "passes": passes,
        "pass_rate": round(passes / trials, 3) if trials else 0.0,
        "wilson95": [round(low, 3), round(high, 3)],
        "mean_llm_calls": _mean([r.get("llm_calls") for r in all_rows]),
        "total_prompt_tokens": sum(
            r.get("prompt_tokens") or 0 for r in all_rows
        ),
        "total_completion_tokens": sum(
            r.get("completion_tokens") or 0 for r in all_rows
        ),
        "mean_wall_s": _mean([_wall_s(r) for r in all_rows]),
        "throttled_cases": sum(1 for r in all_rows if r.get("throttled")),
        "models_used": sorted(models),
        "per_case": {
            cid: f"{e['passes']}/{e['trials']}" for cid, e in sorted(per_case.items())
        },
        "failure_reasons": sorted({
            reason
            for r in all_rows if not r.get("passed")
            for reason in (r.get("reasons") or [])
        }),
    }


def _print_arm(name: str, agg: dict) -> None:
    print(f"\n== {name} ==")
    print(f"runs={agg['runs']} cases={agg['cases']} trials={agg['trials']}")
    print(f"pass: {agg['passes']}/{agg['trials']} = {agg['pass_rate']:.1%} "
          f"(95% Wilson {agg['wilson95'][0]:.1%}..{agg['wilson95'][1]:.1%})")
    print(f"mean llm_calls={agg['mean_llm_calls']} mean wall={agg['mean_wall_s']}s "
          f"tokens={agg['total_prompt_tokens']}+{agg['total_completion_tokens']} "
          f"throttled={agg['throttled_cases']}")
    print(f"models used: {', '.join(agg['models_used']) or '(unknown)'}")
    print("per-case (passes/trials):")
    for cid, frac in agg["per_case"].items():
        print(f"  {cid:40s} {frac}")
    if agg["failure_reasons"]:
        print("failure reasons:")
        for reason in agg["failure_reasons"]:
            print(f"  - {reason[:160]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARM_KEYS:
        parser.add_argument(
            f"--{arm}", default=None,
            help="comma-separated artifacts dirs (one per repetition)",
        )
    parser.add_argument("--out", default=None, help="write aggregated JSON here")
    args = parser.parse_args()

    provided = {arm: getattr(args, arm.replace("-", "_")) for arm in ARM_KEYS}
    if not any(provided.values()):
        raise SystemExit("give at least one of --agentic / --baseline")

    summary: dict[str, dict] = {}
    for arm, spec in provided.items():
        if not spec:
            continue
        agg = aggregate(_load_dirs(spec))
        summary[arm] = agg
        _print_arm(arm, agg)

    if "agentic" in summary and "baseline" in summary:
        delta = summary["agentic"]["pass_rate"] - summary["baseline"]["pass_rate"]
        print(f"\n== comparison ==\nagentic - baseline pass rate: {delta:+.1%}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"\nstats -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
