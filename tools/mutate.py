"""Lightweight mutation sampler for the agentic tool layer.

mutmut does not support native Windows (WSL only, see requirements-dev.txt), so
this is the substitute: apply curated single-site mutations, run a scoped test
selection after each, restore the file byte-for-byte, and report killed vs
survived. The mutations are deliberately aimed at the claims the docs make
(budget caps, repeated-failure abort, throttle classification, frame chaining,
input_step validation, HITL precedence) - this is *sampled* mutation testing
to show the targeted tests have teeth, not an exhaustive coverage score.

Usage:
    python tools/mutate.py            # all mutations
    python tools/mutate.py --list     # show mutations without running
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MUTATIONS: list[dict] = [
    {
        "label": "llm call budget off-by-one (>= becomes >)",
        "file": "insight/tools/loop.py",
        "old": "if outcome.llm_calls >= budget.max_llm_calls:",
        "new": "if outcome.llm_calls > budget.max_llm_calls:",
        "tests": [
            "tests/test_agentic_tools.py",
            "tests/test_agentic_adversarial.py",
        ],
    },
    {
        "label": "repeated-failure abort fires one call early (>= 3 becomes >= 2)",
        "file": "insight/tools/loop.py",
        "old": "if len(recent_failures) >= 3 and len(set(recent_failures[-3:])) == 1:",
        "new": "if len(recent_failures) >= 2 and len(set(recent_failures[-2:])) == 1:",
        "tests": [
            "tests/test_agentic_tools.py",
            "tests/test_agentic_pipeline.py",
            "tests/test_agentic_adversarial.py",
        ],
    },
    {
        "label": "repeated-failure abort never fires (== 1 becomes == 0)",
        "file": "insight/tools/loop.py",
        "old": "if len(recent_failures) >= 3 and len(set(recent_failures[-3:])) == 1:",
        "new": "if len(recent_failures) >= 3 and len(set(recent_failures[-3:])) == 0:",
        "tests": [
            "tests/test_agentic_tools.py",
            "tests/test_agentic_pipeline.py",
            "tests/test_agentic_adversarial.py",
        ],
    },
    {
        "label": "throttle classification disabled (_is_rate_limit returns False)",
        "file": "insight/tools/loop.py",
        "old": '    text = str(exc).lower()\n    return "429" in text or "rate limit" in text or "rate_limit" in text or "tpm" in text',
        "new": "    return False",
        "tests": [
            "tests/test_agentic_tools.py",
            "tests/test_agentic_adversarial.py",
        ],
    },
    {
        "label": "working-frame update dropped (filter_rows no longer narrows later calls)",
        "file": "insight/tools/loop.py",
        "old": "                    if spec is not None and spec.mutates and input_step is None:\n                        working = result.frame",
        "new": "                    if spec is not None and spec.mutates and input_step is None:\n                        working = working",
        "tests": [
            "tests/test_agentic_pipeline.py",
        ],
    },
    {
        "label": "missing input_step target accepted (guard condition disabled)",
        "file": "insight/tools/loop.py",
        "old": "if input_step is not None and input_step not in state.frames:",
        "new": "if input_step is not None and input_step not in state.frames and False:",
        "tests": [
            "tests/test_agentic_pipeline.py",
            "tests/test_agentic_tools.py",
        ],
    },
    {
        "label": "inbound tool args string parsing disabled",
        "file": "insight/llm/providers/openai_compat.py",
        "old": "if isinstance(args_raw, str):",
        "new": "if False and isinstance(args_raw, str):",
        "tests": [
            "tests/test_agentic_pipeline.py",
        ],
    },
    {
        "label": "HITL precedence dropped (settings.plan_approval ignored)",
        "file": "insight/orchestrator.py",
        "old": "if request.plan_only or settings.plan_approval:",
        "new": "if request.plan_only:",
        "tests": [
            "tests/test_agentic_pipeline.py",
            "tests/test_hitl_plan_approval.py",
        ],
    },
]


def run(tests: list[str]) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *tests, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list mutations only")
    args = parser.parse_args()

    if args.list:
        for m in MUTATIONS:
            print(f"- {m['label']}  [{m['file']}]")
        return 0

    killed = survived = 0
    for m in MUTATIONS:
        path = ROOT / m["file"]
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(m["old"])
        if occurrences == 0:
            print(f"[SKIP] {m['label']}: pattern not found in {m['file']}")
            continue
        try:
            path.write_text(original.replace(m["old"], m["new"]), encoding="utf-8")
            rc = run(m["tests"])
        finally:
            path.write_text(original, encoding="utf-8")
        sites = f" ({occurrences} sites)" if occurrences > 1 else ""
        if rc != 0:
            killed += 1
            print(f"[KILLED ] {m['label']}{sites}")
        else:
            survived += 1
            print(f"[SURVIVED] {m['label']}{sites}  <-- tests did not detect this")

    print(f"\nsampled mutations: {killed} killed / {killed + survived} "
          f"({survived} survived)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
