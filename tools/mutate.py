"""Lightweight mutation sampler for trust-critical code paths.

mutmut does not support native Windows (WSL only, see requirements-dev.txt), so
this is the substitute: apply curated single-site mutations, run a scoped test
selection after each, restore the file byte-for-byte, and report killed vs
survived. The mutations are deliberately aimed at the claims the docs make
(budget caps, repeated-failure abort, throttle classification, frame chaining,
input_step validation, reasoning-effort stripping, the truncation hint, HITL
precedence) and at the trust core (independent recomputation, final-validator
gates, grounding, executor failure propagation) - this is *sampled* mutation
testing to show the targeted tests have teeth, not an exhaustive coverage
score.

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
        "label": "harmony-token sanitization of inbound tool names disabled",
        "file": "insight/llm/providers/openai_compat.py",
        "old": "name = name.split(\"<|\", 1)[0].strip()",
        "new": "name = name",
        "tests": [
            "tests/test_agentic_pipeline.py",
        ],
    },
    {
        "label": "reasoning_effort stripping disabled (unsupported models would get the param)",
        "file": "insight/orchestrator.py",
        "old": "if info.supports_reasoning_effort is False:",
        "new": "if False:",
        "tests": [
            "tests/test_chain_fallback.py",
        ],
    },
    {
        "label": "truncation hint dropped from the empty-content error",
        "file": "insight/orchestrator.py",
        "old": "and response.completion_tokens == effective.max_tokens",
        "new": "and False",
        "tests": [
            "tests/test_chain_fallback.py",
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
    {
        "label": "verifier numeric closeness disabled (all finite values accepted)",
        "file": "insight/validation/verification.py",
        "old": "    return abs(a_f - b_f) / abs(b_f) <= _TOLERANCE",
        "new": "    return True",
        "tests": ["tests/test_verification.py"],
    },
    {
        "label": "verifier group-aggregate row-count check disabled",
        "file": "insight/validation/verification.py",
        "old": """    if len(table.rows) > len(expected) or (
        not table.truncated and len(table.rows) < len(expected)
    ):
        issues.append(
            _recompute_mismatch(
                step,
                f"group_aggregate returned {len(table.rows)} rows, expected "
                f"{len(expected)}",
            )
        )""",
        "new": """    if False:
        issues.append(
            _recompute_mismatch(
                step,
                f"group_aggregate returned {len(table.rows)} rows, expected "
                f"{len(expected)}",
            )
        )""",
        "tests": ["tests/test_verification.py"],
    },
    {
        "label": "verifier correlation cell recomputation disabled",
        "file": "insight/validation/verification.py",
        "old": """            got_value = row[table.columns.index(column_label)]
            if not _table_value_close(got_value, expected_value):
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"correlation [{row_label}][{column_label}] reported \"""",
        "new": """            got_value = row[table.columns.index(column_label)]
            if False:
                issues.append(
                    _recompute_mismatch(
                        step,
                        f"correlation [{row_label}][{column_label}] reported \"""",
        "tests": ["tests/test_verification.py"],
    },
    {
        "label": "verifier top-N output-schema check disabled",
        "file": "insight/validation/verification.py",
        "old": """    for name in expected_columns:
        if name not in table.columns:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"top_n output column '{name}' missing from result",""",
        "new": """    for name in expected_columns:
        if False:
            issues.append(
                _recompute_mismatch(
                    step,
                    f"top_n output column '{name}' missing from result",""",
        "tests": ["tests/test_verification.py"],
    },
    {
        "label": "verifier strongest_correlation recomputation disabled",
        "file": "insight/validation/verification.py",
        "old": '        if "strongest_correlation" in calcs and not _rounded_close(',
        "new": '        if False and "strongest_correlation" in calcs and not _rounded_close(',
        "tests": ["tests/test_verification.py"],
    },
    {
        "label": "validator non-finite scan disabled",
        "file": "insight/validation/result_validator.py",
        "old": "            if math.isnan(as_float) or math.isinf(as_float):",
        "new": "            if False:",
        "tests": ["tests/test_validator.py"],
    },
    {
        "label": "validator failed-step error suppressed",
        "file": "insight/validation/result_validator.py",
        "old": "        if not result.success:",
        "new": "        if False:",
        "tests": ["tests/test_validator.py"],
    },
    {
        "label": "validator chart-source missing-table guard disabled",
        "file": "insight/validation/result_validator.py",
        "old": "        if source_table is None:",
        "new": "        if False:",
        "tests": ["tests/test_validator.py"],
    },
    {
        "label": "grounding accepts every number (evidence lookup disabled)",
        "file": "insight/validation/grounding.py",
        "old": "        if round(number, 6) in floats:",
        "new": "        if True:",
        "tests": ["tests/test_invariants.py"],
    },
    {
        "label": "grounding percentage rescaling disabled",
        "file": "insight/validation/grounding.py",
        "old": '        if suffix.startswith("%") and any(',
        "new": "        if False and any(",
        "tests": ["tests/test_invariants.py"],
    },
    {
        "label": "executor overall success ignores failed steps",
        "file": "insight/execution/executor.py",
        "old": "        success=bool(step_results) and all(s.success for s in step_results),",
        "new": "        success=True,",
        "tests": ["tests/test_executor.py"],
    },
    {
        "label": "executor missing input_step guard disabled",
        "file": "insight/execution/executor.py",
        "old": "            if source is None:",
        "new": "            if False:",
        "tests": ["tests/test_chaining.py"],
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
        raw = path.read_bytes()
        original = raw.decode("utf-8")
        text = original.replace("\r\n", "\n")
        occurrences = text.count(m["old"])
        if occurrences == 0:
            print(f"[SKIP] {m['label']}: pattern not found in {m['file']}")
            continue
        try:
            path.write_bytes(
                text.replace(m["old"], m["new"]).encode("utf-8")
            )
            rc = run(m["tests"])
        finally:
            path.write_bytes(raw)
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
