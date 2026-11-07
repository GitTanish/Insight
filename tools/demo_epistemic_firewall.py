"""Demo: a deterministic firewall for numbers in an agent's narrative.

Run with::

    python tools/demo_epistemic_firewall.py

The demo computes a real group-by aggregate, then verifies three narratives:
one correct, one with a fabricated number, and one that makes a causal claim
from an observational result. Each certificate is written to
``.artifacts/epistemic_firewall_demo/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from insight.toolkit import AnalysisToolkit

SALES_CSV = (
    "region,revenue,units\n"
    "North,100,2\n"
    "South,40,1\n"
    "East,60,3\n"
    "North,50,2\n"
    "West,30,1\n"
)

CASES = (
    ("correct", "North leads with 150 in revenue across 5 rows."),
    ("fabricated", "North leads with 999 in revenue across 5 rows."),
    ("causal_overreach", "North's 150 in revenue drove the regional lead."),
)


def main() -> None:
    toolkit = AnalysisToolkit()
    toolkit.load_table("sales", content=SALES_CSV)
    run = toolkit.execute_operation(
        "sales",
        "group_aggregate",
        {
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
        },
    )

    out_dir = Path(__file__).resolve().parents[1] / ".artifacts" / "epistemic_firewall_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, answer in CASES:
        result = toolkit.verify_claims(answer, run_id=run["run_id"])
        certificate = result["certificate"]
        (out_dir / f"{label}.json").write_text(
            json.dumps(certificate, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        verdict = certificate["verdict"]
        ungrounded = [
            claim["text"]
            for claim in certificate["claims"]
            if claim["kind"] == "numeric" and claim["status"] == "ungrounded"
        ]
        causal = [
            claim["text"]
            for claim in certificate["claims"]
            if claim["kind"] == "causal" and claim["status"] == "flagged"
        ]
        print(f"\n=== {label} ===")
        print(f"answer:   {answer}")
        print(f"overall:  {verdict['overall']}")
        print(f"grounded: {verdict['grounded']}  recomputed: {verdict['recomputed']}")
        print(f"stats:    {verdict['statistical_validity']}  causal: {verdict['causal_overreach']}")
        if ungrounded:
            print(f"ungrounded numbers: {ungrounded}")
        if causal:
            print(f"causal language flagged: {causal}")

    print(f"\nCertificates written to {out_dir}")


if __name__ == "__main__":
    main()
