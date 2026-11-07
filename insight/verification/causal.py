"""Advisory causal-language checks.

This is deliberately a small, deterministic lexicon, not a causal-inference
engine. It flags causal verbs in a narrative when the execution plan (or the
caller's notes) shows no identification strategy such as randomisation, an
instrument, diff-in-diff or regression discontinuity.

The result is advisory: a flagged claim is a prompt to inspect the analysis,
not proof that the causal statement is false.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

_CAUSAL_PATTERNS = (
    r"\bcaus(?:e|es|ed|ing)\b",
    r"\bdriv(?:e|es|en|ing)\b",
    r"\bdrove\b",
    r"\bled\s+to\b",
    r"\bleads?\s+to\b",
    r"\bleading\s+to\b",
    r"\bresult(?:s|ed|ing)?\s+in\b",
    r"\bimpact(?:s|ed|ing)?\b",
    r"\beffect\s+of\b",
    r"\bbecause\s+of\b",
    r"\bdue\s+to\b",
    r"\battributable\s+to\b",
    r"\bresponsible\s+for\b",
    r"\blift(?:s|ed|ing)?\b",
)

_IDENTIFICATION_PATTERNS = (
    r"\brandomi[sz](?:e|ed|ation|ing)\b",
    r"\brandom\s+assignment\b",
    r"\bexperiment(?:al|s)?\b",
    r"\ba\s*/\s*b\s+test\b",
    r"\bab\s+test\b",
    r"\bcontrolled\s+trial\b",
    r"\binstrumental\s+variable\b",
    r"\bdifference[-\s]in[-\s]differences?\b",
    r"\bdiff[-\s]in[-\s]diff\b",
    r"\bregression\s+discontinuity\b",
    r"\bnatural\s+experiment\b",
    r"\bquasi[-\s]experiment(?:al)?\b",
    r"\bpropensity\s+score\b",
    r"\bcausal\s+inference\b",
    r"\bsynthetic\s+control\b",
    r"\bmatched\s+pairs?\b",
    r"\bdo[-\s]calculus\b",
)

_IDENTIFICATION_RE = tuple(
    re.compile(pattern, re.IGNORECASE) for pattern in _IDENTIFICATION_PATTERNS
)


@dataclass(frozen=True)
class CausalClaim:
    text: str
    start: int
    end: int
    status: str  # "flagged" | "not_checked"
    rule: str
    message: str


def _plan_text(plan: Any) -> str:
    if plan is None:
        return ""
    parts = [str(getattr(plan, "objective", "") or "")]
    for step in getattr(plan, "steps", []) or []:
        parts.append(str(getattr(step, "operation", "") or ""))
        params = getattr(step, "params", None)
        if params:
            try:
                parts.append(json.dumps(params, default=str, sort_keys=True))
            except (TypeError, ValueError):
                parts.append(str(params))
    return " ".join(parts)


def _has_identification_strategy(plan: Any, notes: Iterable[str]) -> bool:
    haystack = " ".join([_plan_text(plan), " ".join(str(n) for n in notes or [])])
    return any(pattern.search(haystack) for pattern in _IDENTIFICATION_RE)


def detect_causal_claims(
    answer: str,
    *,
    plan: Any = None,
    notes: Iterable[str] = (),
) -> list[CausalClaim]:
    """Return causal language spans, flagged when no design marker is present."""
    text = answer or ""
    identified = _has_identification_strategy(plan, notes)
    claims: list[CausalClaim] = []
    seen: set[tuple[int, int]] = set()

    for pattern in _CAUSAL_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            span = (match.start(), match.end())
            if span in seen:
                continue
            seen.add(span)
            if identified:
                claims.append(
                    CausalClaim(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        status="not_checked",
                        rule="CAUSAL_LANGUAGE_WITH_IDENTIFICATION",
                        message=(
                            "Causal language appears alongside an identification-strategy "
                            "marker; the design itself was not verified."
                        ),
                    )
                )
            else:
                claims.append(
                    CausalClaim(
                        text=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        status="flagged",
                        rule="CAUSAL_LANGUAGE_WITHOUT_IDENTIFICATION",
                        message=(
                            "Causal language appears without a randomised or "
                            "quasi-experimental design marker in the plan or notes."
                        ),
                    )
                )

    return sorted(claims, key=lambda claim: claim.start)
