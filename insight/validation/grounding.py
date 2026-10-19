from __future__ import annotations

import re
from typing import Any, Iterable

_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE_RE = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b")
_ORDINAL_WORDS = {
    "rank", "step", "top", "no", "number", "part", "phase", "q",
    "question", "iteration", "round", "row", "column",
}
_ORDINAL_SUFFIXES = ("st", "nd", "rd", "th")


def _norm_token(token: str) -> str:
    return token.replace(",", "").replace("%", "").strip()


def _to_float(value: Any):
    try:
        return float(_norm_token(str(value)))
    except (TypeError, ValueError):
        return None


def _is_ordinal_context(answer: str, start: int, end: int) -> bool:
    if answer[end:end + 2].lower() in _ORDINAL_SUFFIXES:
        return True
    match = re.search(r"([A-Za-z#]+)\s*$", answer[:start])
    if not match:
        return False
    return match.group(1).lower().lstrip("#") in _ORDINAL_WORDS


def _is_date_component(answer: str, start: int, end: int) -> bool:
    window_start = max(0, start - 12)
    window_end = min(len(answer), end + 12)
    window = answer[window_start:window_end]
    for match in _DATE_RE.finditer(window):
        if window_start + match.start() <= start and end <= window_start + match.end():
            return True
    return False


def collect_evidence_numbers(calcs, tables, notes: str = "", warnings: Iterable[str] = ()) -> tuple[set[float], str]:
    """All numeric literals the answer is allowed to cite."""
    floats: set[float] = set()
    texts: list[str] = []

    def absorb(value: Any) -> None:
        if value is None:
            return
        text = str(value)
        texts.append(_norm_token(text))
        number = _to_float(text)
        if number is not None:
            floats.add(round(number, 6))
        for match in _NUMBER_RE.finditer(text):
            parsed = _to_float(match.group())
            if parsed is not None:
                floats.add(round(parsed, 6))

    for calc in calcs or []:
        absorb(getattr(calc, "value", None))
        absorb(getattr(calc, "detail", None))
        absorb(getattr(calc, "label", None))
    for table in tables or []:
        texts.append(_norm_token(" ".join(str(c) for c in getattr(table, "columns", []))))
        for row in getattr(table, "rows", []):
            for cell in row:
                absorb(cell)
    absorb(notes)
    for warning in warnings or []:
        absorb(warning)
    return floats, "\n".join(texts)


def find_ungrounded_numbers(
    answer: str,
    calcs,
    tables,
    notes: str = "",
    warnings: Iterable[str] = (),
) -> list[str]:
    """Numbers stated in `answer` that cannot be traced to computed evidence."""
    floats, evidence_blob = collect_evidence_numbers(calcs, tables, notes, warnings)
    evidence_tokens = {
        _norm_token(match.group()) for match in _NUMBER_RE.finditer(evidence_blob)
    }

    violations: list[str] = []
    seen: set[str] = set()
    for match in _NUMBER_RE.finditer(answer or ""):
        raw = _norm_token(match.group())
        if raw in seen or raw == "":
            continue
        seen.add(raw)
        number = _to_float(raw)
        if number is None:
            continue

        if round(number, 6) in floats:
            continue
        suffix = answer[match.end():match.end() + 2].lstrip()
        if suffix.startswith("%") and any(
            abs(number - candidate * 100) <= max(0.001 * abs(candidate * 100), 0.11)
            for candidate in floats
            if candidate != 0
        ):
            continue
        if (
            number.is_integer()
            and len(raw) == 4
            and raw.isdigit()
            and 1900 <= number <= 2100
        ):
            continue
        if _is_ordinal_context(answer, match.start(), match.end()):
            continue
        if _is_date_component(answer, match.start(), match.end()):
            continue
        if raw in evidence_tokens:
            continue
        violations.append(raw)
    return violations


def build_grounding_retry_message(violations: list[str]) -> str:
    listed = ", ".join(violations[:10])
    return (
        "Your draft contained figures that do NOT exist in the computed results: "
        f"{listed}.\nRewrite the answer using ONLY numbers that appear in the "
        "execution results above. Do not compute totals, counts or sample sizes "
        "yourself."
    )
