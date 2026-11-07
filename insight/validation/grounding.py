from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

_NUMBER_RE = re.compile(
    r"-?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_DATE_RE = re.compile(r"\b\d{1,4}[/.-]\d{1,2}[/.-]\d{1,4}\b")
_INEQUALITY_RE = re.compile(
    r"(?P<name>[A-Za-z][\w-]*)\s*(?P<op><=|>=|<|>|≤|≥)\s*$"
)
_P_VALUE_NAMES = {"p", "pvalue", "pvalues"}


def _normalized_name(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value or "").lower()


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


@dataclass(frozen=True)
class EvidenceNumber:
    """One numeric literal found in computed evidence.

    ``id`` points at the source (a calculation, a table cell, a note or a
    warning), so a certificate can cite the exact evidence that grounds a
    claim instead of just saying "this number appeared somewhere".
    """

    id: str
    value: Optional[float]
    raw: str
    kind: str
    label: str
    detail: Optional[str] = None
    step_id: Optional[int] = None
    table: Optional[str] = None
    row: Optional[int] = None
    column: Optional[str] = None


@dataclass(frozen=True)
class NumericToken:
    raw: str
    value: Optional[float]
    start: int
    end: int
    suffix_pct: bool = False


@dataclass
class NumericClaim:
    """A number in the narrative, matched (or not) to evidence."""

    raw: str
    value: Optional[float]
    start: int
    end: int
    status: str  # "grounded" | "ungrounded" | "ignored" | "not_checked"
    evidence_ids: list[str] = field(default_factory=list)
    rule: Optional[str] = None
    reason: Optional[str] = None


def iter_numeric_tokens(answer: str) -> Iterable[NumericToken]:
    text = answer or ""
    for match in _NUMBER_RE.finditer(text):
        raw = _norm_token(match.group())
        if not raw:
            continue
        suffix = text[match.end():match.end() + 2].lstrip()
        yield NumericToken(
            raw=raw,
            value=_to_float(raw),
            start=match.start(),
            end=match.end(),
            suffix_pct=suffix.startswith("%"),
        )


def build_evidence_index(
    calcs,
    tables,
    notes: str = "",
    warnings: Iterable[str] = (),
) -> list[EvidenceNumber]:
    """Index every numeric literal in the evidence with a stable source id."""
    items: list[EvidenceNumber] = []
    seen: set[tuple] = set()

    def add(
        raw: str,
        value: Optional[float],
        *,
        evidence_id: str,
        kind: str,
        label: str,
        detail: Optional[str] = None,
        step_id: Optional[int] = None,
        table: Optional[str] = None,
        row: Optional[int] = None,
        column: Optional[str] = None,
    ) -> None:
        key = (evidence_id, raw, value, kind, label, step_id, table, row, column)
        if key in seen:
            return
        seen.add(key)
        items.append(
            EvidenceNumber(
                id=evidence_id,
                value=value,
                raw=raw,
                kind=kind,
                label=label,
                detail=detail,
                step_id=step_id,
                table=table,
                row=row,
                column=column,
            )
        )

    def add_text(
        text: Any,
        *,
        evidence_id: str,
        kind: str,
        label: str,
        detail: Optional[str] = None,
        step_id: Optional[int] = None,
        table: Optional[str] = None,
        row: Optional[int] = None,
        column: Optional[str] = None,
    ) -> None:
        if text is None:
            return
        raw_text = str(text)
        full = _to_float(raw_text)
        if full is not None:
            add(
                _norm_token(raw_text),
                full,
                evidence_id=evidence_id,
                kind=kind,
                label=label,
                detail=detail,
                step_id=step_id,
                table=table,
                row=row,
                column=column,
            )
        for match in _NUMBER_RE.finditer(raw_text):
            raw = _norm_token(match.group())
            value = _to_float(raw)
            if value is None:
                continue
            add(
                raw,
                value,
                evidence_id=evidence_id,
                kind=kind,
                label=label,
                detail=detail,
                step_id=step_id,
                table=table,
                row=row,
                column=column,
            )

    for i, calc in enumerate(calcs or []):
        evidence_id = f"calc:{i}"
        label = str(getattr(calc, "label", f"calculation_{i}"))
        detail = getattr(calc, "detail", None)
        add_text(
            getattr(calc, "value", None),
            evidence_id=evidence_id,
            kind="calculation",
            label=label,
            detail=str(detail) if detail is not None else None,
        )
        add_text(
            detail,
            evidence_id=evidence_id,
            kind="calculation",
            label=label,
            detail=str(detail) if detail is not None else None,
        )
        add_text(
            label,
            evidence_id=evidence_id,
            kind="calculation",
            label=label,
            detail=str(detail) if detail is not None else None,
        )

    for table in tables or []:
        table_name = str(getattr(table, "name", "table"))
        columns = [str(c) for c in getattr(table, "columns", [])]
        for column in columns:
            add_text(
                column,
                evidence_id=f"table:{table_name}:col:{column}",
                kind="table_column",
                label=f"{table_name}.{column}",
                table=table_name,
                column=column,
            )
        for row_idx, row in enumerate(getattr(table, "rows", [])):
            for col_idx, cell in enumerate(row):
                column = columns[col_idx] if col_idx < len(columns) else str(col_idx)
                add_text(
                    cell,
                    evidence_id=f"table:{table_name}:r{row_idx}:{column}",
                    kind="table_cell",
                    label=f"{table_name}[{row_idx}].{column}",
                    detail=str(cell)[:500] if cell is not None else None,
                    table=table_name,
                    row=row_idx,
                    column=column,
                )

    add_text(notes, evidence_id="note:0", kind="note", label="evidence_notes")
    for i, warning in enumerate(warnings or []):
        add_text(
            warning,
            evidence_id=f"warning:{i}",
            kind="warning",
            label=f"warning[{i}]",
            detail=str(warning)[:500],
        )

    return items


def _percentage_close(value: float, candidate: float) -> bool:
    if candidate == 0:
        return False
    scaled = candidate * 100
    return abs(value - scaled) <= max(0.001 * abs(scaled), 0.11)


def _decimal_places(raw: str) -> Optional[int]:
    mantissa = raw.split("e")[0].split("E")[0]
    if "." not in mantissa:
        return 0
    return len(mantissa.split(".")[1])


def _exact_close(a: float, b: float) -> bool:
    """Six-decimal-style equality without collapsing tiny values to zero.

    A plain ``round(x, 6)`` comparison makes 4.794e-16 equal to 0.0, which
    would ground any tiny fabricated p-value against a rounded-away zero.
    Small magnitudes therefore require relative closeness.
    """
    if a == b:
        return True
    diff = abs(a - b)
    scale = max(abs(a), abs(b))
    if scale < 1e-6:
        return diff <= 1e-6 * scale
    if diff <= 5e-7:
        return True
    return diff <= 1e-9 * scale


def _rounded_matches(
    token: NumericToken, index: list[EvidenceNumber]
) -> tuple[list[EvidenceNumber], Optional[str]]:
    """Match a claim that is the evidence rounded to the claim's precision.

    Rounding is only allowed in the direction narrative -> evidence: the
    claimed token must equal ``round(evidence, places)`` at exactly the
    precision the author wrote. A fabricated token that is merely close to
    an evidence value still fails (for example, 8.56 does not match a
    reported statistic of 8.554537, while 8.55 does).
    """
    if token.value is None or "e" in token.raw.lower():
        return [], None
    places = _decimal_places(token.raw)
    if places is None or places > 6:
        return [], None

    signed: list[EvidenceNumber] = []
    magnitude: list[EvidenceNumber] = []
    for item in index:
        if item.value is None or "e" in item.raw.lower():
            continue
        try:
            if round(item.value, places) == token.value:
                signed.append(item)
            elif round(abs(item.value), places) == abs(token.value):
                magnitude.append(item)
        except (TypeError, ValueError):
            continue
    if signed:
        return signed, "rounded_numeric_match"
    if magnitude:
        return magnitude, "rounded_magnitude_match"
    return [], None


def _inequality_context(answer: str, token: NumericToken) -> Optional[str]:
    prefix = answer[max(0, token.start - 24):token.start]
    match = _INEQUALITY_RE.search(prefix)
    if not match:
        return None
    if _normalized_name(match.group("name")) not in _P_VALUE_NAMES:
        return None
    return match.group("op").replace("≤", "<=").replace("≥", ">=")


def _is_p_value_item(item: EvidenceNumber) -> bool:
    for value in (item.label, item.column):
        if value and _normalized_name(str(value)) in _P_VALUE_NAMES:
            return True
    return False


def _threshold_holds(actual: float, op: str, threshold: float) -> bool:
    if op == "<":
        return actual < threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == ">=":
        return actual >= threshold
    return False


def _match_inequality(
    token: NumericToken,
    op: str,
    index: list[EvidenceNumber],
) -> tuple[list[str], str, Optional[str], Optional[str]]:
    """Evaluate ``p < 0.001`` style claims against p-value evidence."""
    p_items = [
        item
        for item in index
        if item.value is not None and _is_p_value_item(item)
    ]
    if not p_items:
        return [], "not_checked", None, "inequality_threshold_unverified"
    outcomes = [
        _threshold_holds(float(item.value), op, float(token.value))
        for item in p_items
    ]
    if all(outcomes):
        return _unique_ids(p_items), "grounded", "inequality_threshold_match", None
    if not any(outcomes):
        return (
            _unique_ids(p_items),
            "ungrounded",
            "inequality_threshold_match",
            "inequality_threshold_violated",
        )
    return [], "not_checked", None, "inequality_threshold_ambiguous"


def _match_evidence(
    token: NumericToken, index: list[EvidenceNumber]
) -> tuple[list[str], Optional[str]]:
    if token.value is None:
        return [], None

    exact = [
        item
        for item in index
        if item.value is not None and _exact_close(item.value, token.value)
    ]
    if exact:
        return _unique_ids(exact), "exact_numeric_match"

    if token.suffix_pct:
        scaled = [
            item
            for item in index
            if item.value is not None and _percentage_close(token.value, item.value)
        ]
        if scaled:
            return _unique_ids(scaled), "percentage_scale_match"

    rounded, rounded_rule = _rounded_matches(token, index)
    if rounded:
        return _unique_ids(rounded), rounded_rule

    verbatim = [item for item in index if item.raw == token.raw]
    if verbatim:
        return _unique_ids(verbatim), "verbatim_token_match"

    return [], None


def _unique_ids(items: Iterable[EvidenceNumber]) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    for item in items:
        if item.id in seen:
            continue
        seen.add(item.id)
        ids.append(item.id)
    return ids


def _ignored_reason(token: NumericToken, answer: str) -> Optional[str]:
    if token.value is None:
        return "unparseable"
    if (
        token.value.is_integer()
        and len(token.raw) == 4
        and token.raw.isdigit()
        and 1900 <= token.value <= 2100
    ):
        return "year_like"
    if _is_ordinal_context(answer, token.start, token.end):
        return "ordinal_context"
    if _is_date_component(answer, token.start, token.end):
        return "date_component"
    return None


def ground_numeric_claims(
    answer: str,
    calcs,
    tables,
    notes: str = "",
    warnings: Iterable[str] = (),
    *,
    index: Optional[list[EvidenceNumber]] = None,
) -> list[NumericClaim]:
    """Extract numeric tokens from ``answer`` and bind each to evidence.

    Unlike :func:`find_ungrounded_numbers`, this keeps the token span and the
    source ids that justify it. Duplicate raw tokens are reported once, in
    order of first appearance, matching the legacy grounding behaviour.
    """
    evidence = index if index is not None else build_evidence_index(
        calcs, tables, notes, warnings
    )
    text = answer or ""
    claims: list[NumericClaim] = []
    seen: set[str] = set()

    for token in iter_numeric_tokens(text):
        if token.raw in seen:
            continue
        seen.add(token.raw)

        evidence_ids, rule = _match_evidence(token, evidence)
        inequality_op = (
            _inequality_context(text, token) if token.value is not None else None
        )
        if inequality_op is not None:
            evidence_ids, status, rule, reason = _match_inequality(
                token, inequality_op, evidence
            )
            claims.append(
                NumericClaim(
                    raw=token.raw,
                    value=token.value,
                    start=token.start,
                    end=token.end,
                    status=status,
                    evidence_ids=evidence_ids,
                    rule=rule,
                    reason=reason,
                )
            )
            continue

        if evidence_ids:
            claims.append(
                NumericClaim(
                    raw=token.raw,
                    value=token.value,
                    start=token.start,
                    end=token.end,
                    status="grounded",
                    evidence_ids=evidence_ids,
                    rule=rule,
                )
            )
            continue

        reason = _ignored_reason(token, text)
        if reason is not None:
            claims.append(
                NumericClaim(
                    raw=token.raw,
                    value=token.value,
                    start=token.start,
                    end=token.end,
                    status="ignored",
                    reason=reason,
                )
            )
            continue

        claims.append(
            NumericClaim(
                raw=token.raw,
                value=token.value,
                start=token.start,
                end=token.end,
                status="ungrounded",
                reason="no_matching_evidence",
            )
        )

    return claims


def collect_evidence_numbers(
    calcs,
    tables,
    notes: str = "",
    warnings: Iterable[str] = (),
) -> tuple[set[float], str]:
    """All numeric literals the answer is allowed to cite."""
    index = build_evidence_index(calcs, tables, notes, warnings)
    floats = {
        round(item.value, 6)
        for item in index
        if item.value is not None
    }
    return floats, "\n".join(item.raw for item in index)


def find_ungrounded_numbers(
    answer: str,
    calcs,
    tables,
    notes: str = "",
    warnings: Iterable[str] = (),
) -> list[str]:
    """Numbers stated in `answer` that cannot be traced to computed evidence."""
    claims = ground_numeric_claims(answer, calcs, tables, notes, warnings)
    return [claim.raw for claim in claims if claim.status == "ungrounded"]


def build_grounding_retry_message(violations: list[str]) -> str:
    listed = ", ".join(violations[:10])
    return (
        "Your draft contained figures that do NOT exist in the computed results: "
        f"{listed}.\nRewrite the answer using ONLY numbers that appear in the "
        "execution results above. Do not compute totals, counts or sample sizes "
        "yourself."
    )
