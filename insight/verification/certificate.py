"""Versioned verification certificates.

The certificate is the stable, machine-readable artifact the rest of the
project (and external agents) can consume. It deliberately separates four
different questions:

- ``numeric_grounding``: does every number in the prose appear in evidence?
- ``independent_recomputation``: do the reported aggregates survive a second
  arithmetic path?
- ``statistical_validity``: are there known sample-size / missingness caveats?
- ``causal_language``: is causal prose used without a design marker?

Only the first two can fail a certificate. Statistical and causal findings are
warnings: they are judgement calls, not arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from insight.domain.execution import Calculation, DataTable
from insight.domain.visualization import ValidationIssue
from insight.validation.grounding import (
    EvidenceNumber,
    build_evidence_index,
    ground_numeric_claims,
)
from insight.verification.causal import detect_causal_claims
from insight.verification.signing import sign_payload

CERTIFICATE_SCHEMA_VERSION = "1.0"
VERIFIER_NAME = "insight-verifier"
VERIFIER_VERSION = "0.2.0"

_RECOMPUTE_COVERED_OPS = {
    "group_aggregate",
    "top_n",
    "value_counts",
    "correlation",
    "compare_subsets",
    "statistical_test",
    "anova_test",
    "regression",
}
_RECOMPUTE_CODES = {"RECOMPUTE_MISMATCH", "RECOMPUTE_SKIPPED"}
_STATISTICAL_CODES = {"SMALL_SAMPLE", "MISSINGNESS_MCAR"}
_STRUCTURAL_CODES = {
    "STEP_FAILED",
    "TABLE_MISSING",
    "EMPTY_TABLE",
    "CHART_SOURCE",
    "NON_FINITE",
    "VALIDATION",
    "VALIDATION_FAILED",
}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fallback_code(message: str) -> str:
    text = (message or "").lower()
    if "non-finite" in text:
        return "NON_FINITE"
    if "small sample" in text:
        return "SMALL_SAMPLE"
    if "mcar" in text:
        return "MISSINGNESS_MCAR"
    if "is empty" in text:
        return "EMPTY_TABLE"
    if "recomputation mismatch" in text:
        return "RECOMPUTE_MISMATCH"
    if "independent verification skipped" in text:
        return "RECOMPUTE_SKIPPED"
    if "step" in text and "failed" in text:
        return "STEP_FAILED"
    if "chart" in text:
        return "CHART_SOURCE"
    return "VALIDATION"


class CertificateEvidence(BaseModel):
    id: str
    kind: Literal["calculation", "table_cell", "table_column", "note", "warning"]
    label: str
    value: Any = None
    detail: Optional[str] = None
    numbers: list[float] = Field(default_factory=list)
    step_id: Optional[int] = None
    table: Optional[str] = None
    row: Optional[int] = None
    column: Optional[str] = None


class CertificateClaim(BaseModel):
    id: str
    kind: Literal["numeric", "causal"]
    text: str
    span: tuple[int, int]
    status: Literal["grounded", "ungrounded", "ignored", "flagged", "not_checked"]
    evidence_ids: list[str] = Field(default_factory=list)
    rule: Optional[str] = None
    severity: Literal["info", "warning", "error"] = "info"
    message: Optional[str] = None
    normalized: Optional[str] = None
    normalized_value: Optional[float] = None


class CertificateTable(BaseModel):
    name: str
    columns: list[str]
    rows: int
    truncated: bool = False
    digest: str


class CertificateIssue(BaseModel):
    code: str
    message: str
    severity: Literal["error", "warning"] = "warning"
    step_id: Optional[int] = None


class CertificateCheck(BaseModel):
    status: Literal["pass", "warn", "fail", "not_checked"]
    checked: int = 0
    failed: int = 0
    issues: list[CertificateIssue] = Field(default_factory=list)
    detail: Optional[str] = None


class ReportedAggregate(BaseModel):
    step_id: int
    operation: str
    label: str
    value: Any = None
    detail: Optional[str] = None
    recomputation: Literal["verified", "mismatch", "not_covered", "not_checked"]


class CertificateVerdict(BaseModel):
    overall: Literal["pass", "pass_with_warnings", "fail"]
    grounded: bool
    recomputed: Literal["pass", "fail", "not_checked"]
    statistical_validity: Literal["pass", "warn", "fail", "not_checked"]
    causal_overreach: Literal["pass", "warn", "not_checked"]
    structural: Literal["pass", "warn", "fail", "not_checked"]


class VerificationCertificate(BaseModel):
    schema_version: str = CERTIFICATE_SCHEMA_VERSION
    certificate_id: str
    issued_at: datetime
    verifier: dict[str, str]
    attestation_level: Literal["grounding", "recomputed"]
    subject: dict[str, Any]
    tables: list[CertificateTable] = Field(default_factory=list)
    evidence: list[CertificateEvidence] = Field(default_factory=list)
    aggregates: list[ReportedAggregate] = Field(default_factory=list)
    claims: list[CertificateClaim] = Field(default_factory=list)
    checks: dict[str, CertificateCheck] = Field(default_factory=dict)
    verdict: CertificateVerdict
    limitations: list[str] = Field(default_factory=list)
    signature: Optional[dict[str, Any]] = None


def _issue_model(issue: ValidationIssue) -> CertificateIssue:
    return CertificateIssue(
        code=issue.code or _fallback_code(issue.message),
        message=issue.message,
        severity=issue.severity,
        step_id=issue.step_id,
    )


def _check_status(issues: list[ValidationIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "fail"
    if issues:
        return "warn"
    return "pass"


def _build_evidence_items(
    calcs: list[Calculation],
    tables: list[DataTable],
    notes: str,
    warnings: list[str],
    referenced_ids: set[str],
    evidence_index: list[EvidenceNumber],
) -> list[CertificateEvidence]:
    items: list[CertificateEvidence] = []
    numbers_by_id: dict[str, list[float]] = {}
    for number in evidence_index:
        if number.value is None:
            continue
        numbers_by_id.setdefault(number.id, []).append(number.value)

    for i, calc in enumerate(calcs or []):
        evidence_id = f"calc:{i}"
        items.append(
            CertificateEvidence(
                id=evidence_id,
                kind="calculation",
                label=calc.label,
                value=_jsonable(calc.value),
                detail=calc.detail,
                numbers=numbers_by_id.get(evidence_id, []),
            )
        )

    for table in tables or []:
        table_name = str(table.name)
        columns = [str(column) for column in table.columns]
        for column in columns:
            evidence_id = f"table:{table_name}:col:{column}"
            if evidence_id not in referenced_ids:
                continue
            items.append(
                CertificateEvidence(
                    id=evidence_id,
                    kind="table_column",
                    label=f"{table_name}.{column}",
                    value=column,
                    numbers=numbers_by_id.get(evidence_id, []),
                    table=table_name,
                    column=column,
                )
            )
        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row):
                column = columns[col_idx] if col_idx < len(columns) else str(col_idx)
                evidence_id = f"table:{table_name}:r{row_idx}:{column}"
                if evidence_id not in referenced_ids:
                    continue
                items.append(
                    CertificateEvidence(
                        id=evidence_id,
                        kind="table_cell",
                        label=f"{table_name}[{row_idx}].{column}",
                        value=_jsonable(cell),
                        detail=str(cell)[:500] if cell is not None else None,
                        numbers=numbers_by_id.get(evidence_id, []),
                        table=table_name,
                        row=row_idx,
                        column=column,
                    )
                )

    if notes:
        items.append(
            CertificateEvidence(
                id="note:0",
                kind="note",
                label="evidence_notes",
                value=notes,
                detail=notes[:500],
                numbers=numbers_by_id.get("note:0", []),
            )
        )
    for i, warning in enumerate(warnings or []):
        items.append(
            CertificateEvidence(
                id=f"warning:{i}",
                kind="warning",
                label=f"warning[{i}]",
                value=str(warning),
                detail=str(warning)[:500],
                numbers=numbers_by_id.get(f"warning:{i}", []),
            )
        )

    return items


def _build_aggregates(
    plan: Any,
    execution: Any,
    recompute_issues: list[ValidationIssue],
) -> list[ReportedAggregate]:
    if plan is None or execution is None:
        return []

    plan_by_id = {step.step_id: step for step in getattr(plan, "steps", []) or []}
    mismatch_messages: dict[int, list[str]] = {}
    skipped_steps = {
        issue.step_id
        for issue in recompute_issues
        if issue.code == "RECOMPUTE_SKIPPED"
    }
    for issue in recompute_issues:
        if issue.code == "RECOMPUTE_MISMATCH" and issue.step_id is not None:
            mismatch_messages.setdefault(issue.step_id, []).append(
                issue.message or ""
            )

    def is_numeric(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float))

    aggregates: list[ReportedAggregate] = []
    for step in getattr(execution, "steps", []) or []:
        if not step.success or step.output is None:
            continue
        plan_step = plan_by_id.get(step.step_id)
        covered = (
            plan_step is not None
            and plan_step.input_step is None
            and step.operation in _RECOMPUTE_COVERED_OPS
        )
        numeric_labels = {
            str(calc.label)
            for calc in step.output.calculations
            if is_numeric(calc.value)
        }
        named_mismatches: set[str] = set()
        if step.step_id in mismatch_messages:
            named_mismatches = {
                label
                for label in numeric_labels
                if any(
                    re.search(rf"\b{re.escape(label)}\b", message)
                    for message in mismatch_messages[step.step_id]
                )
            }
        if not covered:
            status = "not_covered"
        elif step.step_id in skipped_steps:
            status = "not_checked"
        elif step.step_id in mismatch_messages:
            # If the issue names no aggregate label at all, treat it as a
            # step-wide mismatch; otherwise only the named aggregate failed.
            status = "mismatch" if not named_mismatches else "verified"
        else:
            status = "verified"

        for calc in step.output.calculations:
            calc_status = status
            if not is_numeric(calc.value):
                calc_status = "not_checked"
            elif status == "verified" and named_mismatches and str(calc.label) in named_mismatches:
                calc_status = "mismatch"
            aggregates.append(
                ReportedAggregate(
                    step_id=step.step_id,
                    operation=step.operation,
                    label=calc.label,
                    value=_jsonable(calc.value),
                    detail=calc.detail,
                    recomputation=calc_status,
                )
            )

    return aggregates


def build_certificate(
    answer: str,
    *,
    run_id: Optional[str],
    calcs: list[Calculation],
    tables: list[DataTable],
    notes: list[str],
    warnings: list[str],
    issues: list[ValidationIssue],
    plan: Any = None,
    execution: Any = None,
    dataset_sha256: Optional[str] = None,
    attestation_level: Literal["grounding", "recomputed"] = "grounding",
    verifier_version: str = VERIFIER_VERSION,
    signing_key: Optional[str] = None,
    signing_key_id: str = "default",
) -> VerificationCertificate:
    """Assemble a certificate from already-computed evidence and issues."""
    answer = answer or ""
    notes_blob = "; ".join(str(note) for note in notes or [] if note)
    evidence_index = build_evidence_index(calcs, tables, notes_blob, warnings)

    numeric_claims = ground_numeric_claims(
        answer,
        calcs,
        tables,
        notes_blob,
        warnings,
        index=evidence_index,
    )
    causal_claims = detect_causal_claims(answer, plan=plan, notes=notes)

    recompute_issues = [
        issue
        for issue in issues
        if issue.code in _RECOMPUTE_CODES
        or "independent recomputation" in (issue.message or "").lower()
    ]
    statistical_issues = [
        issue for issue in issues if issue.code in _STATISTICAL_CODES
    ]
    structural_issues = [
        issue
        for issue in issues
        if issue.code in _STRUCTURAL_CODES
        and issue.code not in _RECOMPUTE_CODES
        and issue.code not in _STATISTICAL_CODES
    ]
    classified = {id(issue) for issue in recompute_issues + statistical_issues + structural_issues}
    structural_issues.extend(
        issue for issue in issues if id(issue) not in classified
    )

    checked_numeric = [
        claim for claim in numeric_claims if claim.status in {"grounded", "ungrounded"}
    ]
    ungrounded = [
        claim for claim in checked_numeric if claim.status == "ungrounded"
    ]
    unchecked_numeric = [
        claim for claim in numeric_claims if claim.status == "not_checked"
    ]
    if ungrounded:
        numeric_status = "fail"
    elif unchecked_numeric:
        numeric_status = "warn"
    else:
        numeric_status = "pass"
    if checked_numeric:
        numeric_detail = (
            f"{len(checked_numeric) - len(ungrounded)}/{len(checked_numeric)} "
            "numeric claims grounded"
        )
    else:
        numeric_detail = "No numeric claims found."
    if unchecked_numeric:
        numeric_detail += (
            f"; {len(unchecked_numeric)} threshold claim(s) could not be verified"
        )
    numeric_check = CertificateCheck(
        status=numeric_status,
        checked=len(checked_numeric) + len(unchecked_numeric),
        failed=len(ungrounded),
        detail=numeric_detail,
    )

    covered_step_ids = {
        step.step_id
        for step in (getattr(plan, "steps", []) or [])
        if step.input_step is None and step.operation in _RECOMPUTE_COVERED_OPS
    }

    if attestation_level == "recomputed":
        if not covered_step_ids:
            recompute_status = "not_checked"
            recompute_detail = (
                "No covered operations were present; SQL and chained steps are "
                "not independently recomputed."
            )
        else:
            recompute_status = _check_status(recompute_issues)
            recompute_detail = (
                "A second arithmetic path recomputed the covered operations."
            )
        recompute_check = CertificateCheck(
            status=recompute_status,
            checked=len(covered_step_ids),
            failed=sum(1 for issue in recompute_issues if issue.severity == "error"),
            issues=[_issue_model(issue) for issue in recompute_issues],
            detail=recompute_detail,
        )
        statistical_check = CertificateCheck(
            status=_check_status(statistical_issues),
            checked=len(getattr(plan, "steps", []) or []),
            failed=sum(
                1 for issue in statistical_issues if issue.severity == "error"
            ),
            issues=[_issue_model(issue) for issue in statistical_issues],
            detail="Sample-size and missingness caveats from the execution record.",
        )
        structural_check = CertificateCheck(
            status=_check_status(structural_issues),
            checked=len(getattr(plan, "steps", []) or []),
            failed=sum(1 for issue in structural_issues if issue.severity == "error"),
            issues=[_issue_model(issue) for issue in structural_issues],
            detail="Execution, table and chart-source checks.",
        )
    else:
        recompute_check = CertificateCheck(
            status="not_checked",
            detail="No execution trace was supplied; aggregates were not recomputed.",
        )
        statistical_check = CertificateCheck(
            status="not_checked",
            detail="Statistical checks require an execution trace.",
        )
        structural_check = CertificateCheck(
            status="not_checked",
            detail="No execution trace was supplied.",
        )

    flagged_causal = [
        claim for claim in causal_claims if claim.status == "flagged"
    ]
    unchecked_causal = [
        claim for claim in causal_claims if claim.status == "not_checked"
    ]
    if flagged_causal:
        causal_status = "warn"
        causal_detail = (
            f"{len(flagged_causal)} causal phrase(s) without a design marker."
        )
    elif unchecked_causal:
        causal_status = "not_checked"
        causal_detail = (
            "Causal language appears with a design marker; the design was not verified."
        )
    else:
        causal_status = "pass"
        causal_detail = "No causal language detected."

    causal_check = CertificateCheck(
        status=causal_status,
        checked=len(causal_claims),
        failed=len(flagged_causal),
        detail=causal_detail,
    )

    checks = {
        "numeric_grounding": numeric_check,
        "independent_recomputation": recompute_check,
        "statistical_validity": statistical_check,
        "causal_language": causal_check,
        "structural": structural_check,
    }

    hard_fail = any(check.status == "fail" for check in checks.values())
    has_warning = any(
        check.status in {"warn", "not_checked"} for check in checks.values()
    )
    if hard_fail:
        overall = "fail"
    elif has_warning:
        overall = "pass_with_warnings"
    else:
        overall = "pass"

    recomputed_status = (
        "not_checked"
        if recompute_check.status in {"not_checked", "warn"}
        else recompute_check.status
    )

    certificate_claims: list[CertificateClaim] = []
    for i, claim in enumerate(numeric_claims):
        certificate_claims.append(
            CertificateClaim(
                id=f"n{i}",
                kind="numeric",
                text=answer[claim.start:claim.end],
                span=(claim.start, claim.end),
                status=claim.status,
                evidence_ids=claim.evidence_ids,
                rule=claim.rule,
                severity="error" if claim.status == "ungrounded" else "info",
                message=claim.reason,
                normalized=claim.raw,
                normalized_value=claim.value,
            )
        )
    for i, claim in enumerate(causal_claims):
        certificate_claims.append(
            CertificateClaim(
                id=f"c{i}",
                kind="causal",
                text=claim.text,
                span=(claim.start, claim.end),
                status=claim.status,
                rule=claim.rule,
                severity="warning",
                message=claim.message,
            )
        )

    referenced_ids = {
        evidence_id
        for claim in numeric_claims
        for evidence_id in claim.evidence_ids
    }
    evidence = _build_evidence_items(
        calcs, tables, notes_blob, warnings, referenced_ids, evidence_index
    )
    table_summaries = [
        CertificateTable(
            name=str(table.name),
            columns=[str(column) for column in table.columns],
            rows=table.total_rows,
            truncated=table.truncated,
            digest=_digest(table.model_dump(mode="json")),
        )
        for table in tables or []
    ]
    aggregates = _build_aggregates(plan, execution, recompute_issues)

    subject = {
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "run_id": run_id,
        "plan_digest": (
            _digest(plan.model_dump(mode="json"))
            if plan is not None and hasattr(plan, "model_dump")
            else None
        ),
        "execution_digest": (
            _digest(execution.model_dump(mode="json"))
            if execution is not None and hasattr(execution, "model_dump")
            else None
        ),
        "dataset_sha256": dataset_sha256,
    }

    limitations = [
        "Numeric grounding checks whether a number appears in the supplied evidence; "
        "it does not prove the evidence is true or that it is the right evidence for "
        "the claim.",
        "The verifier does not check emphasis, omitted caveats, semantic entailment, "
        "or whether the answer actually answers the question.",
        "Causal checks are advisory and rule-based; they do not establish causation.",
        "Statistical checks cover sample size and missingness only; they do not "
        "validate test assumptions, multiple comparisons, or study design.",
    ]
    if attestation_level == "grounding":
        limitations.insert(
            0,
            "No execution trace was supplied: aggregates were not independently "
            "recomputed.",
        )

    certificate = VerificationCertificate(
        certificate_id=uuid.uuid4().hex,
        issued_at=datetime.now(timezone.utc),
        verifier={
            "name": VERIFIER_NAME,
            "version": verifier_version,
            "engine": "pandas/scipy/duckdb",
            "python": platform.python_version(),
        },
        attestation_level=attestation_level,
        subject=subject,
        tables=table_summaries,
        evidence=evidence,
        aggregates=aggregates,
        claims=certificate_claims,
        checks=checks,
        verdict=CertificateVerdict(
            overall=overall,
            grounded=numeric_check.status != "fail",
            recomputed=recomputed_status,
            statistical_validity=statistical_check.status,
            causal_overreach=causal_check.status,
            structural=structural_check.status,
        ),
        limitations=limitations,
    )
    if signing_key:
        payload = certificate.model_dump(mode="json", exclude={"signature"})
        certificate = certificate.model_copy(
            update={"signature": sign_payload(payload, signing_key, signing_key_id)}
        )
    return certificate
