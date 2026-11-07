"""Binding metadata for human-approved analysis plans.

The plan review event carries a :class:`PlanApproval`. The client echoes it back
when submitting the approved plan. Validation checks that the approval belongs
to the current dataset generation and has not expired, so a plan reviewed against
one upload cannot silently execute against another.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Optional

from insight.domain.dataset import DatasetProfile
from insight.domain.query import PlanApproval
from insight.settings import get_settings


def plan_digest(plan: dict) -> str:
    """Stable digest of the plan payload as the client sees it."""
    payload = json.dumps(
        plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build_plan_approval(
    plan: dict,
    profile: DatasetProfile,
    *,
    generation: Optional[int] = None,
    now: Optional[float] = None,
) -> PlanApproval:
    issued_at = time.time() if now is None else now
    ttl_minutes = max(1, get_settings().plan_approval_ttl_minutes)
    return PlanApproval(
        dataset_id=profile.fingerprint.dataset_id,
        generation=int(generation or 0),
        plan_hash=plan_digest(plan),
        issued_at=issued_at,
        expires_at=issued_at + (ttl_minutes * 60),
    )


def validate_plan_approval(
    approval: PlanApproval,
    profile: DatasetProfile,
    *,
    generation: Optional[int] = None,
    submitted_plan: Optional[dict] = None,
    now: Optional[float] = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return (issues, audit metadata) for an approval about to be executed.

    ``generation`` is optional for library callers that do not track a dataset
    generation. When supplied, it must match the approval.
    """
    issues: list[str] = []
    current = time.time() if now is None else now

    if approval.dataset_id != profile.fingerprint.dataset_id:
        issues.append(
            "this approval was issued for a different dataset; review the plan again"
        )
    if generation is not None and approval.generation != int(generation):
        issues.append(
            "the dataset changed after this plan was reviewed; review the plan again"
        )
    if approval.expires_at <= current:
        issues.append("this plan approval expired; review the plan again")

    submitted_hash = (
        plan_digest(submitted_plan) if submitted_plan is not None else None
    )
    info: dict[str, Any] = {
        "bound": True,
        "dataset_id": approval.dataset_id,
        "generation": approval.generation,
        "approved_plan_hash": approval.plan_hash,
        "submitted_plan_hash": submitted_hash,
        "edited": bool(submitted_hash and submitted_hash != approval.plan_hash),
        "issued_at": approval.issued_at,
        "expires_at": approval.expires_at,
    }
    return issues, info
