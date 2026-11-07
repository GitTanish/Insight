"""Local HTTP surface for the verification certificate.

Run it with::

    uvicorn insight.verify_service:app --host 127.0.0.1 --port 8787

Then::

    POST /v1/verify
    {
      "answer": "North leads with 150 in revenue.",
      "data": {"csv": "region,revenue\\nNorth,100\\nNorth,50\\n", "table_name": "sales"},
      "operation": {
        "name": "group_aggregate",
        "params": {
          "group_by": "region",
          "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}]
        }
      }
    }

or evidence-only::

    {
      "answer": "Revenue was 77.",
      "evidence": {"numbers": [77]}
    }

This is a local/dev surface, not a multi-tenant service: no persistence, no
queues, no quotas. Set ``INSIGHT_VERIFY_AUTH_TOKEN`` to require a bearer token;
set ``INSIGHT_VERIFY_MAX_CSV_MB`` to cap inline CSV size.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from insight import __version__
from insight.toolkit import AnalysisToolkit, ToolkitError

MAX_CSV_MB_DEFAULT = 20

app = FastAPI(
    title="Insight verification service",
    version=__version__,
    description="Deterministic numeric grounding and recomputation certificates.",
)


class EvidencePayload(BaseModel):
    numbers: list[float] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    notes: Optional[str] = None


class DataPayload(BaseModel):
    csv: str
    table_name: str = "data"


class OperationPayload(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class VerificationRequest(BaseModel):
    answer: str
    evidence: Optional[EvidencePayload] = None
    data: Optional[DataPayload] = None
    operation: Optional[OperationPayload] = None
    sql: Optional[str] = None


def require_token(authorization: Optional[str] = Header(default=None)) -> None:
    expected = (os.environ.get("INSIGHT_VERIFY_AUTH_TOKEN") or "").strip()
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _max_csv_bytes() -> int:
    raw = os.environ.get("INSIGHT_VERIFY_MAX_CSV_MB", str(MAX_CSV_MB_DEFAULT))
    try:
        return max(1, int(raw)) * 1024 * 1024
    except ValueError:
        return MAX_CSV_MB_DEFAULT * 1024 * 1024


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/verify", dependencies=[Depends(require_token)])
def verify(request: VerificationRequest) -> dict[str, Any]:
    if request.data is None and request.evidence is None:
        raise HTTPException(
            status_code=422,
            detail="provide evidence or data with an operation/sql",
        )
    if request.data is not None:
        if bool(request.operation) == bool(request.sql):
            raise HTTPException(
                status_code=422,
                detail="data requires exactly one of operation or sql",
            )
        if len(request.data.csv.encode("utf-8")) > _max_csv_bytes():
            raise HTTPException(status_code=413, detail="inline CSV too large")

    toolkit = AnalysisToolkit()
    evidence = request.evidence or EvidencePayload()

    try:
        if request.data is not None:
            toolkit.load_table(request.data.table_name, content=request.data.csv)
            if request.sql:
                toolkit.run_sql(request.sql, tables=[request.data.table_name])
            else:
                toolkit.execute_operation(
                    request.data.table_name,
                    request.operation.name,
                    request.operation.params,
                )
        result = toolkit.verify_claims(
            request.answer,
            evidence_numbers=evidence.numbers,
            evidence_tables=evidence.tables,
            evidence_notes=evidence.notes,
        )
    except ToolkitError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return result["certificate"]
