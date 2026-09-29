from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class ColumnRole(str, Enum):
    numeric = "numeric"
    categorical = "categorical"
    boolean = "boolean"
    datetime = "datetime"
    text = "text"
    identifier = "identifier"


class ValueCount(BaseModel):
    value: str | int | float | bool
    count: int


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    role: ColumnRole
    missing_count: int = 0
    missing_pct: float = 0.0
    unique_count: int = 0
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    top_values: list[ValueCount] = Field(default_factory=list)

    def describe_line(self, max_top: int = 3) -> str:
        parts = [f"{self.name} [{self.role.value}] missing={self.missing_pct:.1%}"]
        if self.role == ColumnRole.numeric and self.mean is not None:
            parts.append(
                f"min={self._fmt(self.min)} max={self._fmt(self.max)} "
                f"mean={self._fmt(self.mean)} median={self._fmt(self.median)}"
            )
        elif self.top_values:
            tops = ", ".join(
                f"{vc.value}({vc.count})" for vc in self.top_values[:max_top]
            )
            parts.append(f"unique={self.unique_count} top: {tops}")
        else:
            parts.append(f"unique={self.unique_count}")
        return " | ".join(parts)

    @staticmethod
    def _fmt(v: float | None) -> str:
        if v is None:
            return "?"
        return f"{v:,.4g}"


class DatasetFingerprint(BaseModel):
    dataset_id: str
    schema_hash: str
    content_hash: str
    profile_version: str = "v1"


class DatasetProfile(BaseModel):
    name: str
    row_count: int
    column_count: int
    missing_cells: int
    duplicate_rows: int
    memory_mb: float
    columns: list[ColumnProfile]
    fingerprint: DatasetFingerprint
    artifact_columns: list[str] = Field(default_factory=list)

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def canonical_column(self, name: str) -> str | None:
        target = name.strip().lower()
        for col in self.columns:
            if col.name.lower() == target:
                return col.name
        return None

    def get_column(self, name: str) -> ColumnProfile | None:
        canonical = self.canonical_column(name)
        if canonical is None:
            return None
        return next(c for c in self.columns if c.name == canonical)

    def summary_for_planner(self, max_top: int = 3) -> str:
        lines = [
            f"dataset '{self.name}' rows={self.row_count:,} cols={self.column_count} "
            f"duplicate_rows={self.duplicate_rows:,} missing_cells={self.missing_cells:,}",
            "columns:",
        ]
        for col in self.columns:
            lines.append(f"  - {col.describe_line(max_top)}")
        return "\n".join(lines)


def build_fingerprint(
    df: pd.DataFrame, content_hash: str | None = None
) -> DatasetFingerprint:
    schema_payload = [[str(c), str(t)] for c, t in df.dtypes.items()]
    schema_hash = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True).encode()
    ).hexdigest()[:16]

    if content_hash is None:
        sample = pd.concat([df.head(50), df.tail(50)]) if len(df) > 100 else df
        digest = hashlib.sha256(sample.to_csv(index=False).encode("utf-8", "replace"))
        content_hash = digest.hexdigest()[:16]

    raw_id = f"{schema_hash}:{content_hash}:{len(df)}:{len(df.columns)}"
    dataset_id = uuid.uuid5(uuid.NAMESPACE_URL, raw_id).hex[:16]
    return DatasetFingerprint(
        dataset_id=dataset_id, schema_hash=schema_hash, content_hash=content_hash
    )
