from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from insight.domain.dataset import (
    ColumnProfile,
    ColumnRole,
    DatasetProfile,
    ValueCount,
    build_fingerprint,
)


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


import re

_ID_NAME_PATTERN = re.compile(
    r"(?:^|_)(id|ids|key|keys|code|uuid|guid|sku|ref|no|num|number)(?:$|_)", re.I
)


def _looks_like_identifier_name(name: str) -> bool:
    return bool(_ID_NAME_PATTERN.search(name))


def _infer_role(series: pd.Series, unique_count: int, row_count: int, name: str = "") -> tuple[ColumnRole, str]:
    dtype = str(series.dtype)
    if pd.api.types.is_bool_dtype(series):
        return ColumnRole.boolean, dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnRole.datetime, dtype
    if pd.api.types.is_numeric_dtype(series):
        if (
            row_count > 20
            and unique_count == row_count
            and _looks_like_identifier_name(name)
        ):
            return ColumnRole.identifier, dtype
        return ColumnRole.numeric, dtype

    non_null = series.dropna()
    if non_null.empty:
        return ColumnRole.categorical, dtype

    sample = non_null.astype(str).head(500)
    parsed_dates = pd.to_datetime(sample, errors="coerce", format="mixed")
    if parsed_dates.notna().mean() > 0.85 and sample.str.len().min() >= 6:
        return ColumnRole.datetime, dtype

    numeric_ratio = pd.to_numeric(sample, errors="coerce").notna().mean()
    if numeric_ratio > 0.95:
        if (
            row_count > 20
            and unique_count == row_count
            and _looks_like_identifier_name(name)
        ):
            return ColumnRole.identifier, dtype
        return ColumnRole.numeric, dtype

    avg_len = sample.str.len().mean()
    unique_ratio = unique_count / row_count if row_count else 0.0
    has_id_name = _looks_like_identifier_name(name)
    if row_count > 20 and unique_ratio > 0.9:
        return (ColumnRole.identifier, dtype) if has_id_name else (ColumnRole.text, dtype)
    if avg_len > 40 and unique_count > max(10, row_count * 0.5):
        return ColumnRole.text, dtype
    return ColumnRole.categorical, dtype


def profile_column(df: pd.DataFrame, name: str, row_count: int) -> ColumnProfile:
    series = df[name]
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))
    role, dtype = _infer_role(series, unique_count, row_count, str(name))

    top_values: list[ValueCount] = []
    stats: dict[str, float | None] = {
        "min": None, "max": None, "mean": None, "median": None, "std": None,
    }

    if role == ColumnRole.numeric:
        numeric = pd.to_numeric(series, errors="coerce")
        stats["min"] = _safe_float(numeric.min())
        stats["max"] = _safe_float(numeric.max())
        stats["mean"] = _safe_float(numeric.mean())
        stats["median"] = _safe_float(numeric.median())
        stats["std"] = _safe_float(numeric.std())
    elif role in (ColumnRole.categorical, ColumnRole.boolean):
        value_counts = series.value_counts().head(5)
        top_values = [
            ValueCount(value=_coerce_scalar(v), count=int(c))
            for v, c in value_counts.items()
        ]

    return ColumnProfile(
        name=str(name),
        dtype=dtype,
        role=role,
        missing_count=missing_count,
        missing_pct=round(missing_count / row_count, 4) if row_count else 0.0,
        unique_count=unique_count,
        top_values=top_values,
        **stats,
    )


def _coerce_scalar(v: Any) -> str | int | float | bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return int(f) if f.is_integer() else round(f, 4)
    return str(v)


def profile_dataframe(
    df: pd.DataFrame,
    name: str,
    content_hash: str | None = None,
) -> DatasetProfile:
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) == "object":
            parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
            valid_ratio = parsed.notna().sum() / max(1, df[col].notna().sum())
            if valid_ratio > 0.85 and df[col].notna().any():
                df[col] = parsed

    row_count = len(df)
    columns = [profile_column(df, c, row_count) for c in df.columns]

    return DatasetProfile(
        name=name,
        row_count=row_count,
        column_count=len(df.columns),
        missing_cells=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1_048_576, 3),
        columns=columns,
        fingerprint=build_fingerprint(df, content_hash),
    )
