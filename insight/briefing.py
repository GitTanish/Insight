from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd

from insight.domain.dataset import ColumnRole, DatasetProfile


@dataclass
class Finding:
    rank: int
    severity: float
    category: str
    title: str
    detail: str
    suggested_query: str

    def to_dict(self) -> dict:
        return asdict(self)


_MIN_ROWS = 12
_CORR_SAMPLE_CAP = 20_000
_STRONG_CORR = 0.6
_OUTLIER_FLOOR_SHARE = 0.005
_OUTLIER_CEIL_SHARE = 0.25
_IMBALANCE_TOP_SHARE = 0.85


def _iqr_outlier_share(values: pd.Series) -> tuple[int, int, float, float]:
    valid = values.dropna()
    if len(valid) < 8:
        return 0, 0, 0.0, 0.0
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0, len(valid), 0.0, 0.0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (valid < lower) | (valid > upper)
    return int(mask.sum()), len(valid), float(lower), float(upper)


def _numeric_columns(profile: DatasetProfile) -> list[str]:
    return [c.name for c in profile.columns if c.role == ColumnRole.numeric]


def _categorical_columns(profile: DatasetProfile) -> list[str]:
    return [c.name for c in profile.columns if c.role == ColumnRole.categorical]


def _binary_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    binaries = []
    for name in candidates:
        series = pd.to_numeric(df[name], errors="coerce").dropna()
        if not series.empty and series.nunique() <= 2:
            binaries.append(name)
    return binaries


def _scan_outliers(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    best: Optional[tuple[float, Finding]] = None
    for col in _numeric_columns(profile):
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        count, total, lower, upper = _iqr_outlier_share(numeric)
        if total == 0 or count < 1:
            continue
        share = count / total
        single_spike = count == 1
        if share > _OUTLIER_CEIL_SHARE:
            continue
        if count < 3 and not (single_spike or share >= _OUTLIER_FLOOR_SHARE):
            continue
        severity = min(90.0, max(32.0, share * 300 + 25))
        finding = Finding(
            rank=0,
            severity=severity,
            category="outliers",
            title=f"{count:,} extreme '{col}' values break the expected range",
            detail=(
                f"IQR rule flags {share:.1%} of rows outside "
                f"[{_edge_fmt(lower)}, {_edge_fmt(upper)}]."
            ),
            suggested_query=(
                f"Are there outliers in the {col} column? "
                "Which rows are they and what makes them unusual?"
            ),
        )
        if best is None or severity > best[0]:
            best = (severity, finding)
    return best[1] if best else None


def _edge_fmt(v: float) -> str:
    num = float(v)
    if abs(num) >= 10_000 or float(num).is_integer():
        return f"{int(round(num)):,}"
    return f"{num:.4g}"


def _scan_correlations(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    numeric_cols = _numeric_columns(profile)
    binaries = set(_binary_columns(df, numeric_cols))
    continuous = [c for c in numeric_cols if c not in binaries]
    if len(continuous) < 2:
        return None

    scan = (
        df[continuous].sample(min(len(df), _CORR_SAMPLE_CAP), random_state=0)
        if len(df) > _CORR_SAMPLE_CAP
        else df[continuous]
    )
    numeric_scan = scan.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric_scan = numeric_scan.loc[:, numeric_scan.nunique() > 1]
    if numeric_scan.shape[1] < 2:
        return None

    corr = numeric_scan.corr(method="pearson")
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = corr.iloc[i, j]
            if pd.notna(v):
                pairs.append((abs(float(v)), cols[i], cols[j], float(v)))
    if not pairs:
        return None
    pairs.sort(reverse=True)
    strength, a, b, signed = pairs[0]
    if strength < _STRONG_CORR:
        return None

    direction = "positive" if signed > 0 else "negative"
    return Finding(
        rank=0,
        severity=min(95.0, strength * 80 + 15),
        category="correlation",
        title=f"'{a}' and '{b}' move together strongly ({direction})",
        detail=f"Pearson r = {signed:+.3f} across {len(numeric_scan):,} complete rows.",
        suggested_query=(
            f"How are {a} and {b} related? Explain the relationship and show a chart."
        ),
    )


def _scan_missingness(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    worst: Optional[tuple[float, str]] = None
    for col in profile.columns:
        if col.missing_pct >= 0.25:
            if worst is None or col.missing_pct > worst[0]:
                worst = (col.missing_pct, col.name)
    if worst is None:
        return None
    pct, name = worst
    return Finding(
        rank=0,
        severity=min(85.0, pct * 60 + 20),
        category="quality",
        title=f"{pct:.0%} of '{name}' is missing",
        detail=f"{int(round(pct * 100))}% missing cells may bias any analysis using this column.",
        suggested_query=(
            f"How much data is missing in {name}, and which rows or patterns does it affect?"
        ),
    )


def _scan_imbalance(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    best: Optional[tuple[float, Finding]] = None
    for col in _categorical_columns(profile):
        series = df[col].dropna()
        n = len(series)
        unique = series.nunique()
        if n < 30 or not (2 <= unique <= 12):
            continue
        top_share = float(series.value_counts(normalize=True).iloc[0])
        if top_share < _IMBALANCE_TOP_SHARE:
            continue
        top_value = str(series.value_counts().index[0])[:24]
        severity = top_share * 55 + 15
        finding = Finding(
            rank=0,
            severity=severity,
            category="imbalance",
            title=f"'{col}' is dominated by one value ({top_share:.0%})",
            detail=f"'{top_value}' accounts for {top_share:.0%} of {n:,} non-null rows.",
            suggested_query=(
                f"What is the distribution of {col}? Does the class imbalance skew conclusions?"
            ),
        )
        if best is None or severity > best[0]:
            best = (severity, finding)
    return best[1] if best else None


def _scan_trend(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    date_col = next(
        (c.name for c in profile.columns if c.role == ColumnRole.datetime), None
    )
    if not date_col:
        return None

    binaries = set(_binary_columns(df, _numeric_columns(profile)))
    metric_candidates = [
        c for c in _numeric_columns(profile)
        if c not in binaries
    ][:4]
    if not metric_candidates:
        return None

    work = df[[date_col] + metric_candidates].copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])
    if len(work) < _MIN_ROWS:
        return None

    periods = work[date_col].dt.to_period("M")

    best: Optional[tuple[float, Finding]] = None
    for metric in metric_candidates:
        values = pd.to_numeric(work[metric], errors="coerce")
        valid_mask = values.notna()
        series = values[valid_mask].groupby(periods[valid_mask].values).mean()
        if len(series) < 4:
            continue
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        if first == 0:
            continue
        change = (last - first) / abs(first)
        if abs(change) < 0.25:
            continue

        direction = "rose" if change > 0 else "fell"
        span = f"{series.index[0]} to {series.index[-1]}"
        severity = min(92.0, abs(change) * 90)
        finding = Finding(
            rank=0,
            severity=severity,
            category="trend",
            title=f"Average {metric} {direction} {abs(change):.0%} month over month",
            detail=f"Monthly average moved from {_edge_fmt(first)} to {_edge_fmt(last)} between {span}.",
            suggested_query=(
                f"Why did {metric} {'rise' if change > 0 else 'fall'} between {span}? "
                "Break the change down by segment."
            ),
        )
        if best is None or severity > best[0]:
            best = (severity, finding)

    return best[1] if best else None


def _scan_duplicates(df: pd.DataFrame, profile: DatasetProfile) -> Optional[Finding]:
    total = len(df)
    if total == 0:
        return None
    dup_share = float(df.duplicated().sum()) / total
    if dup_share < 0.05:
        return None
    return Finding(
        rank=0,
        severity=min(70.0, dup_share * 120 + 10),
        category="quality",
        title=f"{dup_share:.1%} of rows are exact duplicates",
        detail=f"{int(dup_share * total):,} duplicated rows may double-count aggregates.",
        suggested_query="How many duplicate rows exist and how do they distort totals?",
    )


_SCANS = (
    _scan_trend,
    _scan_correlations,
    _scan_outliers,
    _scan_imbalance,
    _scan_missingness,
    _scan_duplicates,
)


def generate_briefing(
    df: pd.DataFrame,
    profile: DatasetProfile,
    max_findings: int = 3,
) -> list[Finding]:
    """Deterministic zero-prompt analysis: top findings right after upload."""
    if len(df) < _MIN_ROWS:
        return []

    findings: list[Finding] = []
    seen_categories: set[str] = set()

    for scan in _SCANS:
        try:
            finding = scan(df, profile)
        except Exception:
            finding = None
        if finding is None:
            continue
        findings.append(finding)

    findings.sort(key=lambda f: f.severity, reverse=True)

    selected: list[Finding] = []
    for finding in findings:
        if len(selected) >= max_findings:
            break
        if finding.category in {"quality"} and any(
            s.category == "quality" for s in selected
        ):
            continue
        selected.append(finding)
        seen_categories.add(finding.category)

    for rank, finding in enumerate(selected, start=1):
        finding.rank = rank

    return selected


def findings_to_dicts(findings: list[Finding]) -> list[dict]:
    return [f.to_dict() for f in findings]


async def polish_findings_with_llm(
    findings: list[dict],
    generate,
    model_name: str,
) -> list[dict]:
    """Optional single LLM pass to sharpen headline wording (flag-gated).

    Numbers, ranks and categories are frozen — only title/detail prose may
    change. Any failure returns the original findings untouched.
    """
    import json

    from insight.llm.base import LLMRequest

    if not findings:
        return findings

    system = (
        "You are a newspaper copy editor for an analytics briefing. Sharpen each "
        "finding's 'title' (<=90 chars) and 'detail' (one crisp sentence) without "
        "changing ANY number, column name or direction of the claim. Respond with "
        "ONLY JSON: {\"findings\": [{\"rank\": int, \"title\": str, \"detail\": str}]}"
    )
    user = json.dumps(
        {"findings": [
            {"rank": f.get("rank"), "title": f.get("title", ""), "detail": f.get("detail", "")}
            for f in findings
        ]},
        ensure_ascii=False,
    )
    try:
        response = await generate(LLMRequest(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            model=model_name,
            temperature=0.0,
            json_mode=True,
            max_tokens=800,
        ))
        from insight.llm.registry import extract_json

        data = extract_json(response.content)
        polished = {int(item["rank"]): item for item in data["findings"]}
        out: list[dict] = []
        for finding in findings:
            replacement = polished.get(finding.get("rank"))
            if replacement and isinstance(replacement.get("title"), str):
                merged = dict(finding)
                merged["title"] = replacement["title"]
                if isinstance(replacement.get("detail"), str):
                    merged["detail"] = replacement["detail"]
                out.append(merged)
            else:
                out.append(finding)
        return out
    except Exception:
        return findings
