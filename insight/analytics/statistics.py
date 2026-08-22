from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class TestOutcome:
    test: str
    statistic: float
    p_value: Optional[float]
    effect_size: Optional[float]
    effect_label: str
    sample_note: str
    interpretation: str
    assumptions: list[str]


def isotonic_fit(values: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators isotonic regression (unit weights).

    Returns the non-decreasing least-squares fit for `values` given in their
    current order (caller must pre-sort by the driving score).
    """
    blocks: list[list[float]] = []
    for value in values:
        blocks.append([float(value), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            mean_hi, count_hi = blocks.pop()
            mean_lo, count_lo = blocks.pop()
            pooled_count = count_lo + count_hi
            blocks.append(
                [(mean_lo * count_lo + mean_hi * count_hi) / pooled_count, pooled_count]
            )
    fitted: list[float] = []
    for mean, count in blocks:
        fitted.extend([mean] * int(count))
    return np.asarray(fitted, dtype=float)


@dataclass
class CalibrationReport:
    n: int
    base_rate: float
    brier_raw: float
    brier_calibrated: float
    brier_improvement: float
    ece: float
    reliability_table: pd.DataFrame
    interpretation: str
    notes: str


def _reliability_bins(
    scores: pd.Series, outcomes: pd.Series, max_bins: int = 10
) -> pd.DataFrame:
    ranked = scores.rank(method="first")
    n_bins = max(4, min(max_bins, len(scores) // 40))
    try:
        bins = pd.qcut(ranked, q=n_bins, duplicates="drop")
    except ValueError:
        bins = pd.qcut(ranked, q=max(2, n_bins // 2), duplicates="drop")
    frame = pd.DataFrame({
        "score": scores,
        "outcome": outcomes,
        "bin": bins,
    })
    rows = []
    for _, group in frame.groupby("bin", observed=True):
        low, high = float(group["score"].min()), float(group["score"].max())
        rows.append({
            "bin": f"[{_edge_fmt(low)}, {_edge_fmt(high)}]",
            "n": int(len(group)),
            "avg_score": round(float(group["score"].mean()), 4),
            "observed_rate": round(float(group["outcome"].mean()), 4),
        })
    return pd.DataFrame(rows)


def _edge_fmt(v: float) -> str:
    num = float(v)
    if abs(num) >= 10_000 or float(num).is_integer():
        return f"{int(round(num)):,}"
    return f"{num:.4g}"


def run_isotonic_calibration(
    frame: pd.DataFrame,
    score_column: str,
    outcome_column: str,
) -> CalibrationReport:
    work = frame[[score_column, outcome_column]].copy()
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[outcome_column] = pd.to_numeric(work[outcome_column], errors="coerce")
    work = work.dropna()
    n = len(work)
    if n < 20:
        raise ValueError(
            f"isotonic calibration needs at least 20 complete rows (got {n})"
        )

    outcome_levels = set(np.unique(work[outcome_column].to_numpy()))
    if not outcome_levels.issubset({0.0, 1.0}):
        raise ValueError(
            f"'{outcome_column}' must be binary 0/1 after cleaning "
            f"(found levels: {sorted(outcome_levels)[:6]})"
        )
    if work[score_column].nunique() < 2:
        raise ValueError(f"'{score_column}' has no variance to calibrate")

    scores = work[score_column]
    outcomes = work[outcome_column]

    order = scores.sort_values(kind="stable").index
    sorted_outcomes = outcomes.loc[order].to_numpy(dtype=float)
    fitted_sorted = isotonic_fit(sorted_outcomes)

    fitted = pd.Series(fitted_sorted, index=order).reindex(work.index)

    brier_raw = float(((scores - outcomes) ** 2).mean())
    brier_calibrated = float(((fitted - outcomes) ** 2).mean())
    improvement = (
        (brier_raw - brier_calibrated) / brier_raw if brier_raw > 0 else 0.0
    )
    base_rate = float(outcomes.mean())

    reliability = _reliability_bins(scores, outcomes)
    gaps = (reliability["avg_score"] - reliability["observed_rate"]).abs()
    weights = reliability["n"] / reliability["n"].sum()
    ece = float((gaps * weights).sum())
    signed_gap = float(
        ((reliability["avg_score"] - reliability["observed_rate"]) * weights).sum()
    )

    if ece < 0.05:
        verdict = f"Scores are well calibrated (mean |gap| = {ece:.3f})."
    elif signed_gap > 0:
        verdict = (
            f"Scores are over-confident: they run about {signed_gap:.1%} "
            f"above observed outcome rates on average."
        )
    else:
        verdict = (
            f"Scores are under-confident: they run about {abs(signed_gap):.1%} "
            f"below observed outcome rates on average."
        )
    interpretation = (
        f"{verdict} Isotonic recalibration changes Brier score from "
        f"{brier_raw:.4f} to {brier_calibrated:.4f} "
        f"({'+' if improvement >= 0 else ''}{improvement:.1%} relative)."
    )
    monotone_segments = int((np.diff(fitted_sorted) != 0).sum()) + 1
    notes = (
        f"n={n:,}, base rate={base_rate:.3f}, {len(reliability)} reliability bins, "
        f"{monotone_segments} fitted level(s); assumes independent observations "
        "and a binary outcome."
    )
    return CalibrationReport(
        n=n,
        base_rate=base_rate,
        brier_raw=brier_raw,
        brier_calibrated=brier_calibrated,
        brier_improvement=round(improvement, 6),
        ece=round(ece, 6),
        reliability_table=reliability,
        interpretation=interpretation,
        notes=notes,
    )


def _is_binary(series: pd.Series) -> bool:
    return series.nunique(dropna=True) <= 2


def _is_categorical_like(series: pd.Series, max_categorical_ratio: float = 0.05) -> bool:
    n = len(series.dropna())
    if n == 0:
        return False
    return series.nunique(dropna=True) <= max(2, int(n * max_categorical_ratio))


def _coerce_continuous(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def _cramers_v(table: pd.DataFrame) -> float:
    chi2 = float(table.values.sum() and sps.chi2_contingency(table)[0])
    n = table.values.sum()
    if n == 0:
        return 0.0
    r, k = table.shape
    phi2 = chi2 / n
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def _pct_sentence(baseline_mean: float, other_mean: float, baseline_label: str, other_label: str, value_name: str) -> str:
    if baseline_mean == 0:
        return (
            f"{other_label} averages {_other_fmt(other_mean)} {value_name} "
            f"vs {baseline_label} at {baseline_mean:.4g}."
        )
    change = (other_mean - baseline_mean) / abs(baseline_mean)
    direction = "higher" if change >= 0 else "lower"
    return (
        f"{other_label} averages {abs(change):.0%} {direction} {value_name} "
        f"compared to {baseline_label} "
        f"({_other_fmt(other_mean)} vs {_other_fmt(baseline_mean)})."
    )


def _other_fmt(v: float) -> str:
    return f"{v:,.4g}"


def run_group_comparison(
    frame: pd.DataFrame,
    binary_column: str,
    value_column: str,
) -> TestOutcome:
    work = frame[[binary_column, value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna()
    levels = sorted(work[binary_column].astype(str).unique())
    if len(levels) != 2:
        raise ValueError(
            f"'{binary_column}' must have exactly two levels after cleaning "
            f"(found {len(levels)})"
        )

    a_label, b_label = levels[0], levels[1]
    a = work.loc[work[binary_column].astype(str) == a_label, value_column].values
    b = work.loc[work[binary_column].astype(str) == b_label, value_column].values

    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two observations per group")

    normalish = min(len(a), len(b)) >= 15
    if normalish:
        stat, p = sps.ttest_ind(b, a, equal_var=False)
        test = "Welch's two-sample t-test"
        assumptions = ["independent groups", "approximate normality (n>=15 per group)"]
    else:
        stat, p = sps.mannwhitneyu(b, a, alternative="two-sided")
        test = "Mann-Whitney U"
        assumptions = ["independent groups", "ordinal/continuous values"]

    d = _cohens_d(a, b)
    magnitude = abs(d)
    size = "small" if magnitude < 0.5 else "moderate" if magnitude < 0.8 else "large"

    mean_a, mean_b = float(a.mean()), float(b.mean())
    sentence = _pct_sentence(mean_a, mean_b, a_label, b_label, value_column)

    return TestOutcome(
        test=test,
        statistic=float(stat),
        p_value=float(p) if p is not None else None,
        effect_size=round(d, 4),
        effect_label=f"Cohen's d ({size})",
        sample_note=f"group sizes: {a_label}={len(a):,}, {b_label}={len(b):,}",
        interpretation=(
            f"{sentence} Effect size {size} (d={d:+.2f}); "
            f"difference {'IS' if (p is not None and p < 0.05) else 'is NOT'} "
            f"statistically significant at alpha=0.05 (p={p:.4g})."
        ),
        assumptions=assumptions,
    )


def run_chi_square(frame: pd.DataFrame, col_a: str, col_b: str) -> TestOutcome:
    work = frame[[col_a, col_b]].dropna()
    table = pd.crosstab(work[col_a].astype(str), work[col_b].astype(str))
    if min(table.shape) < 2:
        raise ValueError("need at least two categories in each column")
    expected = sps.chi2_contingency(table)[3]
    low_expected = float((expected < 5).mean())
    if low_expected > 0.2:
        raise ValueError(
            "chi-square unreliable: over 20% of cells have expected count < 5; "
            "aggregate rare categories first"
        )
    chi2, p, dof, _ = sps.chi2_contingency(table)
    v = _cramers_v(table)
    magnitude = "weak" if v < 0.1 else "moderate" if v < 0.3 else "strong" if v < 0.5 else "very strong"
    return TestOutcome(
        test="Chi-square independence",
        statistic=float(chi2),
        p_value=float(p),
        effect_size=round(v, 4),
        effect_label=f"Cramer's V ({magnitude})",
        sample_note=f"contingency table {table.shape[0]}x{table.shape[1]}, n={len(work):,}",
        interpretation=(
            f"Association between '{col_a}' and '{col_b}' is {magnitude} "
            f"(V={v:.3f}), {'significant' if p < 0.05 else 'not significant'} at alpha=0.05 (p={p:.4g})."
        ),
        assumptions=["independent observations", "expected cell counts >= 5"],
    )


def run_correlation(frame: pd.DataFrame, col_a: str, col_b: str, method: str) -> TestOutcome:
    work = frame[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(work) < 3:
        raise ValueError("need at least three complete pairs")
    rho, p = sps.pearsonr(work[col_a], work[col_b]) if method == "pearson" else sps.spearmanr(work[col_a], work[col_b])
    magnitude = abs(float(rho))
    size = "weak" if magnitude < 0.3 else "moderate" if magnitude < 0.5 else "strong"
    return TestOutcome(
        test=f"{method.title()} correlation",
        statistic=float(rho),
        p_value=(float(p) if p is not None and not pd.isna(p) else None),
        effect_size=round(magnitude, 4),
        effect_label=f"|rho| ({size})",
        sample_note=f"complete pairs: {len(work):,}",
        interpretation=(
            f"{size.title()} {method} relationship between '{col_a}' and '{col_b}' "
            f"(rho={rho:+.3f})"
            + (f", {'significant' if p < 0.05 else 'not significant'} at alpha=0.05 (p={float(p):.3g})." if p is not None and not pd.isna(p) else ".")
        ),
        assumptions=["approximately linear (pearson)", "monotonic (spearman)"],
    )


def run_anova(frame: pd.DataFrame, group_column: str, value_column: str) -> TestOutcome:
    work = frame[[group_column, value_column]].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna()
    grouped: dict[str, np.ndarray] = {
        str(level): group[value_column].values
        for level, group in work.groupby(group_column)
    }
    levels = sorted(grouped)
    if len(levels) < 3:
        raise ValueError(
            f"'{group_column}' needs at least three groups for ANOVA (found {len(levels)})"
        )
    samples = [grouped[level] for level in levels]
    if min(len(s) for s in samples) < 2:
        raise ValueError("need at least two observations per group")

    stat, p = sps.f_oneway(*samples)
    all_values = np.concatenate(samples)
    grand_mean = float(all_values.mean())
    ss_between = sum(float(len(s)) * float(s.mean() - grand_mean) ** 2 for s in samples)
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    magnitude = (
        "small" if eta_sq < 0.01
        else "moderate" if eta_sq < 0.06
        else "large" if eta_sq < 0.14
        else "very large"
    )
    means = {level: float(grouped[level].mean()) for level in levels}
    hi = max(means, key=means.get)
    lo = min(means, key=means.get)
    spread = (
        f"Group means range from {_other_fmt(means[lo])} ({lo}) to "
        f"{_other_fmt(means[hi])} ({hi})."
    )
    return TestOutcome(
        test="One-way ANOVA",
        statistic=float(stat),
        p_value=float(p),
        effect_size=round(eta_sq, 4),
        effect_label=f"eta-squared ({magnitude})",
        sample_note=f"{len(levels)} groups, sizes " + ", ".join(
            f"{level}={len(grouped[level]):,}" for level in levels
        ),
        interpretation=(
            f"{spread} Group differences explain {eta_sq:.1%} of variance in "
            f"'{value_column}' ({magnitude} effect); differences across groups are "
            f"{'statistically significant' if p < 0.05 else 'NOT statistically significant'} "
            f"at alpha=0.05 (p={p:.4g})."
        ),
        assumptions=["independent groups", "roughly normal residuals", "comparable variances"],
    )


def route_and_run(
    frame: pd.DataFrame,
    col_a: str,
    col_b: str,
    force: Optional[str] = None,
) -> tuple[str, TestOutcome]:
    s_a = frame[col_a].dropna()
    s_b = frame[col_b].dropna()

    a_num = pd.to_numeric(s_a, errors="coerce").notna().mean() > 0.95
    b_num = pd.to_numeric(s_b, errors="coerce").notna().mean() > 0.95

    if force == "correlation":
        return "correlation:pearson", run_correlation(frame, col_a, col_b, "pearson")
    if force == "correlation_spearman":
        return "correlation:spearman", run_correlation(frame, col_a, col_b, "spearman")
    if force == "chi_square":
        return "chi_square", run_chi_square(frame, col_a, col_b)

    if a_num and b_num:
        if _is_binary(s_a) or _is_binary(s_b):
            binary_col = col_a if _is_binary(s_a) else col_b
            value_col = col_b if binary_col == col_a else col_a
            outcome = run_group_comparison(frame, binary_col, value_col)
            return "group_comparison", outcome
        return "correlation:pearson", run_correlation(frame, col_a, col_b, "pearson")

    if not a_num and not b_num:
        return "chi_square", run_chi_square(frame, col_a, col_b)

    if (a_num and _is_binary(s_a)) or (b_num and _is_binary(s_b)):
        return "chi_square", run_chi_square(frame, col_a, col_b)

    cat_col = col_a if not a_num else col_b
    value_col = col_b if cat_col == col_a else col_a
    grouped = frame.groupby(cat_col)[value_col].apply(
        lambda s: pd.to_numeric(s, errors="coerce").dropna().shape[0]
    )
    if grouped.min() >= 2 and grouped.shape[0] == 2:
        outcome = run_group_comparison(frame, cat_col, value_col)
        return "group_comparison", outcome
    if grouped.shape[0] >= 3 and grouped.min() >= 2:
        try:
            return "anova_test", run_anova(frame, cat_col, value_col)
        except ValueError:
            pass
    return "chi_square", run_chi_square(frame, cat_col, value_col)
