"""Generate deterministic benchmark datasets with known ground truth.

Usage:
    python evaluation/generate_datasets.py [--out evaluation/data]

Writes sales.csv / ecommerce.csv / support.csv plus ground_truth.json holding
the exact aggregates that cases.json expectations are derived from.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 20260822


def make_sales(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    n = 480
    dates = pd.date_range("2024-01-03", periods=n, freq="D")
    regions = np.array(["North", "South", "East", "West"] * (n // 4))
    categories = np.array((["Electronics", "Apparel", "Home"] * (n // 3))[:n])
    base = np.select(
        [regions == "North", regions == "South", regions == "East", regions == "West"],
        [200.0, 240.0, 280.0, 400.0],
    )
    noise = rng.normal(0, 28, n)
    revenue = np.round(base + noise, 2)
    spike_idx = rng.choice(n, size=6, replace=False)
    revenue[spike_idx] = np.round(revenue[spike_idx] + 340.0, 2)
    units = rng.integers(1, 6, n)
    df = pd.DataFrame({
        "order_id": np.arange(1001, 1001 + n),
        "date": dates.astype(str),
        "region": regions,
        "category": categories,
        "revenue": revenue,
        "units": units,
    })

    by_region = df.groupby("region")["revenue"].sum().round(2)
    q1, q3 = df["revenue"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = int(((df["revenue"] < q1 - 1.5 * iqr) | (df["revenue"] > q3 + 1.5 * iqr)).sum())
    truth = {
        "revenue_by_region": {k: float(v) for k, v in by_region.items()},
        "top_region": str(by_region.idxmax()),
        "top_region_total": float(by_region.max()),
        "mean_order_revenue": round(float(df["revenue"].mean()), 4),
        "outlier_rows_iqr": outliers,
        "row_count": int(n),
    }
    return df, truth


def make_ecommerce(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    n = 600
    dates = pd.date_range("2025-01-01", periods=n, freq="h", ).astype(str)
    device = rng.choice(["mobile", "desktop"], n, p=[0.6, 0.4])
    channel = rng.choice(["paid", "organic", "social", "email"], n, p=[0.35, 0.30, 0.20, 0.15])
    p_convert = np.where(device == "mobile", 0.30, 0.12)
    converted = rng.binomial(1, p_convert)
    base_value = np.where(device == "mobile", 85.0, 120.0)
    order_value = np.round(base_value + rng.normal(0, 22, n), 2)
    order_value = np.where(converted == 1, order_value, np.nan)
    duration = np.round(np.exp(rng.normal(4.1, 0.55, n)), 0)
    p_score = p_convert.copy()
    lead_score = np.round(np.clip(p_score + rng.normal(0, 0.07, n), 0.01, 0.99), 4)

    df = pd.DataFrame({
        "session_id": [f"S{i:05d}" for i in range(1, n + 1)],
        "visit_ts": dates,
        "device": device,
        "channel": channel,
        "converted": converted,
        "order_value": order_value,
        "session_duration_sec": duration,
        "lead_score": lead_score,
    })

    conv = df.groupby("device")["converted"].agg(["mean", "sum", "count"])
    aov = df.dropna(subset=["order_value"]).groupby("device")["order_value"].mean()
    from insight.analytics.statistics import run_isotonic_calibration

    report = run_isotonic_calibration(df, "lead_score", "converted")
    truth = {
        "conversion_rate_by_device": {
            k: round(float(v), 6) for k, v in conv["mean"].items()
        },
        "aov_by_device": {
            k: round(float(v), 4) for k, v in aov.items()
        },
        "overall_conversion_rate": round(float(df["converted"].mean()), 6),
        "top_channel_by_sessions": str(df["channel"].mode().iloc[0]),
        "lead_score_ece": report.ece,
        "lead_score_brier_raw": round(report.brier_raw, 6),
        "row_count": int(n),
    }
    return df, truth


def make_support(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    n = 420
    dates = (
        pd.Timestamp("2025-01-06")
        + pd.to_timedelta(rng.integers(0, 175, n), unit="D")
    ).astype(str)
    channel = rng.choice(["email", "chat", "phone"], n, p=[0.40, 0.35, 0.25])
    priority = rng.choice(["low", "medium", "high"], n, p=[0.50, 0.30, 0.20])
    resolve_p = np.select(
        [channel == "email", channel == "chat", channel == "phone"],
        [0.55, 0.85, 0.70],
    )
    resolved = np.where(rng.binomial(1, resolve_p) == 1, "yes", "no")
    handle_mu = np.select(
        [channel == "email", channel == "chat", channel == "phone"],
        [18.0, 8.0, 12.0],
    )
    handle_time = np.maximum(1.0, np.round(handle_mu + rng.normal(0, 4.5, n), 1))

    df = pd.DataFrame({
        "ticket_id": [f"T{i:05d}" for i in range(1, n + 1)],
        "created_at": dates,
        "channel": channel,
        "priority": priority,
        "resolved": resolved,
        "handle_time_min": handle_time,
    })

    res = (
        df.assign(resolved_flag=(df["resolved"] == "yes").astype(int))
        .groupby("channel")["resolved_flag"]
        .mean()
    )
    handle_mean = df.groupby("channel")["handle_time_min"].mean()
    counts = df["channel"].value_counts()
    truth = {
        "resolution_rate_by_channel": {
            k: round(float(v), 6) for k, v in res.items()
        },
        "handle_time_mean_by_channel": {
            k: round(float(v), 4) for k, v in handle_mean.items()
        },
        "mean_handle_time_overall": round(float(df["handle_time_min"].mean()), 4),
        "tickets_by_channel": {k: int(v) for k, v in counts.items()},
        "top_channel_by_tickets": str(counts.idxmax()),
        "row_count": int(n),
    }
    return df, truth


def make_retail_large(rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Large-scale retail orders (~250k rows) with planted structure."""
    n = 250_000
    dates = (
        pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 731, n), unit="D")
    )
    regions = rng.choice(
        ["North", "South", "East", "West", "Central"], n,
        p=[0.22, 0.24, 0.18, 0.26, 0.10],
    )
    formats = rng.choice(["online", "flagship", "outlet"], n, p=[0.55, 0.30, 0.15])
    categories = rng.choice(
        ["electronics", "home", "apparel", "grocery", "beauty"], n,
        p=[0.24, 0.22, 0.26, 0.18, 0.10],
    )
    cat_base = np.select(
        [
            categories == "electronics",
            categories == "home",
            categories == "apparel",
            categories == "grocery",
        ],
        [420.0, 160.0, 70.0, 35.0],
        default=48.0,
    )
    region_mult = np.select(
        [regions == "West", regions == "East", regions == "South", regions == "North"],
        [1.42, 1.08, 1.02, 0.98],
        default=0.92,
    )
    units = rng.integers(1, 9, n)
    noise = rng.lognormal(0.0, 0.25, n)
    revenue = np.round(units * cat_base * region_mult * noise, 2)
    cost_ratio = rng.uniform(0.50, 0.78, n)
    cost = np.round(revenue * cost_ratio, 2)
    margin = np.round(revenue - cost, 2)
    return_p = np.where(formats == "online", 0.09, 0.04)
    returned = rng.binomial(1, return_p)
    satisfaction = np.clip(rng.binomial(5, 0.78, n) + 1, 1, 5)

    df = pd.DataFrame({
        "order_id": np.arange(1, n + 1),
        "order_date": dates.astype(str),
        "region": regions,
        "store_format": formats,
        "category": categories,
        "units": units,
        "revenue": revenue,
        "margin": margin,
        "returned": returned,
        "satisfaction_score": satisfaction,
    })

    by_region = df.groupby("region")["revenue"].sum().round(2)
    rho = float(
        pd.to_numeric(df["revenue"]).corr(pd.to_numeric(df["margin"]), method="pearson")
    )
    ret = (
        df.groupby("store_format")["returned"].agg(["mean", "count"])
    )
    q1, q3 = df["revenue"].quantile([0.25, 0.75])
    iqr = q3 - q1
    outliers = int(((df["revenue"] < q1 - 1.5 * iqr) | (df["revenue"] > q3 + 1.5 * iqr)).sum())
    truth = {
        "row_count": int(n),
        "revenue_by_region": {k: float(v) for k, v in by_region.items()},
        "top_region": str(by_region.idxmax()),
        "top_region_total": float(by_region.max()),
        "mean_order_revenue": round(float(df["revenue"].mean()), 4),
        "pearson_r_revenue_margin": round(rho, 6),
        "return_rate_by_format": {
            k: round(float(v), 6) for k, v in ret["mean"].items()
        },
        "outlier_rows_iqr": outliers,
    }
    return df, truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    builders = {
        "sales.csv": make_sales,
        "ecommerce.csv": make_ecommerce,
        "support.csv": make_support,
        "retail_250k.csv": make_retail_large,
    }
    ground_truth: dict[str, dict] = {}
    for filename, builder in builders.items():
        frame, truth = builder(rng)
        frame.to_csv(out_dir / filename, index=False)
        ground_truth[filename] = truth
        print(f"wrote {out_dir / filename} ({len(frame):,} rows)")

    (out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'ground_truth.json'}")


if __name__ == "__main__":
    main()
