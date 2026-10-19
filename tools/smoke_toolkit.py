"""End-to-end smoke of the Insight toolkit + MCP server on real datasets.

Downloads four public CSVs into ``data/`` when missing, then cross-checks the
toolkit against independent pandas/numpy/scipy computations and the generated
``evaluation/data/ground_truth.json``. Covers all 15 operations, the SQL
sandbox, DAG chaining, ``verify_claims``, and the MCP surface (in-memory
client). Manual tool — not part of CI. Run from the repo root::

    python evaluation/generate_datasets.py   # if evaluation/data/ is missing
    python tools/smoke_toolkit.py
"""

import asyncio
import json
import logging
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats as sps

from insight.toolkit import AnalysisToolkit, ToolkitError

logging.getLogger("mcp").setLevel(logging.WARNING)

PUBLIC_DATASETS = {
    "data/superstore.csv": "https://raw.githubusercontent.com/leonism/sample-superstore/master/data/superstore.csv",
    "data/penguins.csv": "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/main/inst/extdata/penguins.csv",
    "data/telco_churn.csv": "https://raw.githubusercontent.com/KimathiNewton/Telco-Customer-Churn/master/Datasets/telco_churn.csv",
    "data/daily_min_temperatures.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
}


def ensure_datasets() -> None:
    for rel, url in PUBLIC_DATASETS.items():
        dest = ROOT / rel
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"INFO  downloading {rel} ...")
        try:
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh)
        except Exception as exc:
            raise SystemExit(f"could not download {url}: {exc}") from exc
    if not (ROOT / "evaluation" / "data" / "ground_truth.json").exists():
        raise SystemExit(
            "evaluation/data/ not found — run: python evaluation/generate_datasets.py"
        )


ensure_datasets()

PASS = 0
FAIL = 0
FAILURES = []


def smoke(label, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"PASS  {label}")
    except Exception as exc:
        FAIL += 1
        FAILURES.append((label, f"{type(exc).__name__}: {exc}"))
        print(f"FAIL  {label} :: {type(exc).__name__}: {exc}")


def rows_of(out):
    table = out.get("table")
    if not table:
        return []
    return [dict(zip(table["columns"], row)) for row in table["rows"]]


def calcs(out):
    return {c["label"]: c["value"] for c in out.get("calculations", [])}


def near(a, b, tol=1e-6):
    assert a is not None and b is not None, f"near(): got {a!r} vs {b!r}"
    assert abs(float(a) - float(b)) <= tol, f"near(): {a!r} != {b!r} (tol {tol})"


def find_number(payload, ref, tol):
    def walk(x):
        if isinstance(x, dict):
            return any(walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(walk(v) for v in x)
        if isinstance(x, bool) or x is None:
            return False
        if isinstance(x, (int, float)):
            return abs(float(x) - float(ref)) <= tol
        if isinstance(x, str):
            try:
                return abs(float(x.replace(",", "").replace("$", "").replace("%", "")) - float(ref)) <= tol
            except ValueError:
                return False
        return False

    assert walk(payload), f"no value within {tol} of {ref} in payload"


def vc_map(out, cat_field=None):
    table = out["table"]
    cols = table["columns"]
    cat_idx = cols.index(cat_field) if cat_field in cols else 0
    count_idx = next(
        (i for i, c in enumerate(cols) if c in ("count", "row_count", "n")), 1
    )
    return {str(row[cat_idx]): row[count_idx] for row in table["rows"]}


def raises(fn, needle):
    try:
        fn()
    except ToolkitError as exc:
        assert needle in str(exc), f"expected {needle!r} in {exc}"
        return
    raise AssertionError("ToolkitError not raised")


data = lambda rel: str(ROOT / Path(rel.replace("\\", "/")))  # noqa: E731

with open(data(r"evaluation\data\ground_truth.json"), encoding="utf-8") as fh:
    GT = json.load(fh)


# ---------------------------------------------------------------- load tables
# The leonism sample-superstore copy concatenates unrelated sample tables after
# the real 9,994 data rows; keep only the first block for the numeric checks.
ss = pd.read_csv(data(r"data\superstore.csv")).iloc[:9994].reset_index(drop=True)
CLEAN_SS = str(Path(tempfile.gettempdir()) / "insight_smoke_superstore_clean.csv")
ss.to_csv(CLEAN_SS, index=False)

tk = AnalysisToolkit()
FILES = {
    "superstore": CLEAN_SS,
    "penguins": data(r"data\penguins.csv"),
    "telco": data(r"data\telco_churn.csv"),
    "temps": data(r"data\daily_min_temperatures.csv"),
    "sales": data(r"evaluation\data\sales.csv"),
    "support": data(r"evaluation\data\support.csv"),
    "ecommerce": data(r"evaluation\data\ecommerce.csv"),
    "diamonds": data(r"evaluation\data\diamonds.csv"),
}
for name, path in FILES.items():
    smoke(f"load {name}", lambda n=name, p=path: tk.load_table(n, path=p))

t0 = time.perf_counter()
tk.load_table("retail250k", path=data(r"evaluation\data\retail_250k.csv"))
retail_load_s = time.perf_counter() - t0
print(f"INFO  retail250k load: {retail_load_s:.2f}s")
assert retail_load_s < 120

pg = pd.read_csv(data(r"data\penguins.csv"))
tel = pd.read_csv(data(r"data\telco_churn.csv"))
tmp = pd.read_csv(data(r"data\daily_min_temperatures.csv"))
sales = pd.read_csv(data(r"evaluation\data\sales.csv"))
sup = pd.read_csv(data(r"evaluation\data\support.csv"))
ecm = pd.read_csv(data(r"evaluation\data\ecommerce.csv"))


# ---------------------------------------------------------------- profile
def check_profile():
    prof = tk.profile_table("superstore")
    assert prof["row_count"] == 9994 and prof["column_count"] == 21
    pg_prof = tk.profile_table("penguins")
    assert pg_prof["row_count"] == 344
    cols = {c["name"]: c for c in pg_prof["columns"]}
    assert "sex" in cols


smoke("profile_table superstore + penguins", check_profile)


# ---------------------------------------------------------------- operation sweep
def check_summarize():
    out = tk.execute_operation("superstore", "summarize", {"columns": ["Sales", "Profit"]})
    assert out["table"] is not None and len(out["table"]["rows"]) >= 2
    out2 = tk.execute_operation("penguins", "summarize", {})
    assert out2["table"] is not None and len(out2["table"]["rows"]) >= 4


def check_value_counts_category():
    out = tk.execute_operation("superstore", "value_counts", {"column": "Category"})
    m = vc_map(out)
    ref = ss["Category"].value_counts().to_dict()
    assert set(m) == set(ref), (m, ref)
    for key, val in ref.items():
        assert m[key] == val, f"{key}: {m[key]} != {val}"


def check_value_counts_other_after():
    out = tk.execute_operation(
        "superstore", "value_counts", {"column": "City", "top_n": 25, "other_after": 20}
    )
    m = vc_map(out)
    other_keys = [k for k in m if k.startswith("Other")]
    assert len(other_keys) == 1, m
    assert sum(m.values()) == ss["City"].notna().sum()


def check_value_counts_normalized():
    out = tk.execute_operation("penguins", "value_counts", {"column": "island", "normalize": True})
    table = out["table"]
    assert "proportion" in table["columns"], table["columns"]
    m = {str(r[0]): r[table["columns"].index("proportion")] for r in table["rows"]}
    ref = pg["island"].value_counts(normalize=True).to_dict()
    assert set(m) == set(ref)
    for key, val in ref.items():
        near(m[key], val, 2e-4)
    near(sum(m.values()), 1.0, 2e-3)


def check_top_n():
    out = tk.execute_operation(
        "superstore", "top_n", {"sort_by": "Sales", "n": 5, "columns": ["Order ID", "Sales"]}
    )
    rows = rows_of(out)
    assert len(rows) == 5
    vals = [r["Sales"] for r in rows]
    assert vals == sorted(vals, reverse=True)
    near(vals[0], ss["Sales"].max(), 1e-6)


def check_group_aggregate_superstore():
    out = tk.execute_operation(
        "superstore",
        "group_aggregate",
        {
            "group_by": "Region",
            "metrics": [
                {"column": "Sales", "agg": "sum", "alias": "sales_total"},
                {"column": "Profit", "agg": "sum", "alias": "profit_total"},
            ],
            "sort_by": "sales_total",
            "ascending": False,
        },
    )
    rows = rows_of(out)
    ref_sales = ss.groupby("Region")["Sales"].sum().to_dict()
    ref_profit = ss.groupby("Region")["Profit"].sum().to_dict()
    assert len(rows) == len(ref_sales)
    for r in rows:
        assert r["Region"] is not None
        near(r["sales_total"], ref_sales[r["Region"]], 1e-6)
        near(r["profit_total"], ref_profit[r["Region"]], 1e-6)
    totals = [r["sales_total"] for r in rows]
    assert totals == sorted(totals, reverse=True)


def check_group_aggregate_telco_filtered():
    out = tk.execute_operation(
        "telco",
        "group_aggregate",
        {
            "group_by": "Contract",
            "metrics": [
                {"column": "MonthlyCharges", "agg": "mean", "alias": "avg_charges"},
                {"agg": "count", "alias": "rows"},
            ],
            "filters": [{"column": "Churn", "op": "eq", "value": "Yes"}],
        },
    )
    rows = rows_of(out)
    sel = tel[tel["Churn"] == "Yes"]
    ref = sel.groupby("Contract")["MonthlyCharges"].mean().to_dict()
    ref_n = sel.groupby("Contract").size().to_dict()
    assert len(rows) == len(ref)
    for r in rows:
        near(r["avg_charges"], ref[r["Contract"]], 1e-6)
        assert r["rows"] == ref_n[r["Contract"]]


def check_group_aggregate_penguins_na():
    out = tk.execute_operation(
        "penguins",
        "group_aggregate",
        {
            "group_by": "species",
            "metrics": [
                {"column": "body_mass_g", "agg": "mean", "alias": "avg_mass"},
                {"column": "body_mass_g", "agg": "count", "alias": "n"},
            ],
        },
    )
    rows = rows_of(out)
    ref = pg.groupby("species")["body_mass_g"].mean().to_dict()
    ref_n = pg.groupby("species").size().to_dict()
    assert len(rows) == 3
    total = 0
    for r in rows:
        near(r["avg_mass"], ref[r["species"]], 1e-6)
        assert r["n"] == ref_n[r["species"]], (r, ref_n)
        total += r["n"]
    assert total == len(pg)


def check_correlation_superstore():
    cols = ["Sales", "Quantity", "Discount", "Profit"]
    out = tk.execute_operation("superstore", "correlation", {"columns": cols})
    ref = ss[cols].corr().stack()
    for (a, b), val in ref.items():
        if a != b:
            find_number(out, val, 5e-5)


def check_correlation_penguins_spearman():
    cols = ["bill_length_mm", "body_mass_g"]
    out = tk.execute_operation(
        "penguins", "correlation", {"columns": cols, "method": "spearman"}
    )
    ref = pg[cols].corr(method="spearman").iloc[0, 1]
    find_number(out, ref, 1e-6)


def check_statistical_welch():
    pair = pg[["sex", "flipper_length_mm"]].dropna()
    groups = {k: v["flipper_length_mm"].to_numpy() for k, v in pair.groupby("sex")}
    ref = sps.ttest_ind(*groups.values(), equal_var=False)
    out = tk.execute_operation(
        "penguins",
        "statistical_test",
        {"column_a": "sex", "column_b": "flipper_length_mm"},
    )
    c = calcs(out)
    assert "welch" in str(c["test"]).lower(), c["test"]
    near(abs(c["statistic"]), abs(ref.statistic), 1e-6)
    near(c["p_value"], ref.pvalue, 1e-6)


def check_statistical_chi_square():
    pair = pg[["species", "island"]].dropna()
    table = pd.crosstab(pair["species"], pair["island"]).to_numpy()
    ref = sps.chi2_contingency(table)
    out = tk.execute_operation(
        "penguins", "statistical_test", {"column_a": "species", "column_b": "island"}
    )
    c = calcs(out)
    near(c["statistic"], ref.statistic, 1e-6)


def check_statistical_correlation_pair():
    pair = pg[["bill_length_mm", "flipper_length_mm"]].dropna()
    ref = sps.pearsonr(pair["bill_length_mm"], pair["flipper_length_mm"])
    out = tk.execute_operation(
        "penguins",
        "statistical_test",
        {"column_a": "bill_length_mm", "column_b": "flipper_length_mm"},
    )
    c = calcs(out)
    near(c["statistic"], ref.statistic, 1e-6)


def check_anova_penguins():
    pair = pg[["species", "body_mass_g"]].dropna()
    groups = [v["body_mass_g"].to_numpy() for _, v in pair.groupby("species")]
    ref = sps.f_oneway(*groups)
    out = tk.execute_operation(
        "penguins", "anova_test", {"group_column": "species", "value_column": "body_mass_g"}
    )
    c = calcs(out)
    near(c["statistic"], ref.statistic, 1e-6)
    near(c["p_value"], ref.pvalue, 1e-6)
    rows = rows_of(out)
    assert len(rows) == 3
    ref_mean = pair.groupby("species")["body_mass_g"].mean().to_dict()
    for r in rows:
        near(r["mean"], ref_mean[r["species"]], 1e-6)


def check_anova_superstore():
    pair = ss[["Category", "Profit"]].dropna()
    groups = [v["Profit"].to_numpy() for _, v in pair.groupby("Category")]
    ref = sps.f_oneway(*groups)
    out = tk.execute_operation(
        "superstore", "anova_test", {"group_column": "Category", "value_column": "Profit"}
    )
    near(calcs(out)["statistic"], ref.statistic, 1e-6)


def check_regression_superstore():
    cols = ["Profit", "Sales", "Discount"]
    data_ = ss[cols].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data_)
    design = np.column_stack([np.ones(n), data_[["Sales", "Discount"]].to_numpy(float)])
    betas, _, _, _ = np.linalg.lstsq(design, data_["Profit"].to_numpy(float), rcond=None)
    fitted = design @ betas
    y = data_["Profit"].to_numpy(float)
    r2 = 1 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = tk.execute_operation(
        "superstore",
        "regression",
        {"target_column": "Profit", "feature_columns": ["Sales", "Discount"]},
    )
    rows = {r["term"]: r for r in rows_of(out)}
    near(rows["intercept"]["coefficient"], betas[0], 1e-6)
    near(rows["Sales"]["coefficient"], betas[1], 1e-6)
    near(rows["Discount"]["coefficient"], betas[2], 1e-6)
    near(calcs(out)["r_squared"], r2, 2e-6)


def check_isotonic_ecommerce():
    out = tk.execute_operation(
        "ecommerce",
        "isotonic_calibration",
        {"score_column": "lead_score", "outcome_column": "converted"},
    )
    c = calcs(out)
    near(c["brier_raw"], GT["ecommerce.csv"]["lead_score_brier_raw"], 5e-4)
    near(c["ece"], GT["ecommerce.csv"]["lead_score_ece"], 1e-3)
    rows = rows_of(out)
    assert len(rows) >= 5
    base = ecm["converted"].mean()
    near(c["base_rate"], base, 1e-6)


def check_sql_qualify_superstore():
    out = tk.run_sql(
        "SELECT Region, subcat, profit FROM ("
        'SELECT Region, "Sub-Category" AS subcat, SUM(Profit) AS profit '
        "FROM superstore GROUP BY 1, 2) "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY Region ORDER BY profit DESC) = 1 "
        "ORDER BY Region"
    )
    rows = rows_of(out)
    ref = ss.groupby(["Region", "Sub-Category"])["Profit"].sum().reset_index()
    best = ref.loc[ref.groupby("Region")["Profit"].idxmax()]
    assert len(rows) == len(best)
    for r in rows:
        match = best[best["Region"] == r["Region"]].iloc[0]
        assert r["subcat"] == match["Sub-Category"], (r, match.to_dict())
        near(r["profit"], match["Profit"], 1e-6)


def check_sql_cte_join_temps():
    out = tk.run_sql(
        "WITH daily AS (SELECT CAST(Date AS DATE) AS d, Temp FROM temps), "
        "monthly AS (SELECT date_trunc('month', d) AS m, AVG(Temp) AS avg_t "
        "FROM daily GROUP BY 1) "
        "SELECT d, Temp, ROUND(Temp - avg_t, 2) AS anomaly FROM daily "
        "JOIN monthly ON date_trunc('month', d) = m "
        "ORDER BY Temp - avg_t DESC LIMIT 5"
    )
    rows = rows_of(out)
    assert len(rows) == 5
    anomalies = [r["anomaly"] for r in rows]
    assert anomalies == sorted(anomalies, reverse=True)


def check_sql_sandbox_blocks():
    raises(lambda: tk.run_sql("DROP TABLE superstore"), "SELECT")
    raises(lambda: tk.run_sql("CREATE TABLE x AS SELECT 1"), "SELECT")
    raises(
        lambda: tk.run_sql("SELECT * FROM read_csv_auto('data/superstore.csv')"),
        "",
    )


def check_sql_table_restriction():
    raises(
        lambda: tk.run_sql("SELECT * FROM penguins LIMIT 1", tables=["superstore"]),
        "penguins",
    )


def check_distribution():
    out = tk.execute_operation("temps", "distribution", {"column": "Temp", "bins": 20})
    table = out["table"]
    counts = [row[-1] for row in table["rows"]]
    assert sum(counts) == len(tmp), (sum(counts), len(tmp))


def check_outliers_iqr_superstore():
    out = tk.execute_operation("superstore", "detect_outliers", {"column": "Profit"})
    valid = ss["Profit"].astype(float).dropna()
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    ref = int(((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)).sum())
    assert calcs(out)["outlier_rows"] == ref, (calcs(out)["outlier_rows"], ref)


def check_outliers_zscore_telco():
    out = tk.execute_operation(
        "telco", "detect_outliers", {"column": "MonthlyCharges", "method": "zscore"}
    )
    valid = tel["MonthlyCharges"].astype(float).dropna()
    z = ((valid - valid.mean()) / valid.std()).abs()
    ref = int((z > 3.0).sum())
    assert calcs(out)["outlier_rows"] == ref, (calcs(out)["outlier_rows"], ref)


def check_trend_temps_monthly():
    out = tk.execute_operation(
        "temps", "trend", {"date_column": "Date", "metric_column": "Temp", "freq": "month", "agg": "mean"}
    )
    rows = rows_of(out)
    dates = pd.to_datetime(tmp["Date"])
    ref = tmp.groupby(dates.dt.to_period("M"))["Temp"].mean()
    assert out["table"]["total_rows"] == len(ref), (out["table"]["total_rows"], len(ref))
    assert rows[0]["period"] == str(ref.index[0])
    assert calcs(out)["buckets"] == len(ref)


def check_trend_superstore_string_dates():
    out = tk.execute_operation(
        "superstore", "trend", {"date_column": "Order Date", "metric_column": "Sales", "freq": "month"}
    )
    rows = rows_of(out)
    dates = pd.to_datetime(ss["Order Date"], errors="coerce")
    ref = ss.groupby(dates.dt.to_period("M"))["Sales"].sum()
    assert len(rows) == len(ref), (len(rows), len(ref))
    assert rows[0]["period"] == str(ref.index[0])
    total = sum(r["sum_Sales"] for r in rows)
    near(total, ss["Sales"].sum(), 0.01)


def check_compare_subsets_superstore():
    out = tk.execute_operation(
        "superstore",
        "compare_subsets",
        {
            "metric": {"column": "Profit", "agg": "sum", "alias": "profit"},
            "subset_a": [{"column": "Region", "op": "eq", "value": "West"}],
            "subset_b": [{"column": "Region", "op": "eq", "value": "East"}],
            "label_a": "West",
            "label_b": "East",
        },
    )
    ref_w = ss.loc[ss["Region"] == "West", "Profit"].sum()
    ref_e = ss.loc[ss["Region"] == "East", "Profit"].sum()
    find_number(out, ref_w, 1e-6)
    find_number(out, ref_e, 1e-6)


def check_filter_rows():
    out = tk.execute_operation(
        "telco", "filter_rows", {"column": "SeniorCitizen", "op": "eq", "value": 1}
    )
    ref = int((tel["SeniorCitizen"].astype(str) == "1").sum())
    assert calcs(out)["rows_matching"] == ref, (calcs(out)["rows_matching"], ref)
    out2 = tk.execute_operation(
        "superstore", "filter_rows", {"column": "Category", "op": "contains", "value": "Technology"}
    )
    ref2 = int(ss["Category"].str.contains("Technology", na=False).sum())
    assert calcs(out2)["rows_matching"] == ref2


for label, fn in [
    ("summarize", check_summarize),
    ("value_counts superstore Category", check_value_counts_category),
    ("value_counts other_after folding", check_value_counts_other_after),
    ("value_counts normalize penguins", check_value_counts_normalized),
    ("top_n superstore", check_top_n),
    ("group_aggregate superstore", check_group_aggregate_superstore),
    ("group_aggregate telco filtered", check_group_aggregate_telco_filtered),
    ("group_aggregate penguins NaN", check_group_aggregate_penguins_na),
    ("correlation superstore", check_correlation_superstore),
    ("correlation spearman penguins", check_correlation_penguins_spearman),
    ("statistical_test welch penguins", check_statistical_welch),
    ("statistical_test chi-square penguins", check_statistical_chi_square),
    ("statistical_test pearson penguins", check_statistical_correlation_pair),
    ("anova_test penguins", check_anova_penguins),
    ("anova_test superstore", check_anova_superstore),
    ("regression superstore", check_regression_superstore),
    ("isotonic_calibration ecommerce vs ground truth", check_isotonic_ecommerce),
    ("sql_query QUALIFY superstore", check_sql_qualify_superstore),
    ("sql_query CTE join temps", check_sql_cte_join_temps),
    ("sql sandbox blocks writes + file access", check_sql_sandbox_blocks),
    ("sql table restriction enforced", check_sql_table_restriction),
    ("distribution temps", check_distribution),
    ("detect_outliers iqr superstore", check_outliers_iqr_superstore),
    ("detect_outliers zscore telco", check_outliers_zscore_telco),
    ("trend temps monthly", check_trend_temps_monthly),
    ("trend superstore string dates", check_trend_superstore_string_dates),
    ("compare_subsets superstore", check_compare_subsets_superstore),
    ("filter_rows telco + superstore", check_filter_rows),
]:
    smoke(label, fn)


# ---------------------------------------------------------------- ground truth (generated data)
def check_gt_sales_revenue():
    out = tk.execute_operation(
        "sales",
        "group_aggregate",
        {"group_by": "region", "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}]},
    )
    m = {r["region"]: r["total"] for r in rows_of(out)}
    for region, val in GT["sales.csv"]["revenue_by_region"].items():
        near(m[region], val, 0.02)


def check_gt_sales_mean():
    out = tk.execute_operation("sales", "summarize", {"columns": ["revenue"]})
    find_number(out, GT["sales.csv"]["mean_order_revenue"], 1e-4)


def check_gt_sales_outliers():
    out = tk.execute_operation("sales", "detect_outliers", {"column": "revenue"})
    assert calcs(out)["outlier_rows"] == GT["sales.csv"]["outlier_rows_iqr"]


def check_gt_support_channels():
    out = tk.execute_operation(
        "support",
        "group_aggregate",
        {
            "group_by": "channel",
            "metrics": [
                {"column": "handle_time_min", "agg": "mean", "alias": "avg_handle"},
                {"agg": "count", "alias": "n"},
            ],
        },
    )
    for r in rows_of(out):
        ch = r["channel"]
        near(r["avg_handle"], GT["support.csv"]["handle_time_mean_by_channel"][ch], 1e-4)
        assert r["n"] == GT["support.csv"]["tickets_by_channel"][ch]


def check_gt_ecommerce_conversion():
    out = tk.execute_operation(
        "ecommerce",
        "group_aggregate",
        {"group_by": "device", "metrics": [{"column": "converted", "agg": "mean", "alias": "rate"}]},
    )
    for r in rows_of(out):
        near(r["rate"], GT["ecommerce.csv"]["conversion_rate_by_device"][r["device"]], 1e-6)


def check_gt_retail_timed():
    t0 = time.perf_counter()
    out = tk.execute_operation(
        "retail250k",
        "group_aggregate",
        {"group_by": "region", "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}]},
    )
    dur = time.perf_counter() - t0
    print(f"INFO  retail250k group_aggregate: {dur:.2f}s")
    assert dur < 60
    m = {r["region"]: r["total"] for r in rows_of(out)}
    for region, val in GT["retail_250k.csv"]["revenue_by_region"].items():
        near(m[region], val, 0.05)


for label, fn in [
    ("ground truth sales.csv revenue by region", check_gt_sales_revenue),
    ("ground truth sales.csv mean order revenue", check_gt_sales_mean),
    ("ground truth sales.csv IQR outliers", check_gt_sales_outliers),
    ("ground truth support.csv channel metrics", check_gt_support_channels),
    ("ground truth ecommerce conversion by device", check_gt_ecommerce_conversion),
    ("ground truth retail_250k revenue by region (timed)", check_gt_retail_timed),
]:
    smoke(label, fn)


# ---------------------------------------------------------------- DAG plans
def check_dag_superstore():
    plan = {
        "objective": "discount means by category excluding Central",
        "steps": [
            {"step_id": 0, "operation": "filter_rows",
             "params": {"column": "Region", "op": "ne", "value": "Central"}},
            {"step_id": 1, "operation": "group_aggregate", "input_step": 0,
             "params": {"group_by": "Category",
                        "metrics": [{"column": "Discount", "agg": "mean", "alias": "mean_discount"}]}},
            {"step_id": 2, "operation": "top_n", "input_step": 1,
             "params": {"sort_by": "mean_discount", "n": 2,
                        "columns": ["Category", "mean_discount"]}},
        ],
        "charts": [],
    }
    out = tk.run_plan("superstore", plan)
    assert out["success"] is True, out["failed_messages"]
    filtered = ss[ss["Region"] != "Central"]
    ref = filtered.groupby("Category")["Discount"].mean().nlargest(2)
    rows = out["steps"][2]["table"]["rows"]
    assert len(rows) == 2
    for (cat, val), (ref_cat, ref_val) in zip(rows, ref.items()):
        assert cat == ref_cat, (rows, ref)
        near(val, ref_val, 1e-6)


def check_dag_sql_then_isotonic():
    plan = {
        "objective": "calibrate churn vs charges",
        "steps": [
            {"step_id": 0, "operation": "sql_query",
             "params": {"query": "SELECT MonthlyCharges / 100.0 AS score, "
                                 "CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS outcome "
                                 "FROM telco"}},
            {"step_id": 1, "operation": "isotonic_calibration", "input_step": 0,
             "params": {"score_column": "score", "outcome_column": "outcome"}},
        ],
        "charts": [],
    }
    out = tk.run_plan("telco", plan)
    assert out["success"] is True, out["failed_messages"]
    step1 = out["steps"][1]
    assert step1["success"] is True
    labels = {c["label"]: c["value"] for c in step1["calculations"]}
    assert labels.get("brier_raw") is not None
    assert step1["table"] is not None


def check_dag_invalid_chain():
    bad = {
        "objective": "chain from non-table op",
        "steps": [
            {"step_id": 0, "operation": "statistical_test",
             "params": {"column_a": "revenue", "column_b": "units"}},
            {"step_id": 1, "operation": "top_n", "input_step": 0,
             "params": {"sort_by": "units"}},
        ],
    }
    raises(lambda: tk.run_plan("sales", bad), "produces no table")

    self_ref = {
        "objective": "self reference",
        "steps": [
            {"step_id": 0, "operation": "group_aggregate",
             "params": {"group_by": "region",
                        "metrics": [{"agg": "count"}]}},
            {"step_id": 1, "operation": "top_n", "input_step": 1,
             "params": {"sort_by": "row_count"}},
        ],
    }
    raises(lambda: tk.run_plan("sales", self_ref), "itself")

    forward = {
        "objective": "forward reference",
        "steps": [
            {"step_id": 0, "operation": "top_n", "input_step": 1,
             "params": {"sort_by": "revenue"}},
            {"step_id": 1, "operation": "group_aggregate",
             "params": {"group_by": "region",
                        "metrics": [{"agg": "count"}]}},
        ],
    }
    raises(lambda: tk.run_plan("sales", forward), "later in execution order")


for label, fn in [
    ("DAG superstore filter->group->top_n", check_dag_superstore),
    ("DAG telco sql_query->isotonic_calibration", check_dag_sql_then_isotonic),
    ("DAG invalid chains rejected", check_dag_invalid_chain),
]:
    smoke(label, fn)


# ---------------------------------------------------------------- verify_claims
def check_verify_grounded_fabricated():
    out = tk.execute_operation(
        "superstore",
        "group_aggregate",
        {"group_by": "Region",
         "metrics": [{"column": "Sales", "agg": "sum", "alias": "sales_total"}],
         "sort_by": "sales_total", "ascending": False},
    )
    top = rows_of(out)[0]
    exact = str(top["sales_total"])
    good = tk.verify_claims(f"{top['Region']} led with {exact} in sales.", run_id=out["run_id"])
    assert good["grounded"] is True and good["verified"] is True, good
    bad = tk.verify_claims(
        f"{top['Region']} led with {top['sales_total'] + 12345.678} in sales.",
        run_id=out["run_id"],
    )
    assert bad["grounded"] is False and bad["ungrounded_numbers"], bad


def check_verify_percent_escalation():
    out = tk.execute_operation(
        "ecommerce", "value_counts", {"column": "converted", "normalize": True}
    )
    table = out["table"]
    m = {str(r[0]): r[table["columns"].index("proportion")] for r in table["rows"]}
    p = m.get("1")
    assert p is not None, m
    answer = f"{round(float(p) * 100, 4)}% of sessions converted."
    ok = tk.verify_claims(answer, run_id=out["run_id"])
    assert ok["grounded"] is True, (answer, p, ok)


def check_verify_caller_evidence():
    mean_temp = float(tmp["Temp"].mean())
    fresh = AnalysisToolkit()
    out = fresh.verify_claims(
        f"Mean daily temperature was {mean_temp}.",
        evidence_numbers=[mean_temp],
        evidence_notes="computed with pandas",
    )
    assert out["grounded"] is True and out["run_id"] is None
    bad = fresh.verify_claims("Mean daily temperature was 99.99.", evidence_numbers=[mean_temp])
    assert bad["grounded"] is False


def check_verify_run_id_targeting():
    first = tk.execute_operation("penguins", "value_counts", {"column": "island"})
    tk.execute_operation("penguins", "value_counts", {"column": "sex"})
    m = vc_map(first)
    answer = f"Biscoe appears {m['Biscoe']} times."
    ok = tk.verify_claims(answer, run_id=first["run_id"])
    assert ok["grounded"] is True and ok["run_id"] == first["run_id"], ok


def check_verify_tampered():
    out = tk.execute_operation("penguins", "value_counts", {"column": "island"})
    record = tk._runs[out["run_id"]]
    record.execution.steps[0].output.table.rows[0][-1] = 123456
    report = tk.verify_claims("The analysis completed.", run_id=out["run_id"])
    assert report["verified"] is False, report
    assert any(i["severity"] == "error" for i in report["verification_issues"])


def check_verify_error_paths():
    fresh = AnalysisToolkit()
    raises(lambda: fresh.verify_claims("nothing computed"), "nothing to verify")
    raises(lambda: tk.verify_claims("x", run_id="does-not-exist"), "unknown run_id")


def check_unknown_column_error():
    raises(
        lambda: tk.execute_operation("sales", "value_counts", {"column": "nope"}),
        "unknown column",
    )


for label, fn in [
    ("verify grounded + fabricated", check_verify_grounded_fabricated),
    ("verify percent escalation", check_verify_percent_escalation),
    ("verify caller evidence", check_verify_caller_evidence),
    ("verify explicit run_id targeting", check_verify_run_id_targeting),
    ("verify tampered execution flagged", check_verify_tampered),
    ("verify error paths", check_verify_error_paths),
    ("toolkit unknown column rejected", check_unknown_column_error),
]:
    smoke(label, fn)


# ---------------------------------------------------------------- MCP end-to-end
async def _mcp_flow():
    from mcp.shared.memory import create_connected_server_and_client_session

    from insight.mcp_server import create_server

    server = create_server(AnalysisToolkit())

    async def call(session, name, arguments=None):
        result = await session.call_tool(name, arguments or {})
        return json.loads(result.content[0].text)

    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        assert {t.name for t in tools.tools} == {
            "list_operations", "load_table", "list_tables", "profile_table",
            "execute_operation", "run_plan", "run_sql", "verify_claims",
        }
        loaded = await call(
            session, "load_table",
            {"name": "penguins", "path": data(r"data\penguins.csv")},
        )
        assert loaded["loaded"][0]["rows"] == 344, loaded
        assert len((await call(session, "list_tables"))["tables"]) == 1

        vc = await call(
            session, "execute_operation",
            {"table": "penguins", "operation": "value_counts", "params": {"column": "island"}},
        )
        assert vc_map(vc)["Biscoe"] == 168, vc

        ok = await call(session, "verify_claims", {"answer": "Biscoe hosts 168 penguins."})
        assert ok["grounded"] is True and ok["verified"] is True, ok

        sql = await call(session, "run_sql", {"sql": "SELECT COUNT(*) AS n FROM penguins"})
        assert sql["table"]["rows"][0][0] == 344, sql

        err = await call(session, "run_sql", {"sql": "DROP TABLE penguins"})
        assert "error" in err, err
        err2 = await call(
            session, "execute_operation",
            {"table": "ghost", "operation": "value_counts", "params": {"column": "x"}},
        )
        assert "error" in err2 and "ghost" in err2["error"], err2


def check_mcp_end_to_end():
    asyncio.run(_mcp_flow())


smoke("MCP in-memory end-to-end on penguins", check_mcp_end_to_end)

print("=" * 64)
print(f"TOTAL: {PASS} passed, {FAIL} failed")
for label, err in FAILURES:
    print(f"  FAILED: {label} :: {err}")
sys.exit(1 if FAIL else 0)
