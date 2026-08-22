import pandas as pd

from insight.profiling.profiler import profile_dataframe


def make_profile(df):
    return profile_dataframe(df, "x.csv")


def test_roles_detected(sales_df):
    p = make_profile(sales_df)
    roles = {c.name: c.role.value for c in p.columns}
    assert roles["order_id"] == "identifier"
    assert roles["date"] == "datetime"
    assert roles["region"] == "categorical"
    assert roles["revenue"] == "numeric"
    assert roles["units"] == "numeric"


def test_unique_numeric_metric_not_identifier():
    df = pd.DataFrame({
        "revenue": range(100),
        "customer_id": [f"CUST{i:05d}" for i in range(100)],
    })
    roles = {c.name: c.role.value for c in make_profile(df).columns}
    assert roles["revenue"] == "numeric"
    assert roles["customer_id"] == "identifier"


def test_high_cardinality_text():
    df = pd.DataFrame({
        "note": [
            f"customer wrote something fairly long here number {i} with padding"
            for i in range(80)
        ],
    })
    roles = {c.name: c.role.value for c in make_profile(df).columns}
    assert roles["note"] == "text"


def test_missing_and_duplicates_counted():
    df = pd.DataFrame({
        "a": [1.0, None, 3.0, 3.0],
        "b": ["x", "y", "z", "z"],
    })
    p = make_profile(df)
    assert p.missing_cells == 1
    assert p.duplicate_rows == 1
    a_col = next(c for c in p.columns if c.name == "a")
    assert abs(a_col.missing_pct - 0.25) < 1e-9


def test_fingerprint_stable_per_content():
    df = pd.DataFrame({"a": [1, 2, 3]})
    first = make_profile(df).fingerprint
    second = make_profile(df).fingerprint
    assert first.dataset_id == second.dataset_id

    other = make_profile(pd.DataFrame({"a": [1, 2, 4]})).fingerprint
    assert other.dataset_id != first.dataset_id


def test_summary_for_planner_compact(sales_df):
    summary = make_profile(sales_df).summary_for_planner()
    assert "rows=60" in summary
    assert "revenue" in summary
    assert len(summary) < 2000
