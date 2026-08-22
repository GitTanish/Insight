import numpy as np
import pandas as pd
import pytest

from insight.briefing import findings_to_dicts, generate_briefing


@pytest.fixture
def rich_df():
    rng = np.random.default_rng(42)
    n = 400
    x = rng.normal(0, 1, n)
    revenue = 100 + 25 * x + rng.normal(0, 2, n)

    df = pd.DataFrame({
        "order_id": [f"O{i:05d}" for i in range(n)],
        "date": pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "revenue": revenue,
        "signal": x,
        "spend": 3 * x * 4 + rng.normal(0, 1, n),
        "flag": rng.choice([0, 1], n, p=[0.93, 0.07]),
    })
    df.loc[10:18, "revenue"] = df.loc[10:18, "revenue"] + 900
    df["notes"] = [None if i % 3 == 0 else f"note {i}" for i in range(n)]
    return df


def test_briefing_finds_multiple_kinds(rich_df):
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(rich_df, "rich.csv")
    findings = generate_briefing(rich_df, profile)

    assert len(findings) >= 3
    categories = {f.category for f in findings}
    assert categories & {"outliers", "correlation", "quality"}
    assert all(f.suggested_query for f in findings)
    assert [f.rank for f in findings] == sorted(f.rank for f in findings)


def test_briefing_severity_ordering(rich_df):
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(rich_df, "rich.csv")
    findings = generate_briefing(rich_df, profile)
    severities = [f.severity for f in findings]
    assert severities == sorted(severities, reverse=True)


def test_briefing_empty_on_tiny_data():
    df = pd.DataFrame({"a": [1, 2, 3]})
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(df, "tiny.csv")
    assert generate_briefing(df, profile) == []


def test_briefing_dicts_serializable(rich_df):
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(rich_df, "rich.csv")
    payload = findings_to_dicts(generate_briefing(rich_df, profile))
    assert isinstance(payload, list)
    assert {"rank", "severity", "category", "title", "detail", "suggested_query"} <= set(payload[0])


def test_briefing_detects_duplicates():
    rows = [[1, "x"]] * 60 + [[2, "y"], [3, "z"]]
    df = pd.DataFrame(rows, columns=["a", "b"])
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(df, "dup.csv")
    findings = generate_briefing(df, profile)
    assert any(f.category == "quality" and "duplicate" in f.title.lower() for f in findings)


def test_briefing_is_deterministic(rich_df):
    from insight.profiling.profiler import profile_dataframe

    profile = profile_dataframe(rich_df, "rich.csv")
    first = [f.title for f in generate_briefing(rich_df, profile)]
    second = [f.title for f in generate_briefing(rich_df, profile)]
    assert first == second
