import numpy as np
import pandas as pd
import pytest

from insight.agentic import suggest_followups
from insight.conversation.state import (
    AnalysisSessionState,
    record_result,
    serialize_state_for_planner,
)
from insight.profiling.profiler import profile_dataframe


@pytest.fixture
def profile():
    df = pd.DataFrame({
        "order_id": range(1, 61),
        "date": pd.date_range("2025-01-01", periods=60, freq="D").astype(str),
        "region": ["North", "South", "East", "West"] * 15,
        "channel": ["web", "store"] * 30,
        "revenue": [100.0 + (i % 17) * 3 for i in range(60)],
        "units": [(i % 5) + 1 for i in range(60)],
        "converted": [i % 2 for i in range(60)],
        "lead_score": [(i % 10) / 10.0 for i in range(60)],
    })
    return df, profile_dataframe(df, "orders.csv", content_hash="a" * 16)


def test_suggestions_cover_trend_outliers_and_association(profile):
    df, prof = profile
    out = suggest_followups(prof)
    assert out, "expected at least one suggestion"
    assert len(out) <= 3
    joined = " ".join(out).lower()
    assert "trend" in joined or "outlier" in joined or "associated" in joined


def test_suggestions_avoid_repeating_analyzed_work(profile):
    df, prof = profile
    state = AnalysisSessionState(
        recent_results=[
            {
                "question": "how does revenue trend over time",
                "operations": ["trend"],
                "key_facts": ["mean: 285.3"],
            }
        ]
    )
    out = suggest_followups(prof, state=state)
    assert not any("trend over time" in q.lower() for q in out)


def test_suggestions_prefer_join_when_tables_share_a_key(profile):
    df, prof = profile
    other = pd.DataFrame({"order_id": range(1, 20), "ship_days": np.arange(19)})
    out = suggest_followups(prof, tables={"orders.csv": df, "shipments.csv": other})
    assert out
    assert "join" in out[0].lower()


def test_suggestions_respect_cap(profile):
    df, prof = profile
    assert len(suggest_followups(prof, max_items=2)) <= 2


def test_record_result_digest_serialization():
    state = AnalysisSessionState(dataset_id="abc")
    record_result(
        state,
        question="Which region has the highest revenue?",
        operations=["group_aggregate"],
        key_facts=["groups: 4", "total_revenue: 48,805.62"],
    )
    record_result(state, question="Now break it down by category", operations=["sql_query"], key_facts=[])

    assert len(state.recent_results) == 2
    text = serialize_state_for_planner(state)
    assert "Which region has the highest revenue?" in text
    assert "48,805.62" in text
    assert "group_aggregate" in text
    assert "Now break it down by category" in text


def test_digest_is_capped():
    state = AnalysisSessionState()
    for i in range(6):
        record_result(state, question=f"q{i}", operations=["summarize"], key_facts=[])
    assert len(state.recent_results) == 3
    assert state.recent_results[-1].question == "q5"


def test_index_artifact_columns_are_quarantined():
    from insight.profiling.profiler import is_artifact_column

    df = pd.DataFrame({
        "Unnamed: 0": range(10),
        "real": np.arange(10.0),
        "index": range(10),
    })
    assert is_artifact_column("Unnamed: 0", df["Unnamed: 0"])
    assert is_artifact_column("index", df["index"])
    assert not is_artifact_column("real", df["real"])

    prof = profile_dataframe(df, "t.csv", content_hash="b" * 16)
    assert "Unnamed: 0" in prof.artifact_columns
    assert [c.name for c in prof.columns] == ["real"]


def test_birthdate_never_suggests_a_trend():
    df = pd.DataFrame({
        "patient_id": range(1, 41),
        "birthdate": pd.date_range("1980-01-01", periods=40, freq="365D").astype(str),
        "weight": [60.0 + (i % 7) for i in range(40)],
    })
    prof = profile_dataframe(df, "patients.csv", content_hash="c" * 16)
    out = suggest_followups(prof)
    assert not any("trend" in q.lower() for q in out)


def test_suggestions_skip_identifier_like_metrics():
    df = pd.DataFrame({
        "patient_id": range(1, 41),
        "zip_code": [10001 + i for i in range(40)],
        "weight": [60.0 + (i % 7) for i in range(40)],
    })
    prof = profile_dataframe(df, "patients.csv", content_hash="d" * 16)
    out = suggest_followups(prof)
    joined = " ".join(out).lower()
    assert "patient_id" not in joined and "zip_code" not in joined
