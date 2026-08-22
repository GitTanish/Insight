import json
import os
from pathlib import Path

import pandas as pd
import pytest

from evaluation.run_eval import _numeric_found, load_cases, score_case

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation"
DATA_DIR = EVAL_DIR / "data"


class _StubTable:
    def __init__(self, columns, rows):
        self.columns = columns
        self.rows = rows


class _StubResponse:
    def __init__(self, answer="", steps=None, tables=None, evidence=None, error=None):
        self.answer = answer
        self.meta = {"steps": steps or []}
        self.tables = tables or []
        self.evidence = evidence or []
        self.error = error


def test_cases_file_is_valid_and_datasets_exist():
    cases = load_cases(EVAL_DIR / "cases.json")
    assert len(cases) >= 15
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    for case in cases:
        csv_path = DATA_DIR / case["dataset"]
        assert csv_path.exists(), f"missing dataset {case['dataset']}"
        df = pd.read_csv(csv_path)
        assert case["question"].strip()
        for spec in case.get("numeric_expect", []):
            assert "label" in spec and "approx" in spec
        assert len(df) > 100


def test_ground_truth_matches_baked_expectations():
    truth = json.loads((DATA_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    sales = truth["sales.csv"]
    by_region = sales["revenue_by_region"]
    assert max(by_region, key=by_region.get) == "West"
    support = truth["support.csv"]
    rates = support["resolution_rate_by_channel"]
    assert rates["chat"] - 0.10 > max(rates["email"], rates["phone"])
    ecom = truth["ecommerce.csv"]
    conv = ecom["conversion_rate_by_device"]
    assert conv["mobile"] > conv["desktop"] * 1.5


def test_score_case_ops_superset_with_anyof_groups():
    response = _StubResponse(
        steps=[
            {"operation": "filter_rows", "success": True},
            {"operation": "group_aggregate", "success": True},
            {"operation": "detect_outliers", "success": False},
        ]
    )
    ok, reasons = score_case(
        response,
        {"expect_ops": [["value_counts", "group_aggregate"], "group_aggregate"],
         "answer_contains": [], "numeric_expect": []},
    )
    assert ok and not reasons

    ok2, reasons2 = score_case(
        response,
        {"expect_ops": ["trend"], "answer_contains": [], "numeric_expect": []},
    )
    assert not ok2 and any("trend" in r for r in reasons2)


def test_score_case_numeric_lookup_paths():
    calc_evidence = type("E", (), {"kind": "calculation", "label": "outlier_rows", "value": "6"})()
    wide_table = _StubTable(["region", "total_revenue"], [["West", 48805.62], ["East", 100]])
    long_table = _StubTable(
        ["column", "metric", "value"], [["revenue", "mean", 285.31], ["revenue", "std", 60.0]]
    )

    found, _ = _numeric_found(
        _StubResponse(evidence=[calc_evidence]),
        {"label": ["outlier_rows"], "approx": 6, "tol_pct": 0.0},
    )
    assert found

    found2, _ = _numeric_found(
        _StubResponse(tables=[wide_table]),
        {"label": ["sum_revenue", "total_revenue"], "approx": 48805.62},
    )
    assert found2

    found3, fail3 = _numeric_found(
        _StubResponse(tables=[long_table]),
        {"label": "mean", "column": "revenue", "approx": 285.3069},
    )
    assert found3, fail3

    found4, _ = _numeric_found(
        _StubResponse(tables=[long_table]),
        {"label": "mean", "column": "handle_time_min", "approx": 13.3533},
    )
    assert not found4


def test_score_case_flags_pipeline_error():
    ok, reasons = score_case(_StubResponse(error="plan_validation_failed"), {})
    assert not ok and reasons and "plan" in reasons[0]


def _live_provider_available() -> bool:
    from insight.settings import get_settings

    settings = get_settings()
    return bool(
        settings.groq_api_key
        or settings.mistral_api_key
        or settings.openai_api_key
        or settings.anthropic_api_key
        or settings.openrouter_api_key
        or settings.custom_base_url
    )


@pytest.mark.eval
@pytest.mark.skipif(
    os.getenv("INSIGHT_RUN_EVAL", "") != "1",
    reason="set INSIGHT_RUN_EVAL=1 to run the live eval suite",
)
@pytest.mark.skipif(not _live_provider_available(), reason="no LLM provider key configured")
def test_live_eval_smoke():
    from evaluation.run_eval import run_eval

    passed, total, _ = run_eval(limit=2)
    assert total == 2
    assert passed == 2, "first two golden cases failed against the live provider"
