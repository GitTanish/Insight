from pathlib import Path

from insight.domain.query import AnalysisRequest
from insight.orchestrator import run_analysis_sync

from tests.conftest import EXPLANATION, VALID_PLAN


def test_full_pipeline_happy_path(stub_chain_factory, sales_df, profile, tmp_path):
    stub_chain_factory([VALID_PLAN, EXPLANATION])

    response = run_analysis_sync(
        sales_df,
        profile,
        AnalysisRequest(question="Which region has the highest revenue?"),
        artifacts_dir=tmp_path,
    )

    assert response.error is None
    assert response.answer.startswith("**West leads**")
    assert response.meta["model"] == "fake/fake-1"
    assert response.validation.valid

    tables = response.tables
    assert len(tables) == 1
    assert tables[0].columns == ["region", "total_revenue"]

    assert len(response.plots) == 1
    plot = response.plots[0]
    assert plot.title == "Revenue by Region"
    assert plot.path and Path(plot.path).exists()

    calc_labels = [e.label for e in response.evidence if e.kind == "calculation"]
    assert "groups" in calc_labels
    assert any(e.value for e in response.evidence if e.kind == "calculation")


def test_planning_failure_returns_graceful_response(
    stub_chain_factory, sales_df, profile, tmp_path
):
    stub_chain_factory(["no json {", "still no {"])

    response = run_analysis_sync(
        sales_df,
        profile,
        AnalysisRequest(question="anything"),
        artifacts_dir=tmp_path / "unused",
    )

    assert response.error == "plan_validation_failed"
    assert "could not construct a reliable analysis plan" in response.answer.lower()
    assert not (tmp_path / "unused").exists()


def test_repair_after_execution_failure(stub_chain_factory, sales_df, profile, tmp_path):
    broken_plan = """
    {
      "objective": "t",
      "steps": [
        {"step_id": 0, "operation": "group_aggregate",
         "params": {"group_by": "region",
                    "metrics": [{"column": "revenue", "agg": "sum"}],
                    "sort_by": "not_a_real_column"}}
      ]
    }
    """
    llm = stub_chain_factory([broken_plan.strip(), VALID_PLAN, EXPLANATION])

    response = run_analysis_sync(
        sales_df,
        profile,
        AnalysisRequest(question="revenue by region"),
        artifacts_dir=tmp_path,
    )

    assert response.error is None
    assert response.meta["repair_attempted"] is True
    assert response.validation.valid
    assert response.answer.startswith("**West leads**")
    assert len(llm.requests) == 3
