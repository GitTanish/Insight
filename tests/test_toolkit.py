import json

import pandas as pd
import pytest

from insight.analytics.operations import OPERATIONS
from insight.toolkit import AnalysisToolkit, ToolkitError

SALES_CSV = (
    "region,revenue,units\n"
    "North,100,2\nSouth,40,1\nEast,60,3\nNorth,50,2\nWest,30,1\n"
)


@pytest.fixture
def tk():
    toolkit = AnalysisToolkit()
    toolkit.load_table("sales", content=SALES_CSV)
    return toolkit


def _group_agg(tk, table="sales"):
    return tk.execute_operation(
        table,
        "group_aggregate",
        {"group_by": "region", "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}]},
    )


def test_load_table_and_list_tables():
    tk = AnalysisToolkit()
    out = tk.load_table("sales", content=SALES_CSV)
    assert out["loaded"] == [
        {"name": "sales", "rows": 5, "columns": ["region", "revenue", "units"]}
    ]
    assert tk.list_tables()["tables"][0]["rows"] == 5


def test_load_table_sniffs_semicolon_and_tab():
    tk = AnalysisToolkit()
    tk.load_table("semi", content="a;b\n1;2\n")
    tk.load_table("tabby", content="a\tb\n1\t2\n")
    assert tk.list_tables()["tables"][0]["columns"] == ["a", "b"]
    assert tk.list_tables()["tables"][1]["columns"] == ["a", "b"]


def test_load_table_from_path_and_excel_sheets(tmp_path):
    csv_path = tmp_path / "orders.csv"
    csv_path.write_text("order_id,amount\n1,10\n2,20\n", encoding="utf-8")
    xlsx_path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"b": [3]}).to_excel(writer, sheet_name="second", index=False)

    tk = AnalysisToolkit()
    assert tk.load_table("orders", path=str(csv_path))["loaded"][0]["rows"] == 2
    loaded = tk.load_table("book", path=str(xlsx_path))["loaded"]
    names = {entry["name"] for entry in loaded}
    assert names == {"book_first", "book_second"}


def test_load_table_source_validation(tmp_path):
    tk = AnalysisToolkit()
    with pytest.raises(ToolkitError):
        tk.load_table("x")
    with pytest.raises(ToolkitError):
        tk.load_table("x", content="a\n1\n", path=str(tmp_path / "nope.csv"))
    with pytest.raises(ToolkitError):
        tk.load_table("x", path=str(tmp_path / "missing.csv"))


def test_profile_table(tk):
    profile = tk.profile_table("sales")
    assert profile["row_count"] == 5
    assert profile["column_count"] == 3
    assert {c["name"] for c in profile["columns"]} == {"region", "revenue", "units"}


def test_list_operations_matches_registry(tk):
    out = tk.list_operations()
    assert set(out["names"]) == set(OPERATIONS)
    assert "group_aggregate" in out["catalog"]


def test_execute_operation_group_aggregate(tk):
    out = _group_agg(tk)
    rows = dict(zip(out["table"]["columns"], zip(*out["table"]["rows"])))
    totals = dict(zip(rows["region"], rows["total"]))
    assert totals == {"North": 150, "South": 40, "East": 60, "West": 30}
    assert out["run_id"]


def test_execute_operation_filter_rows_mutates_without_table(tk):
    out = tk.execute_operation(
        "sales", "filter_rows", {"column": "region", "op": "eq", "value": "North"}
    )
    assert out["table"] is None
    calcs = {c["label"]: c["value"] for c in out["calculations"]}
    assert calcs["rows_matching"] == 2


def test_execute_operation_rejects_unknown_column(tk):
    with pytest.raises(ToolkitError) as excinfo:
        tk.execute_operation(
            "sales", "value_counts", {"column": "nope"}
        )
    assert "unknown column" in str(excinfo.value)


def test_execute_operation_rejects_unknown_table_and_operation(tk):
    with pytest.raises(ToolkitError):
        tk.execute_operation("ghost", "value_counts", {"column": "region"})
    with pytest.raises(ToolkitError):
        tk.execute_operation("sales", "not_an_op", {})


def test_run_sql_join_across_loaded_tables():
    tk = AnalysisToolkit()
    tk.load_table("orders", content="customer_id,amount\nc1,100\nc1,40\nc2,60\n")
    tk.load_table("customers", content="customer_id,region\nc1,west\nc2,east\n")
    out = tk.run_sql(
        "SELECT c.region, SUM(o.amount) AS total FROM orders o "
        "JOIN customers c USING (customer_id) GROUP BY c.region ORDER BY total DESC"
    )
    rows = dict(zip(out["table"]["columns"], zip(*out["table"]["rows"])))
    assert rows["region"] == ("west", "east")
    assert rows["total"] == (140.0, 60.0)


def test_run_sql_sandbox_blocks_writes(tk):
    with pytest.raises(ToolkitError) as excinfo:
        tk.run_sql("DROP TABLE sales")
    assert "SELECT" in str(excinfo.value)


def test_run_sql_unknown_table_selection(tk):
    with pytest.raises(ToolkitError):
        tk.run_sql("SELECT * FROM sales", tables=["ghost"])


DAG_PLAN = {
    "objective": "top regions after filtering",
    "steps": [
        {"step_id": 0, "operation": "filter_rows",
         "params": {"column": "region", "op": "ne", "value": "West"}},
        {"step_id": 1, "operation": "group_aggregate",
         "params": {"group_by": "region",
                    "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}]}},
        {"step_id": 2, "operation": "top_n",
         "params": {"sort_by": "total", "n": 1, "columns": ["region", "total"]},
         "input_step": 1},
    ],
    "charts": [],
}


def test_run_plan_dag(tk):
    out = tk.run_plan("sales", DAG_PLAN)
    assert out["success"] is True
    assert [s["operation"] for s in out["steps"]] == [
        "filter_rows", "group_aggregate", "top_n",
    ]
    top = out["steps"][2]["table"]
    assert top["rows"] == [["North", 150]]


def test_run_plan_accepts_json_string(tk):
    out = tk.run_plan("sales", json.dumps(DAG_PLAN))
    assert out["success"] is True


def test_run_plan_chaining_from_filter_rows_is_legal(tk):
    plan = {
        "objective": "chained filter",
        "steps": [
            {"step_id": 0, "operation": "filter_rows",
             "params": {"column": "region", "op": "eq", "value": "North"}},
            {"step_id": 1, "operation": "group_aggregate", "input_step": 0,
             "params": {"group_by": "region",
                        "metrics": [{"column": "revenue", "agg": "mean", "alias": "m"}]}},
        ],
        "charts": [],
    }
    out = tk.run_plan("sales", plan)
    assert out["success"] is True
    assert out["steps"][1]["table"]["rows"] == [["North", 75.0]]


def test_run_plan_rejects_invalid_plan(tk):
    bad = {
        "objective": "bad",
        "steps": [{"step_id": 0, "operation": "group_aggregate",
                   "params": {"group_by": "not_a_column",
                              "metrics": [{"agg": "count"}]}}],
    }
    with pytest.raises(ToolkitError) as excinfo:
        tk.run_plan("sales", bad)
    assert "unknown column" in str(excinfo.value)

    with pytest.raises(ToolkitError):
        tk.run_plan("sales", "{not json")
    with pytest.raises(ToolkitError):
        tk.run_plan("sales", {"objective": "no steps"})


def test_run_plan_execution_failure_is_returned_not_raised(tk):
    plan = {
        "objective": "bad regression",
        "steps": [{"step_id": 0, "operation": "regression",
                   "params": {"target_column": "region",
                              "feature_columns": ["revenue"]}}],
    }
    out = tk.run_plan("sales", plan)
    assert out["success"] is False
    assert out["failed_messages"]


def test_verify_claims_grounded_after_operation(tk):
    _group_agg(tk)
    ok = tk.verify_claims("North leads with 150 in revenue across 5 rows.")
    assert ok["grounded"] is True
    assert ok["ungrounded_numbers"] == []

    bad = tk.verify_claims("North leads with 999 in revenue.")
    assert bad["grounded"] is False
    assert "999" in bad["ungrounded_numbers"]


def test_verify_claims_catches_miscited_partial_sum(tk):
    _group_agg(tk)
    out = tk.verify_claims("North leaders earned 100 total.")
    assert out["grounded"] is False


def test_verify_claims_with_caller_evidence_only():
    tk = AnalysisToolkit()
    out = tk.verify_claims(
        "The campaign lifted revenue by 12.5%.",
        evidence_numbers=[0.125],
        evidence_notes="baseline 100.0, treated 112.5",
    )
    assert out["grounded"] is True
    assert out["run_id"] is None


def test_verify_claims_with_caller_evidence_tables():
    tk = AnalysisToolkit()
    out = tk.verify_claims(
        "Adjusted revenue was 77.",
        evidence_tables=[{"columns": ["metric", "value"], "rows": [["adjusted", 77]]}],
    )
    assert out["grounded"] is True
    bad = tk.verify_claims(
        "Adjusted revenue was 78.",
        evidence_tables=[{"columns": ["metric", "value"], "rows": [["adjusted", 77]]}],
    )
    assert bad["grounded"] is False


def test_verify_claims_without_evidence_raises():
    tk = AnalysisToolkit()
    with pytest.raises(ToolkitError):
        tk.verify_claims("nothing computed")


def test_verify_claims_unknown_run_id(tk):
    _group_agg(tk)
    with pytest.raises(ToolkitError):
        tk.verify_claims("North leads with 150.", run_id="nope")


def test_verify_claims_detects_tampered_execution(tk):
    out = _group_agg(tk)
    record = tk._runs[out["run_id"]]
    record.execution.steps[0].output.table.rows[0][1] = 123456
    report = tk.verify_claims("North leads with 150 in revenue.", run_id=out["run_id"])
    assert report["verified"] is False
    errors = [i for i in report["verification_issues"] if i["severity"] == "error"]
    assert errors and errors[0]["step_id"] == 0


def test_file_roots_restrict_path_loading(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    inside = allowed / "inside.csv"
    inside.write_text("a,b\n1,2\n", encoding="utf-8")
    outside = tmp_path / "outside.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")

    tk = AnalysisToolkit(file_roots=[allowed])
    assert tk.load_table("ok", path=str(inside))["loaded"][0]["rows"] == 1
    with pytest.raises(ToolkitError, match="outside the allowed roots"):
        tk.load_table("bad", path=str(outside))


def test_empty_file_roots_disables_path_loading(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a\n1\n", encoding="utf-8")

    tk = AnalysisToolkit(file_roots=[])
    with pytest.raises(ToolkitError, match="disabled"):
        tk.load_table("x", path=str(csv_path))


def test_max_file_bytes_cap(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("a,b\n" + "1,2\n" * 20, encoding="utf-8")

    tk = AnalysisToolkit(file_roots=[tmp_path], max_file_bytes=10)
    with pytest.raises(ToolkitError, match="byte limit"):
        tk.load_table("x", path=str(csv_path))


def test_file_roots_reject_symlink_escape(tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    link = root / "link.csv"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable on this platform")

    tk = AnalysisToolkit(file_roots=[root])
    with pytest.raises(ToolkitError):
        tk.load_table("x", path=str(link))
