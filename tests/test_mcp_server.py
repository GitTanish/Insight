import json
import logging

import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from insight.mcp_server import create_server
from insight.toolkit import AnalysisToolkit

logging.getLogger("mcp").setLevel(logging.WARNING)

EXPECTED_TOOLS = {
    "list_operations",
    "load_table",
    "list_tables",
    "profile_table",
    "execute_operation",
    "run_plan",
    "run_sql",
    "verify_claims",
}

SALES_CSV = (
    "region,revenue\nNorth,100\nSouth,40\nEast,60\nNorth,50\nWest,30\n"
)

GROUP_AGG = {
    "group_by": "region",
    "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
}

DAG_PLAN = {
    "objective": "top regions",
    "steps": [
        {"step_id": 0, "operation": "group_aggregate", "params": GROUP_AGG},
        {"step_id": 1, "operation": "top_n",
         "params": {"sort_by": "total", "n": 2, "columns": ["region", "total"]},
         "input_step": 0},
    ],
    "charts": [],
}


async def _call(session, name, arguments=None):
    result = await session.call_tool(name, arguments or {})
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_tool_surface_registered():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        tools = await session.list_tools()
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS
        descriptions = {t.name: (t.description or "") for t in tools.tools}
        assert "evidence" in descriptions["verify_claims"].lower()


@pytest.mark.asyncio
async def test_end_to_end_verify_flow():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        loaded = await _call(
            session, "load_table", {"name": "sales", "content": SALES_CSV}
        )
        assert loaded["loaded"][0]["rows"] == 5

        run = await _call(
            session,
            "execute_operation",
            {"table": "sales", "operation": "group_aggregate", "params": GROUP_AGG},
        )
        rows = run["table"]["rows"]
        assert ["North", 150] in [r[:2] for r in rows]
        assert run["run_id"]

        good = await _call(
            session,
            "verify_claims",
            {"answer": "North leads with 150 in revenue.", "run_id": run["run_id"]},
        )
        assert good["grounded"] is True
        assert good["verified"] is True

        bad = await _call(
            session, "verify_claims", {"answer": "North leads with 999 in revenue."}
        )
        assert bad["grounded"] is False
        assert "999" in bad["ungrounded_numbers"]


@pytest.mark.asyncio
async def test_error_payloads_are_structured():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        err = await _call(session, "load_table", {"name": "x"})
        assert "error" in err

        err = await _call(
            session,
            "execute_operation",
            {"table": "ghost", "operation": "value_counts",
             "params": {"column": "region"}},
        )
        assert "error" in err and "ghost" in err["error"]

        await _call(session, "load_table", {"name": "sales", "content": SALES_CSV})
        err = await _call(session, "run_sql", {"sql": "DROP TABLE sales"})
        assert "error" in err
        assert "SELECT" in err["error"]


@pytest.mark.asyncio
async def test_run_plan_tool_and_run_id_verification():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        await _call(session, "load_table", {"name": "sales", "content": SALES_CSV})
        out = await _call(session, "run_plan", {"table": "sales", "plan": DAG_PLAN})
        assert out["success"] is True
        assert [s["operation"] for s in out["steps"]] == ["group_aggregate", "top_n"]
        assert out["steps"][1]["table"]["rows"] == [["North", 150], ["East", 60]]

        good = await _call(
            session,
            "verify_claims",
            {"answer": "North (150) beats East (60).", "run_id": out["run_id"]},
        )
        assert good["grounded"] is True and good["verified"] is True


@pytest.mark.asyncio
async def test_verify_with_caller_evidence_numbers():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        err = await _call(session, "verify_claims", {"answer": "Growth was 12.5%."})
        assert "error" in err

        out = await _call(
            session,
            "verify_claims",
            {"answer": "Growth was 12.5%.", "evidence_numbers": [0.125]},
        )
        assert out["grounded"] is True and out["run_id"] is None


@pytest.mark.asyncio
async def test_list_tables_and_profile_and_operations():
    server = create_server(AnalysisToolkit())
    async with create_connected_server_and_client_session(server) as session:
        await _call(session, "load_table", {"name": "sales", "content": SALES_CSV})
        tables = await _call(session, "list_tables")
        assert tables["tables"][0]["columns"] == ["region", "revenue"]

        profile = await _call(session, "profile_table", {"name": "sales"})
        assert profile["row_count"] == 5
        assert profile["column_count"] == 2

        ops = await _call(session, "list_operations")
        assert "group_aggregate" in ops["names"]
        assert "group_aggregate" in ops["catalog"]


def test_main_rejects_unknown_transport(monkeypatch):
    from insight.mcp_server import main

    monkeypatch.setenv("INSIGHT_MCP_TRANSPORT", "carrier-pigeon")
    with pytest.raises(SystemExit):
        main()
