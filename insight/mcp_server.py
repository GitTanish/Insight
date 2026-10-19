"""MCP server exposing the deterministic engine and claim verification.

Run it with::

    python -m insight.mcp_server                    # stdio (default)
    INSIGHT_MCP_TRANSPORT=streamable-http python -m insight.mcp_server

The server holds one :class:`~insight.toolkit.AnalysisToolkit` session: load
tables, run registered operations / plans / sandboxed SQL, then verify any
narrative's numbers against the computed evidence. No LLM calls are made by
any tool in this server.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from insight.toolkit import AnalysisToolkit, ToolkitError

INSTRUCTIONS = (
    "Deterministic analytics engine with independent verification. Typical flow: "
    "load_table → (profile_table / list_tables) → execute_operation, run_plan or "
    "run_sql → write your narrative → verify_claims to check every number in it "
    "against computed evidence. No LLM is involved; results are computed by "
    "pandas/scipy/sandboxed DuckDB. Prefer verify_claims before presenting any "
    "numbers to a user."
)


def create_server(toolkit: Optional[AnalysisToolkit] = None) -> FastMCP:
    engine = toolkit or AnalysisToolkit()
    server = FastMCP(
        "insight",
        instructions=INSTRUCTIONS,
        host=os.environ.get("INSIGHT_MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("INSIGHT_MCP_PORT", "8765") or 8765),
    )

    @server.tool()
    def list_operations() -> dict:
        """List every registered analysis operation with its parameter schema."""
        return engine.list_operations()

    @server.tool()
    def load_table(
        name: str,
        content: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """Register a table from inline CSV text (content) or a file path.

        Paths ending in .xlsx/.xls/.xlsm load one table per sheet. Provide
        exactly one of content or path. Re-loading a name replaces it.
        """
        try:
            return engine.load_table(name, content=content, path=path)
        except ToolkitError as exc:
            return {"error": str(exc)}

    @server.tool()
    def list_tables() -> dict:
        """List loaded tables with row counts and column names."""
        return engine.list_tables()

    @server.tool()
    def profile_table(name: str) -> dict:
        """Full column-level profile (types, missingness, cardinality, stats)."""
        try:
            return engine.profile_table(name)
        except ToolkitError as exc:
            return {"error": str(exc)}

    @server.tool()
    def execute_operation(
        table: str, operation: str, params: Optional[dict] = None
    ) -> dict:
        """Run one registered operation against a loaded table.

        Returns the result table, calculations and a run_id usable with
        verify_claims. Call list_operations for names and parameter schemas.
        """
        try:
            return engine.execute_operation(table, operation, params)
        except ToolkitError as exc:
            return {"error": str(exc)}

    @server.tool()
    def run_plan(table: str, plan: dict) -> dict:
        """Validate and execute a multi-step AnalysisPlan (DAG) on a table.

        Steps use operation/params/input_step like the planner schema; charts
        are accepted but not rendered. Returns per-step results plus a run_id.
        """
        try:
            return engine.run_plan(table, plan)
        except ToolkitError as exc:
            return {"error": str(exc)}

    @server.tool()
    def run_sql(sql: str, tables: Optional[list[str]] = None) -> dict:
        """Run one read-only SELECT/WITH statement in the DuckDB sandbox.

        Defaults to all loaded tables; pass tables to restrict visibility.
        """
        try:
            return engine.run_sql(sql, tables)
        except ToolkitError as exc:
            return {"error": str(exc)}

    @server.tool()
    def verify_claims(
        answer: str,
        run_id: Optional[str] = None,
        evidence_tables: Optional[list[dict]] = None,
        evidence_numbers: Optional[list[float]] = None,
        evidence_notes: Optional[str] = None,
    ) -> dict:
        """Check every number in a narrative against computed evidence.

        Uses the most recent run unless run_id is given; extra evidence may be
        passed as numbers or as tables shaped {"columns": [...], "rows": [[..]]}
        for figures computed elsewhere. Returns ungrounded_numbers plus any
        independent-recomputation issues (verified=false means a mismatch was
        found in the engine's own output).
        """
        try:
            return engine.verify_claims(
                answer,
                run_id=run_id,
                evidence_tables=evidence_tables,
                evidence_numbers=evidence_numbers,
                evidence_notes=evidence_notes,
            )
        except ToolkitError as exc:
            return {"error": str(exc)}

    return server


def main() -> None:
    transport = os.environ.get("INSIGHT_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in {"stdio", "sse", "streamable-http"}:
        raise SystemExit(
            "INSIGHT_MCP_TRANSPORT must be one of: stdio, sse, streamable-http"
        )
    create_server().run(transport=transport)


if __name__ == "__main__":
    main()
