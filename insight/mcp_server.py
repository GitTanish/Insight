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

import hmac
import os
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from insight.toolkit import AnalysisToolkit, ToolkitError

INSTRUCTIONS = (
    "Deterministic analytics engine with independent verification. Typical flow: "
    "load_table → (profile_table / list_tables) → execute_operation, run_plan or "
    "run_sql → write your narrative → verify_claims to check every number in it "
    "against computed evidence. No LLM is involved; results are computed by "
    "pandas/scipy/sandboxed DuckDB. Prefer verify_claims before presenting any "
    "numbers to a user."
)

MAX_FILE_MB_DEFAULT = 80
MAX_RUNS_DEFAULT = 100
MAX_TABLES_DEFAULT = 64


def file_roots_from_env(env: Optional[dict] = None) -> Optional[list[Path]]:
    """Path loading policy: unset → cwd; ``all`` → unrestricted; empty → none."""
    values = os.environ if env is None else env
    raw = values.get("INSIGHT_MCP_FILE_ROOTS")
    if raw is None:
        return [Path.cwd().resolve()]
    raw = raw.strip()
    if raw.lower() == "all":
        return None
    if raw.lower() in {"", "none"}:
        return []
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def resolve_auth_token(
    transport: str, env: Optional[dict] = None
) -> Optional[str]:
    """HTTP transports require a shared secret unless explicitly opted out."""
    values = os.environ if env is None else env
    token = (values.get("INSIGHT_MCP_AUTH_TOKEN") or "").strip() or None
    if transport == "stdio" or token:
        return token
    allow_open = (values.get("INSIGHT_MCP_ALLOW_UNAUTHENTICATED") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if allow_open:
        return None
    raise SystemExit(
        "Refusing to start an unauthenticated MCP HTTP transport. Set "
        "INSIGHT_MCP_AUTH_TOKEN and send 'Authorization: Bearer <token>', or set "
        "INSIGHT_MCP_ALLOW_UNAUTHENTICATED=1 to opt out deliberately."
    )


class BearerAuthMiddleware:
    """Shared-secret gate for the SSE / streamable-http transports."""

    def __init__(self, app, token: str):
        self.app = app
        self.token = token
        self.expected = f"Bearer {token}".encode("latin-1")

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        supplied = headers.get("authorization", "").encode("latin-1")
        if hmac.compare_digest(supplied, self.expected):
            await self.app(scope, receive, send)
            return
        body = b'{"error":"unauthorized"}'
        await send(
            {
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_server(toolkit: Optional[AnalysisToolkit] = None) -> FastMCP:
    host = os.environ.get("INSIGHT_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("INSIGHT_MCP_PORT", "8765") or 8765)
    max_file_mb = int(os.environ.get("INSIGHT_MCP_MAX_FILE_MB", str(MAX_FILE_MB_DEFAULT)) or MAX_FILE_MB_DEFAULT)
    max_runs = int(os.environ.get("INSIGHT_MCP_MAX_RUNS", str(MAX_RUNS_DEFAULT)) or MAX_RUNS_DEFAULT)
    max_tables = int(os.environ.get("INSIGHT_MCP_MAX_TABLES", str(MAX_TABLES_DEFAULT)) or MAX_TABLES_DEFAULT)
    engine = toolkit or AnalysisToolkit(
        file_roots=file_roots_from_env(),
        max_file_bytes=max_file_mb * 1024 * 1024,
        max_runs=max_runs,
        max_tables=max_tables,
    )
    configured_hosts = [
        part.strip()
        for part in os.environ.get("INSIGHT_MCP_ALLOWED_HOSTS", "").split(",")
        if part.strip()
    ]
    allowed_hosts = configured_hosts or [
        f"{host}:{port}",
        f"localhost:{port}",
        f"127.0.0.1:{port}",
        "localhost:*",
        "127.0.0.1:*",
    ]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed_hosts,
        allowed_origins=[
            f"http://localhost:{port}",
            f"http://127.0.0.1:{port}",
        ],
    )
    server = FastMCP(
        "insight",
        instructions=INSTRUCTIONS,
        host=host,
        port=port,
        transport_security=security,
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
    token = resolve_auth_token(transport)
    server = create_server()
    if transport == "stdio":
        server.run(transport="stdio")
        return

    import uvicorn

    asgi_app = (
        server.streamable_http_app()
        if transport == "streamable-http"
        else server.sse_app()
    )
    if token:
        asgi_app = BearerAuthMiddleware(asgi_app, token)
    uvicorn.run(
        asgi_app,
        host=os.environ.get("INSIGHT_MCP_HOST", "127.0.0.1"),
        port=int(os.environ.get("INSIGHT_MCP_PORT", "8765") or 8765),
        log_level="info",
    )


if __name__ == "__main__":
    main()
