import io
import json
import os
import zipfile

import pytest

from tests.test_webapp import AlternateStubChain, CSV_BYTES, parse_sse

JOIN_CSV_A = (
    b"order_id,customer_id,amount\n"
    b"1,c1,100\n2,c1,40\n3,c2,60\n"
)
JOIN_CSV_B = (
    b"customer_id,region\n"
    b"c1,west\nc2,east\n"
)

SQL_PLAN = """
{"objective":"Revenue by region via join","steps":[
 {"step_id":0,"operation":"sql_query",
  "params":{"query":"SELECT c.region AS region, SUM(o.amount) AS total FROM orders o JOIN customers c USING (customer_id) GROUP BY c.region ORDER BY total DESC"},
  "reason":"join and aggregate"}],
 "charts":[]}
"""
SQL_EXPLAIN = "**West leads with $100 total.**"


class SqlStubChain(AlternateStubChain):
    async def generate(self, request):
        from insight.llm.base import LLMResponse

        self.calls += 1
        content = SQL_PLAN if self.calls % 2 == 1 else SQL_EXPLAIN
        self.last_used = "fake/fake-1"
        return LLMResponse(content=content, provider="fake", model="fake-1", latency_ms=1)


@pytest.fixture
def sql_client(monkeypatch):
    os.environ.setdefault("INSIGHT_TRACING", "off")
    from webapp.app import app
    import insight.orchestrator as orch

    stub = SqlStubChain()
    monkeypatch.setattr(orch, "_build_chain_for", lambda *a, **k: stub)
    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app) as c:
        yield c


def test_multi_file_upload_join_query_and_exports(sql_client, tmp_path):
    r = sql_client.post(
        "/upload",
        files=[
            ("files", ("orders.csv", io.BytesIO(JOIN_CSV_A), "text/csv")),
            ("files", ("customers.csv", io.BytesIO(JOIN_CSV_B), "text/csv")),
        ],
        follow_redirects=False,
    )
    assert r.status_code == 303

    page = sql_client.get("/")
    assert "orders.csv loaded" in page.text

    q = sql_client.post("/query", data={"question": "Revenue per region?"})
    assert q.status_code == 200
    events = parse_sse(q.text)
    result_payload = next(d for e, d in events if e == "result")
    assert "West leads" in result_payload["html"]
    assert "Export dbt model" in result_payload["html"]

    dbt = sql_client.post("/export/dbt", data={"step_id": 0})
    assert dbt.status_code == 200
    assert dbt.content[:2] == b"PK"
    with zipfile.ZipFile(io.BytesIO(dbt.content)) as zf:
        names = zf.namelist()
        assert any(n.endswith(".sql") for n in names)
        schema_yml = next(n for n in names if n.endswith("schema.yml"))
        yml_text = zf.read(schema_yml).decode("utf-8")
        assert "generated_by: insight" in yml_text
        assert "SUM(o.amount)" in "".join(
            zf.read(n).decode("utf-8") for n in names if n.endswith(".sql")
        )

    alert = sql_client.get("/export/alert", params={"step_id": 0})
    assert alert.status_code == 200
    monitor = json.loads(alert.content)
    assert monitor["query"].lower().startswith("select")
    assert "JOIN customers" in monitor["query"]
    assert monitor["webhook"]["slack_compatible"] is True

    missing = sql_client.post("/export/dbt", data={"step_id": 99})
    assert missing.status_code == 404


def test_upload_still_accepts_single_legacy_field(sql_client):
    r = sql_client.post(
        "/upload",
        files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = sql_client.get("/")
    assert "sales.csv loaded" in page.text
