import os

os.environ.setdefault("INSIGHT_TRACING", "off")

import pandas as pd
import pytest

from insight.analytics.operations import OPERATIONS, OperationError
from insight.analytics.sql_engine import DuckSession, unique_table_names
from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.execution.executor import execute_plan_sync


@pytest.fixture
def duck():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": ["c1", "c1", "c2", "c3"],
        "amount": [100.0, 50.0, 70.0, 20.0],
    })
    customers = pd.DataFrame({
        "customer_id": ["c1", "c2", "c3"],
        "region": ["west", "east", "west"],
    })
    with DuckSession({"orders.csv": orders, "customers.csv": customers}) as session:
        yield session


def test_join_group_order_matches_pandas(duck):
    frame = duck.execute("""
        SELECT c.region, SUM(o.amount) AS total
        FROM orders o JOIN customers c USING (customer_id)
        GROUP BY c.region ORDER BY total DESC
    """)
    assert frame["region"].tolist() == ["west", "east"]
    assert frame["total"].tolist() == [170.0, 70.0]


def test_window_function_ranking(duck):
    frame = duck.execute("""
        SELECT order_id,
               ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY amount DESC) AS rn
        FROM orders
    """)
    got = dict(zip(frame["order_id"], frame["rn"]))
    assert got == {1: 1, 2: 2, 3: 1, 4: 1}


def test_describe_lists_exact_tables_and_columns(duck):
    text = duck.describe()
    assert "- customers (3 rows): customer_id VARCHAR, region VARCHAR" in text
    assert "- orders (4 rows):" in text


@pytest.mark.parametrize("bad_sql", [
    "SELECT 1; DROP TABLE orders",
    "INSERT INTO orders VALUES (9,'c1',1.0)",
    "COPY orders TO 'evil.csv'",
    "ATTACH 'evil.db' AS e",
    "DELETE FROM orders",
    "UPDATE orders SET amount = 0",
    "CREATE TABLE t(a INT)",
    "SET enable_external_access=true",
    "PRAGMA database_list",
    "-- SELECT ok\nDROP TABLE orders",
])
def test_destructive_and_config_statements_blocked(duck, bad_sql):
    with pytest.raises(ValueError):
        duck.execute(bad_sql)


def test_external_access_disabled_blocks_file_reads(duck):
    with pytest.raises(ValueError):
        duck.execute("SELECT * FROM read_csv_auto('anything.csv')")


def test_missing_table_error_is_actionable(duck):
    with pytest.raises(ValueError, match="nonexistent_table"):
        duck.execute("SELECT * FROM nonexistent_table")


def test_timeout_interrupts_runaway_query(duck, monkeypatch):
    class _FastSettings:
        sql_timeout_s = 1.0
        sql_max_output_rows = 10_000

    monkeypatch.setattr(
        "insight.analytics.sql_engine.get_settings", lambda: _FastSettings()
    )
    duck.timeout_s = 1.0
    import time

    start = time.monotonic()
    with pytest.raises(ValueError, match="interrupted"):
        duck.execute("""
            WITH RECURSIVE t(x) AS
            (SELECT 1 UNION ALL SELECT x+1 FROM t WHERE x < 500000000)
            SELECT count(*) FROM t
        """)
    assert time.monotonic() - start < 5


def test_row_cap_truncates_result(duck, monkeypatch):
    duck.max_rows = 2
    frame = duck.execute("SELECT order_id FROM orders ORDER BY order_id")
    assert len(frame) == 2


def test_unique_table_names_dedupe_and_sanitize():
    mapping = unique_table_names(["Sales 2024.csv", "sales-2024.csv", "orders.csv"])
    assert list(mapping.values()) == ["sales_2024", "sales_2024_2", "orders"]


def test_sql_query_operation_emits_table(duck):
    out = OPERATIONS["sql_query"].run(
        None,
        OPERATIONS["sql_query"].params_model(query="SELECT count(*) AS n FROM orders"),
        step_id=0,
        ctx=duck,
    )
    table = out.table.to_dataframe()
    assert int(table["n"].iloc[0]) == 4
    labels = {c.label: c.value for c in out.calculations}
    assert labels["rows_returned"] == 1


def test_sql_query_without_session_fails_clean():
    with pytest.raises(OperationError, match="no SQL session"):
        OPERATIONS["sql_query"].run(
            pd.DataFrame({"a": [1]}),
            OPERATIONS["sql_query"].params_model(query="SELECT 1"),
            step_id=0,
            ctx=None,
        )


SQL_PLAN = AnalysisPlan(
    objective="top order per customer",
    steps=[
        PlanStep(
            step_id=0,
            operation="sql_query",
            params={"query": (
                "SELECT customer_id, amount FROM orders "
                "QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id "
                "ORDER BY amount DESC) = 1"
            )},
        ),
        PlanStep(
            step_id=1,
            operation="group_aggregate",
            params={
                "group_by": "customer_id",
                "metrics": [{"column": "amount", "agg": "sum", "alias": "best_amount"}],
            },
            input_step=0,
        ),
    ],
    charts=[],
)


def test_executor_chains_from_sql_output():
    orders = pd.DataFrame({
        "order_id": [1, 2, 3],
        "customer_id": ["c1", "c1", "c2"],
        "amount": [90.0, 40.0, 60.0],
    })
    with DuckSession({"orders.csv": orders}) as session:
        result = execute_plan_sync(pd.DataFrame({"x": [0]}), SQL_PLAN, duck_ctx=session)

    assert result.success
    chained = result.get_table(1).to_dataframe()
    assert sorted(chained["customer_id"].tolist()) == ["c1", "c2"]
    amounts = dict(zip(chained["customer_id"], chained["best_amount"]))
    assert amounts == {"c1": 90.0, "c2": 60.0}
