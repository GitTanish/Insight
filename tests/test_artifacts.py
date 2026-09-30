import json

from insight.reports.artifacts import build_alert_monitor, build_dbt_model, slugify


def test_slugify_normalizes_names():
    assert slugify("Insight: Top Regions!!") == "insight_top_regions"
    assert slugify("") == "insight_model"


def test_dbt_model_contains_config_sql_and_yaml_tests():
    files = build_dbt_model(
        "Top Regions",
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region;",
        description="top revenue regions",
        columns=["region", "total"],
    )
    sql_path = "models/top_regions/top_regions.sql"
    yml_path = "models/top_regions/schema.yml"
    assert sql_path in files and yml_path in files and "README.md" in files

    assert files[sql_path].startswith("{{ config(materialized='view', tags=['insight']) }}")
    assert "GROUP BY region;" in files[sql_path]

    parsed = None
    import yaml

    parsed = yaml.safe_load(files[yml_path])
    model = parsed["models"][0]
    assert model["name"] == "top_regions"
    assert model["meta"]["generated_by"] == "insight"
    col_names = [c["name"] for c in model["columns"]]
    assert col_names == ["region", "total"]
    assert all("not_null" in c["data_tests"] for c in model["columns"])


def test_alert_monitor_is_valid_json_payload():
    monitor = build_alert_monitor(
        "West Revenue Spike",
        "SELECT * FROM orders WHERE amount > 1000",
        question="Which orders look anomalous?",
        schedule_cron="*/15 * * * *",
    )
    dumped = json.dumps(monitor)
    reparsed = json.loads(dumped)
    assert reparsed["name"] == "west_revenue_spike"
    assert reparsed["schedule_cron"] == "*/15 * * * *"
    assert reparsed["query"].endswith(";")
    assert reparsed["webhook"]["slack_compatible"] is True
    assert reparsed["webhook"]["url_env"] == "INSIGHT_ALERT_WEBHOOK_URL"
