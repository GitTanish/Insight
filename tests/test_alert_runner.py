"""Offline tests for the alert-runner companion worker (tools/alert_runner.py)."""
import json

import pandas as pd
import pytest

from insight.analytics.sql_engine import DuckSession
from insight.reports.artifacts import build_alert_monitor
from tools.alert_runner import Monitor, Runner, load_datasets, load_monitors, main


@pytest.fixture
def data_dir(tmp_path):
    (tmp_path / "sales.csv").write_text(
        "region,revenue\nNorth,100\nSouth,220\nEast,150\nWest,300\n",
        encoding="utf-8",
    )
    return tmp_path


def _write_monitor(tmp_path, monitor: dict, name: str = "insight_monitor.json") -> None:
    directory = tmp_path / "monitors"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(json.dumps(monitor), encoding="utf-8")


def _export_monitor() -> dict:
    return build_alert_monitor(
        "insight_slow_region",
        "SELECT region, sum(revenue) AS total FROM sales GROUP BY region",
        question="Which region leads revenue?",
    )


def test_monitor_round_trips_exported_descriptor(tmp_path):
    _write_monitor(tmp_path, _export_monitor())
    monitors = load_monitors(tmp_path / "monitors")
    assert len(monitors) == 1
    assert monitors[0].name == "insight_slow_region"
    assert "GROUP BY region" in monitors[0].query
    assert monitors[0].webhook_url_env == "INSIGHT_ALERT_WEBHOOK_URL"


def test_monitor_name_filter(tmp_path):
    _write_monitor(tmp_path, _export_monitor(), "a.json")
    _write_monitor(tmp_path, build_alert_monitor("other", "SELECT 1 AS v"), "b.json")
    monitors = load_monitors(tmp_path / "monitors", "other")
    assert [m.name for m in monitors] == ["other"]


def test_load_datasets_reads_csv_directory(data_dir):
    datasets = load_datasets(data_dir)
    assert list(datasets) == ["sales.csv"]
    assert len(datasets["sales.csv"]) == 4


def test_no_threshold_alerts_when_rows_returned(data_dir):
    monitor = Monitor.from_dict(_export_monitor())
    runner = Runner(monitors=[monitor], datasets=load_datasets(data_dir))
    with DuckSession(runner.datasets) as session:
        outcome = runner.evaluate(monitor, session)
    assert outcome.error is None
    assert outcome.rows == 4
    assert outcome.triggered is True


def test_threshold_metric_triggers(data_dir):
    monitor = Monitor.from_dict({
        "name": "big_total",
        "query": "SELECT sum(revenue) AS total FROM sales",
        "threshold": {"metric_column": "total", "op": ">", "value": 100},
    })
    runner = Runner(monitors=[monitor], datasets=load_datasets(data_dir))
    with DuckSession(runner.datasets) as session:
        outcome = runner.evaluate(monitor, session)
    assert outcome.triggered is True
    assert "total=770" in outcome.threshold_note


def test_threshold_not_met_stays_quiet(data_dir):
    monitor = Monitor.from_dict({
        "name": "huge_total",
        "query": "SELECT sum(revenue) AS total FROM sales",
        "threshold": {"metric_column": "total", "op": ">", "value": 99999},
    })
    runner = Runner(monitors=[monitor], datasets=load_datasets(data_dir))
    with DuckSession(runner.datasets) as session:
        outcome = runner.evaluate(monitor, session)
    assert outcome.triggered is False
    runner.cycle()
    assert runner.posted == 0


def test_destructive_query_is_rejected(data_dir):
    monitor = Monitor.from_dict({
        "name": "evil",
        "query": "DROP TABLE sales;",
    })
    runner = Runner(monitors=[monitor], datasets=load_datasets(data_dir))
    runner.cycle()
    assert runner.posted == 0


def test_state_change_posts_once_then_holds(data_dir, capsys):
    monitor = Monitor.from_dict(_export_monitor())
    runner = Runner(monitors=[monitor], datasets=load_datasets(data_dir), dry_run=True)
    runner.cycle()
    assert "[dry-run] insight_slow_region:" in capsys.readouterr().out
    assert runner.posted == 1

    runner.cycle()
    out = capsys.readouterr().out
    assert "still triggered" in out
    assert runner.posted == 1

    runner.repeat = True
    runner.cycle()
    assert runner.posted == 2


def test_webhook_post_receives_slack_payload(data_dir, monkeypatch):
    sent: dict = {}

    class _FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(request, timeout=0):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    monitor = Monitor.from_dict(_export_monitor())
    runner = Runner(
        monitors=[monitor], datasets=load_datasets(data_dir),
        webhook_url="https://hooks.example.com/services/T000/B000/XXX",
    )
    runner.cycle()
    assert runner.posted == 1
    assert sent["url"] == "https://hooks.example.com/services/T000/B000/XXX"
    assert "insight_slow_region" in sent["body"]["text"]
    assert "4 rows" in sent["body"]["text"]


def test_main_once_dry_run(data_dir, tmp_path, capsys):
    _write_monitor(tmp_path, _export_monitor())
    code = main([
        "--monitors", str(tmp_path / "monitors"),
        "--data", str(data_dir),
        "--once", "--dry-run",
    ])
    assert code == 0
    out = capsys.readouterr().out
    assert "alert-runner: 1 monitor(s)" in out
    assert "[dry-run] insight_slow_region:" in out


def test_main_reports_when_no_monitors(tmp_path, capsys):
    code = main(["--monitors", str(tmp_path / "missing"), "--data", str(tmp_path)])
    assert code == 1
    assert "no monitors found" in capsys.readouterr().out
