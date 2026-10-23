import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from insight.settings import get_settings
from webapp.telemetry import MetricsRegistry, TelemetryMiddleware, metric_path


def test_metric_path_normalization():
    assert metric_path("/healthz") == "/healthz"
    assert metric_path("/artifacts/abc123/chart.png") == "/artifacts/{data_id}/{filename}"
    assert metric_path("/static/app.js") == "/static/{file}"
    assert metric_path("/unknown/route") == "other"


def test_telemetry_request_id_metric_and_log(caplog):
    caplog.set_level(logging.INFO, logger="insight.http")
    registry = MetricsRegistry()

    app = FastAPI()

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    app.add_middleware(TelemetryMiddleware, registry=registry)

    with TestClient(app) as client:
        echoed = client.get("/healthz", headers={"X-Request-ID": "req-abc"})
        generated = client.get("/healthz")

    assert echoed.headers["x-request-id"] == "req-abc"
    assert generated.headers["x-request-id"]
    assert generated.headers["x-request-id"] != "req-abc"

    text = registry.render()
    assert 'insight_http_requests_total{method="GET",path="/healthz",status="200"} 2' in text
    assert "insight_sessions" in text and "insight_uptime_seconds" in text

    records = [r for r in caplog.records if r.name == "insight.http"]
    payload = json.loads(records[-1].message)
    assert payload["event"] == "http_request"
    assert payload["status"] == 200
    assert payload["path"] == "/healthz"
    assert payload["request_id"] == generated.headers["x-request-id"]


def test_readyz_reports_checks(tmp_path, monkeypatch):
    from webapp import readiness

    settings = get_settings().model_copy(update={"artifacts_dir": tmp_path})
    monkeypatch.setattr(readiness, "_provider_configured", lambda s: True)

    ready, report = readiness.readiness_report(settings)

    assert ready is True
    assert report["status"] == "ready"
    assert report["artifacts_writable"] is True
    assert report["version"]


def test_readyz_not_ready_without_provider(tmp_path, monkeypatch):
    from webapp import readiness

    settings = get_settings().model_copy(update={"artifacts_dir": tmp_path})
    monkeypatch.setattr(readiness, "_provider_configured", lambda s: False)

    ready, report = readiness.readiness_report(settings)

    assert ready is False
    assert report["status"] == "not_ready"
    assert report["provider_configured"] is False


def test_readyz_endpoint(monkeypatch):
    from webapp.app import app
    from webapp import app as app_module

    monkeypatch.setattr(
        app_module,
        "readiness_report",
        lambda: (True, {"status": "ready", "version": "x"}),
    )
    response = TestClient(app).get("/readyz")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_metrics_endpoint_exposes_prometheus_text(monkeypatch, tmp_path):
    from webapp.app import app

    response = TestClient(app).get("/metrics")
    assert response.status_code == 200
    assert "insight_http_requests_total" in response.text
    assert "insight_sessions" in response.text
