import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from insight.settings import get_settings
from webapp.ratelimit import RateLimitMiddleware, SlidingWindowLimiter
from webapp.security import (
    AuthMiddleware,
    SecurityHeadersMiddleware,
    request_identity,
    session_cookie_kwargs,
)

CSV_BYTES = b"region,revenue\nNorth,100\nSouth,40\nEast,60\n"


def _auth_app(**kwargs):
    app = FastAPI()

    @app.get("/hidden")
    def hidden():
        return {"ok": True}

    @app.get("/healthz")
    def health():
        return {"ok": True}

    app.add_middleware(AuthMiddleware, **kwargs)
    return app


def test_auth_middleware_rejects_without_credentials():
    client = TestClient(_auth_app(token="s3cret"))
    assert client.get("/hidden").status_code == 401
    assert client.get("/healthz").status_code == 200


def test_auth_middleware_accepts_bearer_and_cookie():
    client = TestClient(_auth_app(token="s3cret"))
    bearer = client.get("/hidden", headers={"Authorization": "Bearer s3cret"})
    assert bearer.status_code == 200
    cookie = client.get("/hidden", headers={"Cookie": "insight_auth=s3cret"})
    assert cookie.status_code == 200
    wrong = client.get("/hidden", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401


def test_auth_middleware_trusts_forwarded_user_only_when_enabled():
    client = TestClient(_auth_app(token="", trust_proxy=True))
    assert client.get("/hidden").status_code == 401
    allowed = client.get("/hidden", headers={"X-Forwarded-User": "alice"})
    assert allowed.status_code == 200


def test_rate_limiter_sliding_window_and_retry_after():
    now = [1000.0]
    limiter = SlidingWindowLimiter(clock=lambda: now[0])
    assert limiter.allow("k", 2) == (True, 0.0)
    assert limiter.allow("k", 2)[0] is True
    allowed, retry_after = limiter.allow("k", 2)
    assert allowed is False and 0 < retry_after <= 60

    now[0] += 61
    assert limiter.allow("k", 2)[0] is True
    assert limiter.allow("other", 2)[0] is True


def test_rate_limit_middleware_returns_429_for_queries():
    now = [1000.0]
    app = FastAPI()

    @app.post("/api/query")
    def query():
        return {"ok": True}

    @app.post("/upload")
    def upload():
        return {"ok": True}

    app.add_middleware(
        RateLimitMiddleware,
        query_limit=2,
        upload_limit=1,
        limiter=SlidingWindowLimiter(clock=lambda: now[0]),
    )

    with TestClient(app) as client:
        assert client.post("/api/query").status_code == 200
        assert client.post("/api/query").status_code == 200
        blocked = client.post("/api/query")
        assert blocked.status_code == 429
        assert blocked.headers["retry-after"] == "60"
        assert "rate limit" in blocked.json()["error"]

        assert client.post("/upload").status_code == 200
        assert client.post("/upload").status_code == 429

        now[0] += 60
        assert client.post("/api/query").status_code == 200


def test_request_identity_prefers_bearer_and_ignores_proxy_header_by_default(monkeypatch):
    scope = {"headers": [(b"authorization", b"Bearer abc")], "client": ("1.2.3.4", 1)}
    assert request_identity(scope, trust_proxy=False).startswith("bearer:")

    scope = {
        "headers": [(b"x-forwarded-user", b"forged")],
        "client": ("1.2.3.4", 1),
    }
    assert request_identity(scope, trust_proxy=False) == "ip:1.2.3.4"
    assert request_identity(scope, trust_proxy=True) == "user:forged"


def test_security_headers_and_hsts():
    app = FastAPI()

    @app.get("/")
    def index():
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware, hsts=True)
    response = TestClient(app).get("/")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "same-origin"
    assert "strict-transport-security" in response.headers


def test_session_cookie_kwargs():
    assert session_cookie_kwargs(secure=True) == {
        "httponly": True,
        "samesite": "lax",
        "secure": True,
    }


@pytest.fixture
def webapp_client(monkeypatch, tmp_path):
    from webapp.app import app

    monkeypatch.setattr(get_settings(), "artifacts_dir", tmp_path)
    with TestClient(app) as client:
        yield client


def test_artifact_requires_owning_session(webapp_client, tmp_path):
    upload = webapp_client.post(
        "/upload",
        files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
        follow_redirects=False,
    )
    assert upload.status_code == 303

    from webapp.app import STORE

    session = STORE.get(webapp_client.cookies.get("insight_sid"))
    artifact_dir = tmp_path / "web" / session.data_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "chart.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    owned = webapp_client.get(f"/artifacts/{session.data_id}/chart.png")
    assert owned.status_code == 200

    other = TestClient(webapp_client.app)
    stolen = other.get(f"/artifacts/{session.data_id}/chart.png")
    assert stolen.status_code == 404


def test_auth_bootstrap_sets_cookie_when_token_configured(monkeypatch):
    from webapp.app import app

    settings = get_settings()
    monkeypatch.setattr(settings, "auth_token", "bootstrap-token")

    client = TestClient(app)
    ok = client.get("/auth?token=bootstrap-token", follow_redirects=False)
    assert ok.status_code == 303
    assert ok.cookies.get("insight_auth") == "bootstrap-token"

    bad = client.get("/auth?token=wrong", follow_redirects=False)
    assert bad.status_code == 401
