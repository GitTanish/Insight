"""Resource and concurrency guards added in the hardening pass."""

import asyncio
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient

CSV_BYTES = b"region,revenue\nnorth,10\nsouth,20\n"


@pytest.fixture
def client():
    from webapp.app import app

    with TestClient(app) as test_client:
        test_client.get("/")
        yield test_client


def _session(client):
    from webapp.app import STORE

    sid = client.cookies.get("insight_sid")
    return sid, STORE.get(sid)


def _upload(client):
    response = client.post(
        "/upload",
        files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_query_rejects_second_in_flight_query(client):
    _upload(client)
    sid, session = _session(client)
    assert session.begin_query()
    try:
        html = client.post("/query", data={"question": "top region"})
        assert html.status_code == 409
        api = client.post("/api/query", json={"question": "top region"})
        assert api.status_code == 409
    finally:
        session.end_query()
        from webapp.app import STORE

        STORE.drop(sid)


def test_upload_is_rejected_while_a_query_runs(client):
    sid, session = _session(client)
    assert session.begin_query()
    try:
        response = client.post(
            "/upload",
            files={"file": ("sales.csv", io.BytesIO(CSV_BYTES), "text/csv")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        assert "flash=" in response.headers["location"]
        assert not session.has_dataset
    finally:
        session.end_query()
        from webapp.app import STORE

        STORE.drop(sid)


def test_clear_is_rejected_while_a_query_runs(client):
    _upload(client)
    sid, session = _session(client)
    assert session.begin_query()
    try:
        response = client.post("/clear", follow_redirects=False)
        assert response.status_code == 303
        assert "flash=" in response.headers["location"]
        assert session.has_dataset
    finally:
        session.end_query()
        from webapp.app import STORE

        STORE.drop(sid)


def test_upload_rejects_excessive_total_bytes(client, monkeypatch):
    import webapp.app as appmod

    monkeypatch.setattr(appmod, "MAX_TOTAL_UPLOAD_BYTES", 10)
    response = client.post(
        "/upload",
        files=[
            ("files", ("a.csv", io.BytesIO(b"a,b\n1,2\n"), "text/csv")),
            ("files", ("b.csv", io.BytesIO(b"a,b\n3,4\n"), "text/csv")),
        ],
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert "flash=" in response.headers["location"]
    sid, session = _session(client)
    assert not session.has_dataset
    from webapp.app import STORE

    STORE.drop(sid)


def test_session_generation_blocks_stale_turns_and_caps_history():
    from insight.domain.visualization import AnalysisResponse
    from webapp.sessions import WebSession

    session = WebSession(sid="gen")
    old_generation = session.generation
    old_data_id = session.data_id
    session.reset_data()
    response = AnalysisResponse(question="q", answer="a")

    assert not session.add_turn(
        generation=old_generation,
        data_id=old_data_id,
        question="stale",
        response=response,
        analysis_state=None,
    )
    for i in range(5):
        assert session.add_turn(
            generation=session.generation,
            data_id=session.data_id,
            question=f"q{i}",
            response=response,
            analysis_state={"turn": i},
            max_turns=3,
        )
    assert [t["question"] for t in session.turns] == ["q2", "q3", "q4"]
    assert session.analysis_state == {"turn": 4}


def test_session_busy_flags_are_mutually_exclusive():
    from webapp.sessions import WebSession

    session = WebSession(sid="busy")
    assert session.begin_query()
    assert not session.begin_query()
    assert not session.begin_upload()

    session.end_query()
    assert session.begin_upload()
    assert not session.begin_query()
    session.end_upload()
    assert not session.busy


def test_store_evicts_inactive_sessions_but_keeps_active_ones():
    from webapp.sessions import SessionStore

    store = SessionStore(max_sessions=2)
    active = store.get_or_create("active")
    store.get_or_create("idle")
    assert active.begin_query()

    store.get_or_create("new")
    assert store.get("active") is active
    assert store.get("idle") is None
    assert store.get("new") is not None
    active.end_query()
    assert len(store) == 2


def test_store_does_not_expire_an_active_session():
    from webapp.sessions import SessionStore

    now = [0.0]
    store = SessionStore(max_sessions=2, ttl_seconds=10, clock=lambda: now[0])
    session = store.get_or_create("a")
    assert session.begin_query()
    now[0] += 60
    assert store.get("a") is session
    session.end_query()


def test_cache_store_is_atomic_and_rejects_unsafe_keys(tmp_path, monkeypatch):
    from insight import cache as result_cache
    from insight.domain.visualization import AnalysisResponse, PlotArtifact

    class _Settings:
        artifacts_dir = tmp_path
        query_cache_enabled = True
        query_cache_ttl_days = 14
        query_cache_max_mb = 512

    monkeypatch.setattr(result_cache, "get_settings", lambda: _Settings())
    source = tmp_path / "chart.png"
    source.write_bytes(b"png-bytes")
    response = AnalysisResponse(
        question="q",
        answer="a",
        plots=[PlotArtifact(path=str(source), title="t", chart_type="bar")],
    )

    result_cache.store_cached_response("safe", response)
    payload = json.loads((tmp_path / "cache" / "safe.json").read_text(encoding="utf-8"))
    assert payload["answer"] == "a"
    assert not list((tmp_path / "cache").glob(".*.tmp"))

    loaded = result_cache.load_cached_response("safe", tmp_path / "out")
    assert loaded is not None
    assert (tmp_path / "out" / "chart.png").read_bytes() == b"png-bytes"

    result_cache.store_cached_response("../escape", response)
    assert not (tmp_path / "escape.json").exists()
    assert result_cache.load_cached_response("../escape", tmp_path / "out") is None


def test_cache_concurrent_writers_leave_valid_json(tmp_path, monkeypatch):
    import threading

    from insight import cache as result_cache
    from insight.domain.visualization import AnalysisResponse

    class _Settings:
        artifacts_dir = tmp_path
        query_cache_enabled = True
        query_cache_ttl_days = 14
        query_cache_max_mb = 512

    monkeypatch.setattr(result_cache, "get_settings", lambda: _Settings())

    def write(answer):
        result_cache.store_cached_response(
            "race", AnalysisResponse(question="q", answer=answer)
        )

    threads = [threading.Thread(target=write, args=(f"answer-{i}",)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    payload = json.loads((tmp_path / "cache" / "race.json").read_text(encoding="utf-8"))
    assert payload["answer"].startswith("answer-")


def test_toolkit_run_history_is_bounded():
    from insight.toolkit import AnalysisToolkit, ToolkitError

    toolkit = AnalysisToolkit(max_runs=2)
    toolkit.load_table("sales", content="region,revenue\nnorth,10\nsouth,20\n")
    run_ids = []
    for _ in range(4):
        out = toolkit.execute_operation(
            "sales",
            "group_aggregate",
            {
                "group_by": "region",
                "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
            },
        )
        run_ids.append(out["run_id"])

    assert len(toolkit._runs) == 2
    assert run_ids[0] not in toolkit._runs
    with pytest.raises(ToolkitError):
        toolkit.verify_claims("nothing", run_id=run_ids[0])


def test_toolkit_table_registry_is_bounded():
    from insight.toolkit import AnalysisToolkit, ToolkitError

    toolkit = AnalysisToolkit(max_tables=2)
    toolkit.load_table("one", content="x\n1\n")
    toolkit.load_table("two", content="x\n2\n")
    with pytest.raises(ToolkitError, match="too many tables"):
        toolkit.load_table("three", content="x\n3\n")

    toolkit.load_table("one", content="x\n10\n")
    assert set(toolkit._tables) == {"one", "two"}


def test_temp_artifact_dirs_are_pruned(monkeypatch, tmp_path):
    import os
    import tempfile
    import time

    from insight.orchestrator import _prune_temp_artifact_dirs

    old = tmp_path / "insight_artifacts_old"
    fresh = tmp_path / "insight_artifacts_fresh"
    old.mkdir()
    fresh.mkdir()
    (old / "chart.png").write_bytes(b"x")
    old_time = time.time() - 2 * 86_400
    os.utime(old, (old_time, old_time))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    _prune_temp_artifact_dirs()

    assert not old.exists()
    assert fresh.exists()


def test_http_pool_reuses_clients_and_keeps_custom_transports_isolated():
    from insight.llm.http_pool import close_http_clients, http_client

    async def scenario():
        async with http_client("https://example.test", 5.0) as first:
            async with http_client("https://example.test", 5.0) as second:
                assert first is second
        await close_http_clients()
        assert first.is_closed

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
        async with http_client("https://example.test", 5.0, transport) as first:
            async with http_client("https://example.test", 5.0, transport) as second:
                assert first is not second
        assert first.is_closed and second.is_closed

    asyncio.run(scenario())


def test_langsmith_client_is_reused(monkeypatch):
    from insight import observability

    class _Session:
        closed = False

        def close(self):
            self.closed = True

    class _Client:
        session = _Session()

    monkeypatch.setattr(observability, "_LSClient", lambda: _Client())
    monkeypatch.setattr(observability, "_CLIENT", None)
    monkeypatch.setattr(observability, "_CLIENT_FAILED", False)

    first = observability._get_client()
    second = observability._get_client()
    assert first is second
    observability.close_langsmith_client()
    assert first.session.closed


def test_csv_ingest_caps_rows_before_full_parse(monkeypatch):
    from insight import ingest

    class _Settings:
        max_upload_rows = 2

    monkeypatch.setattr(ingest, "get_settings", lambda: _Settings())
    frame, error = ingest.parse_csv_content(b"value\n1\n2\n")
    assert error is None and len(frame) == 2

    frame, error = ingest.parse_csv_content(b"value\n1\n2\n3\n")
    assert frame is None
    assert "max 2" in error
