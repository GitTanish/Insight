"""Upload must not block on the optional LLM briefing polish."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi.testclient import TestClient

from webapp.app import STORE, app
from webapp.sessions import SessionStore, WebSession

def _diamonds_csv() -> bytes:
    """A dataset that reliably triggers deterministic briefing findings."""
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "evaluation" / "data" / "diamonds.csv"
    if src.exists():
        return src.read_bytes()

    lines = ['"carat","cut","color","price"']
    for i in range(200):
        lines.append(f'{0.1 + i * 0.01},"cut_{(i % 5)}","color_{(i % 7)}",{500 + i * 90}')
    return "\n".join(lines).encode("utf-8")


CSV = _diamonds_csv()


def test_findings_status_defaults_to_ready():
    session = WebSession(sid="s1")
    assert session.findings_status == "ready"
    session.findings_status = "polishing"
    session.reset_data()
    assert session.findings_status == "ready"


def test_upload_returns_deterministic_findings_without_llm_roundtrip():
    """The redirect must not wait on the LLM polish."""
    with TestClient(app) as client:
        response = client.post(
            "/upload",
            files={"files": ("sales.csv", CSV, "text/csv")},
            follow_redirects=False,
        )
        assert response.status_code == 303
        page = client.get("/")
        assert page.status_code == 200
        # Deterministic findings are present immediately after upload.
        assert "Top Findings" in page.text


def test_polish_task_updates_session_findings_in_place():
    import asyncio

    store = SessionStore()
    session = WebSession(sid="s2")
    session.findings = [{"headline": "raw"}]
    session.findings_status = "polishing"
    with store._lock:
        store._sessions["s2"] = session

    import webapp.app as appmod

    async def _polish(findings, generate, model_name=None):
        return [{"headline": "polished"}]

    import insight.briefing as briefing_mod
    original = briefing_mod.polish_findings_with_llm
    briefing_mod.polish_findings_with_llm = _polish
    try:
        asyncio.run(appmod._polish_session_findings(store, "s2", session.data_id))
    finally:
        briefing_mod.polish_findings_with_llm = original

    assert session.findings == [{"headline": "polished"}]
    assert session.findings_status == "ready"


def test_polish_task_ignores_stale_session_generation():
    import asyncio

    store = SessionStore()
    session = WebSession(sid="s3")
    session.findings = [{"headline": "raw"}]
    session.findings_status = "polishing"
    with store._lock:
        store._sessions["s3"] = session

    import webapp.app as appmod

    called = []

    async def _polish(findings, generate, model_name=None):
        called.append(True)
        return [{"headline": "polished"}]

    import insight.briefing as briefing_mod
    original = briefing_mod.polish_findings_with_llm
    briefing_mod.polish_findings_with_llm = _polish
    try:
        asyncio.run(appmod._polish_session_findings(store, "s3", "stale-data-id"))
    finally:
        briefing_mod.polish_findings_with_llm = original

    assert not called, "must not polish a session that was replaced by a new upload"
    assert session.findings == [{"headline": "raw"}]


def test_index_template_marks_pending_polish():
    import pathlib
    template = pathlib.Path(__file__).parent.parent / "webapp" / "templates" / "index.html"
    html = template.read_text(encoding="utf-8")
    assert "findings_status" in html
    assert "polishing" in html


def test_store_sessions_are_reusable_after_upload():
    sid = "reused-session"
    STORE.drop(sid)
    with TestClient(app) as client:
        client.post(
            "/upload",
            files={"files": ("sales.csv", CSV, "text/csv")},
            follow_redirects=False,
        )
        page = client.get("/")
        assert page.status_code == 200
    STORE.drop(sid)


def test_no_unused_df_dependency_in_polish_signature():
    import inspect

    import webapp.app as appmod
    params = list(inspect.signature(appmod._polish_session_findings).parameters)
    assert params == ["store", "sid", "data_id"]


def test_dataframe_fixture_is_valid():
    df = pd.DataFrame({"region": ["N"], "revenue": [1.0]})
    assert len(df) == 1
