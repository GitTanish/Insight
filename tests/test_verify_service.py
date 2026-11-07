from fastapi.testclient import TestClient

from insight.verify_service import app

client = TestClient(app)

SALES_CSV = "region,revenue\nNorth,100\nNorth,50\nSouth,40\n"
GROUP_AGG = {
    "name": "group_aggregate",
    "params": {
        "group_by": "region",
        "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
    },
}


def test_evidence_only_mode():
    response = client.post(
        "/v1/verify",
        json={"answer": "Revenue was 77.", "evidence": {"numbers": [77]}},
    )

    assert response.status_code == 200
    certificate = response.json()
    assert certificate["attestation_level"] == "grounding"
    assert certificate["verdict"]["grounded"] is True
    assert certificate["verdict"]["recomputed"] == "not_checked"


def test_execute_mode_recomputes():
    response = client.post(
        "/v1/verify",
        json={
            "answer": "North leads with 150.",
            "data": {"csv": SALES_CSV, "table_name": "sales"},
            "operation": GROUP_AGG,
        },
    )

    assert response.status_code == 200
    certificate = response.json()
    assert certificate["attestation_level"] == "recomputed"
    assert certificate["verdict"]["recomputed"] == "pass"
    assert certificate["verdict"]["overall"] == "pass"


def test_failed_verification_is_still_a_certificate():
    response = client.post(
        "/v1/verify",
        json={
            "answer": "North leads with 999.",
            "data": {"csv": SALES_CSV, "table_name": "sales"},
            "operation": GROUP_AGG,
        },
    )

    assert response.status_code == 200
    certificate = response.json()
    assert certificate["verdict"]["overall"] == "fail"
    assert certificate["verdict"]["grounded"] is False


def test_requires_evidence_or_data():
    response = client.post("/v1/verify", json={"answer": "Nothing computed."})
    assert response.status_code == 422


def test_requires_exactly_one_execution_mode():
    response = client.post(
        "/v1/verify",
        json={
            "answer": "North leads with 150.",
            "data": {"csv": SALES_CSV, "table_name": "sales"},
            "operation": GROUP_AGG,
            "sql": "SELECT 1",
        },
    )
    assert response.status_code == 422


def test_bearer_token_is_required_when_configured(monkeypatch):
    monkeypatch.setenv("INSIGHT_VERIFY_AUTH_TOKEN", "secret")

    response = client.post(
        "/v1/verify",
        json={"answer": "Revenue was 77.", "evidence": {"numbers": [77]}},
    )
    assert response.status_code == 401

    response = client.post(
        "/v1/verify",
        json={"answer": "Revenue was 77.", "evidence": {"numbers": [77]}},
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 200
