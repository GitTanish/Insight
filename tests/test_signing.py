import json

import pandas as pd
import pytest

from insight.toolkit import AnalysisToolkit
from insight.verification.signing import (
    dataset_sha256,
    sign_certificate,
    verify_certificate_signature,
)
from insight.verify_cli import main

SALES_CSV = (
    "region,revenue\n"
    "North,100\n"
    "South,40\n"
    "North,50\n"
)

SIGNING_KEY = "test-signing-key"
KEY_ID = "test-key"


@pytest.fixture
def tk():
    return AnalysisToolkit(signing_key=SIGNING_KEY, signing_key_id=KEY_ID)


def _group_run(toolkit):
    return toolkit.execute_operation(
        "sales",
        "group_aggregate",
        {
            "group_by": "region",
            "metrics": [{"column": "revenue", "agg": "sum", "alias": "total"}],
        },
    )


def _certificate(toolkit):
    run = _group_run(toolkit)
    return toolkit.verify_claims(
        "North leads with 150 in revenue.",
        run_id=run["run_id"],
    )["certificate"]


def test_signed_certificate_verifies_and_commits_to_dataset(tk):
    tk.load_table("sales", content=SALES_CSV)
    certificate = _certificate(tk)

    signature = certificate["signature"]
    assert signature["alg"] == "HMAC-SHA256"
    assert signature["key_id"] == KEY_ID
    assert signature["value"]

    dataset_digest = certificate["subject"]["dataset_sha256"]
    assert isinstance(dataset_digest, str)
    assert len(dataset_digest) == 64

    assert verify_certificate_signature(certificate, SIGNING_KEY) is True
    assert verify_certificate_signature(certificate, "wrong-key") is False
    assert SIGNING_KEY not in json.dumps(certificate)


def test_unsigned_certificate_stays_backward_compatible():
    toolkit = AnalysisToolkit(signing_key="")
    toolkit.load_table("sales", content=SALES_CSV)
    certificate = _certificate(toolkit)

    assert certificate["signature"] is None
    assert verify_certificate_signature(certificate, SIGNING_KEY) is False


def test_signature_survives_json_roundtrip(tk):
    tk.load_table("sales", content=SALES_CSV)
    certificate = json.loads(json.dumps(_certificate(tk)))

    assert verify_certificate_signature(certificate, SIGNING_KEY) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cert: cert["subject"].__setitem__("answer_sha256", "0" * 64),
        lambda cert: cert["aggregates"][0].__setitem__("value", 999),
        lambda cert: cert["claims"][0].__setitem__("text", "tampered"),
        lambda cert: cert["verdict"].__setitem__("overall", "fail"),
    ],
)
def test_tampering_invalidates_signature(tk, mutate):
    tk.load_table("sales", content=SALES_CSV)
    certificate = json.loads(json.dumps(_certificate(tk)))
    mutate(certificate)

    assert verify_certificate_signature(certificate, SIGNING_KEY) is False


def test_dataset_digest_is_stable_and_sensitive():
    frame = pd.DataFrame(
        {"region": ["North", "South"], "revenue": [100, 40]}
    )
    assert dataset_sha256(frame) == dataset_sha256(frame.copy())

    changed_value = frame.copy()
    changed_value.loc[0, "revenue"] = 101
    assert dataset_sha256(frame) != dataset_sha256(changed_value)

    changed_schema = frame.rename(columns={"revenue": "sales"})
    assert dataset_sha256(frame) != dataset_sha256(changed_schema)


def test_sign_certificate_helper_matches_builder(tk):
    tk.load_table("sales", content=SALES_CSV)
    certificate = _certificate(tk)
    unsigned = {key: value for key, value in certificate.items() if key != "signature"}

    resigned = sign_certificate(unsigned, SIGNING_KEY, KEY_ID)
    assert verify_certificate_signature(resigned, SIGNING_KEY) is True


def test_cli_verifies_signed_certificate(tk, tmp_path):
    tk.load_table("sales", content=SALES_CSV)
    certificate = _certificate(tk)
    certificate_path = tmp_path / "certificate.json"
    key_path = tmp_path / "signing-key.txt"
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    key_path.write_text(SIGNING_KEY + "\n", encoding="utf-8")

    assert (
        main(
            [
                "--verify-certificate",
                str(certificate_path),
                "--signing-key-file",
                str(key_path),
            ]
        )
        == 0
    )

    tampered = json.loads(json.dumps(certificate))
    tampered["subject"]["answer_sha256"] = "0" * 64
    certificate_path.write_text(json.dumps(tampered), encoding="utf-8")
    assert (
        main(
            [
                "--verify-certificate",
                str(certificate_path),
                "--signing-key-file",
                str(key_path),
            ]
        )
        == 1
    )


def test_cli_requires_a_key(tk, tmp_path, monkeypatch):
    tk.load_table("sales", content=SALES_CSV)
    certificate_path = tmp_path / "certificate.json"
    certificate_path.write_text(json.dumps(_certificate(tk)), encoding="utf-8")
    monkeypatch.delenv("INSIGHT_SIGNING_KEY", raising=False)

    assert main(["--verify-certificate", str(certificate_path)]) == 2
