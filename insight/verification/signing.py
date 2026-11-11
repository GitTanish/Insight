"""Optional HMAC-SHA256 signing for verification certificates.

The certificate remains valid without a signature. When a signing key is
configured, the certificate is signed over a deterministic JSON representation
of every field except ``signature`` itself. The dataset commitment in
``subject.dataset_sha256`` binds the certificate to the parsed root frame used
for execution without embedding the frame in the certificate.

HMAC is symmetric: only holders of the key can verify a signature. Public-key
(Ed25519) receipts are a separate, optional step.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
from typing import Any, Mapping, Optional

import pandas as pd

SIGNATURE_ALGORITHM = "HMAC-SHA256"


def _canonical_value(value: Any) -> Any:
    """Return a JSON-safe value with deterministic number handling.

    Non-finite floats become ``None`` because ``NaN``/``Infinity`` are not
    valid JSON and cannot be canonicalized portably. Tuples become lists;
    other objects (for example datetimes) become strings. The same function
    runs at signing and verification time.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return str(value)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize a payload deterministically for signing.

    This is a stable JSON subset, not full RFC 8785 JCS. It is sufficient for
    Insight-produced and Insight-verified certificates; an external verifier
    should use the same normalization rules.
    """
    normalized = _canonical_value(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def dataset_sha256(df: pd.DataFrame) -> str:
    """Commit to a dataframe's schema and row values.

    The digest includes column names, dtypes, row count, and pandas' row
    hashes. It is stable for a given pandas version. Callers that need
    cross-version reproducibility should record the pandas version alongside
    the certificate.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("dataset_sha256 expects a pandas DataFrame")

    schema = {
        "columns": [
            {"name": str(column), "dtype": str(dtype)}
            for column, dtype in df.dtypes.items()
        ],
        "rows": int(len(df)),
    }
    digest = hashlib.sha256()
    digest.update(b"insight.dataset.v1\n")
    digest.update(canonical_json_bytes(schema))
    row_hashes = pd.util.hash_pandas_object(df, index=False, categorize=False)
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def sign_payload(
    payload: Mapping[str, Any],
    key: str,
    key_id: str = "default",
) -> dict[str, str]:
    """Sign a certificate payload (all fields except ``signature``)."""
    if not key:
        raise ValueError("a non-empty signing key is required")
    resolved_key_id = key_id.strip() or "default"
    envelope = {
        "alg": SIGNATURE_ALGORITHM,
        "key_id": resolved_key_id,
        "payload": dict(payload),
    }
    mac = hmac.new(
        key.encode("utf-8"),
        canonical_json_bytes(envelope),
        hashlib.sha256,
    ).digest()
    return {
        "alg": SIGNATURE_ALGORITHM,
        "key_id": resolved_key_id,
        "value": base64.b64encode(mac).decode("ascii"),
    }


def sign_certificate(
    certificate: Mapping[str, Any],
    key: str,
    key_id: str = "default",
) -> dict[str, Any]:
    """Return a copy of ``certificate`` with a signature attached."""
    payload = {
        str(name): value
        for name, value in certificate.items()
        if name != "signature"
    }
    signed = dict(certificate)
    signed["signature"] = sign_payload(payload, key, key_id)
    return signed


def verify_certificate_signature(
    certificate: Mapping[str, Any],
    key: str,
    expected_key_id: Optional[str] = None,
) -> bool:
    """Return True only when the certificate carries a valid signature."""
    if not isinstance(certificate, Mapping) or not key:
        return False
    signature = certificate.get("signature")
    if not isinstance(signature, Mapping):
        return False
    if signature.get("alg") != SIGNATURE_ALGORITHM:
        return False
    key_id = signature.get("key_id") or "default"
    if not isinstance(key_id, str):
        return False
    if expected_key_id is not None and key_id != expected_key_id:
        return False
    value = signature.get("value")
    if not isinstance(value, str) or not value:
        return False

    payload = {
        str(name): item
        for name, item in certificate.items()
        if name != "signature"
    }
    expected = sign_payload(payload, key, key_id)["value"]
    return hmac.compare_digest(value, expected)
