# Verification certificate

`verify_claims` returns a versioned JSON certificate alongside its legacy
summary. The certificate is the machine-readable contract: an agent (or a CI
job) can inspect it without parsing prose or scraping log lines.

## Trust boundary

The `attestation_level` field is the first thing to read.

| Level | Meaning | What it can attest |
|---|---|---|
| `grounding` | The verifier received prose plus caller-supplied evidence (numbers, tables, notes). | Every numbered claim appears in the evidence you supplied. It does **not** attest that the evidence is correct. |
| `recomputed` | The verifier also received an execution trace (plan + results + source data). | The same, plus the reported aggregates survived a second arithmetic path. |

A certificate never claims that the chosen plan answers the question, that the
data is representative, or that a causal statement is true.

## Checks

The certificate has five named checks:

| Check | Hard failure? | What it covers |
|---|---|---|
| `numeric_grounding` | yes (ungrounded); warn (unverifiable threshold) | Numeric token extraction, span mapping, evidence binding, rounding at the claimed precision, and `p < 0.001`-style inequality evaluation. |
| `independent_recomputation` | yes | Second arithmetic path for `group_aggregate`, `top_n`, `value_counts`, `correlation`, `compare_subsets`, `statistical_test`, `anova_test`, `regression`. Chained and SQL steps are reported as `not_covered`. |
| `statistical_validity` | no (warnings) | Small samples (`n < 30`) and missingness (`> 5%`, MCAR caveat). |
| `causal_language` | no (advisory) | Causal verbs without a randomisation / quasi-experiment marker. |
| `structural` | yes | Failed steps, empty tables, non-finite values, missing chart sources. |

Only `fail` on a hard-failure check makes `verdict.overall` equal `fail`.
Warnings and skipped checks produce `pass_with_warnings`.

## Claim statuses

| Status | Meaning |
|---|---|
| `grounded` | The token matched evidence; `evidence_ids` names the source(s). Matching is by value/rule, not by semantic role, so a grounded number can still be attached to the wrong predicate. |
| `ungrounded` | The token did not match any supplied evidence, or a `p < ...` threshold was contradicted by the evidence. |
| `ignored` | The token is a year, ordinal, or date component, not a claim. |
| `flagged` | A causal phrase was found without a design marker. |
| `not_checked` | A numeric inequality had no unambiguous p-value evidence, or a causal phrase was found alongside a design marker whose design itself was not verified. |

## Grounding and recomputation rules

- **Exact match** compares numeric values at roughly six decimal places, with
  relative closeness at small magnitudes so tiny values do not collapse to
  zero. Scientific notation is tokenized as one number (`4.794e-16`, not
  `4.794` plus `-16`), so a fabricated `1e-9` cannot ground a rounded-away
  `p_value = 0.0`.
- **Rounded match** accepts `8.55` for evidence `8.554537`, but only when
  `round(evidence, claimed_decimals) == claim`. A merely close fabricated
  value such as `8.56` still fails. Magnitude matching (`683` against
  `-683.411797`) is recorded separately as `rounded_magnitude_match`.
- **Percentage scaling** only applies when the narrative token carries a `%`.
- **Inequality thresholds** such as `p < 0.001` are evaluated against
  evidence whose label or column is `p_value`; contradictory thresholds fail,
  and ambiguous or absent p-value evidence becomes `not_checked`, not a silent
  pass.
- **Recomputation is per aggregate.** `verified`, `mismatch`,
  `not_covered`, and `not_checked` are attached to reported calculations.
  Text calculations such as an interpretation string are `not_checked`; a
  mismatch named for one aggregate does not mark the step's other numeric
  aggregates as mismatched.
- **`statistical_test` is independently recomputed.** The verifier re-routes
  the test on the same frame and checks the test name, statistic, p-value,
  effect size, each per-group `mean_*` calculation, and the derived
  `difference`. SQL and chained (`input_step`) steps remain `not_covered`.

## Example

```json
{
  "schema_version": "1.0",
  "certificate_id": "d9264997e80c4147a683eae1ee0d2580",
  "issued_at": "2026-11-07T14:41:34.013997Z",
  "verifier": {
    "name": "insight-verifier",
    "version": "0.2.0",
    "engine": "pandas/scipy/duckdb",
    "python": "3.11.9"
  },
  "attestation_level": "recomputed",
  "subject": {
    "answer_sha256": "04039b47...",
    "run_id": "c72de3fcc074",
    "plan_digest": "72875d86...",
    "execution_digest": "8ba7704a...",
    "dataset_sha256": "9f2c47d1..."
  },
  "tables": [
    {
      "name": "grouped",
      "columns": ["region", "total"],
      "rows": 4,
      "truncated": false,
      "digest": "e3b0c442..."
    }
  ],
  "evidence": [
    {
      "id": "table:grouped:r1:total",
      "kind": "table_cell",
      "label": "grouped[1].total",
      "value": 150,
      "numbers": [150],
      "table": "grouped",
      "row": 1,
      "column": "total"
    }
  ],
  "aggregates": [
    {
      "step_id": 0,
      "operation": "group_aggregate",
      "label": "groups",
      "value": 4,
      "recomputation": "verified"
    }
  ],
  "claims": [
    {
      "id": "n0",
      "kind": "numeric",
      "text": "150",
      "span": [17, 20],
      "status": "grounded",
      "evidence_ids": ["table:grouped:r1:total"],
      "rule": "exact_numeric_match",
      "severity": "info",
      "normalized": "150",
      "normalized_value": 150.0
    },
    {
      "id": "c0",
      "kind": "causal",
      "text": "drove",
      "span": [24, 29],
      "status": "flagged",
      "rule": "CAUSAL_LANGUAGE_WITHOUT_IDENTIFICATION",
      "severity": "warning",
      "message": "Causal language appears without a randomised or quasi-experimental design marker in the plan or notes."
    }
  ],
  "checks": {
    "numeric_grounding": {"status": "pass", "checked": 2, "failed": 0},
    "independent_recomputation": {"status": "pass", "checked": 1, "failed": 0},
    "statistical_validity": {"status": "pass", "checked": 1, "failed": 0},
    "causal_language": {"status": "warn", "checked": 1, "failed": 1},
    "structural": {"status": "pass", "checked": 1, "failed": 0}
  },
  "verdict": {
    "overall": "pass_with_warnings",
    "grounded": true,
    "recomputed": "pass",
    "statistical_validity": "pass",
    "causal_overreach": "warn",
    "structural": "pass"
  },
  "limitations": [
    "Numeric grounding checks whether a number appears in the supplied evidence; it does not prove the evidence is true or that it is the right evidence for the claim.",
    "Causal checks are advisory and rule-based; they do not establish causation."
  ],
  "signature": null
}
```

The example is unsigned. With `INSIGHT_SIGNING_KEY` configured, `signature`
contains the HMAC receipt described below.

## Local HTTP surface

For pipelines that do not speak MCP, run the local FastAPI endpoint:

```bash
uvicorn insight.verify_service:app --host 127.0.0.1 --port 8787
```

`POST /v1/verify` accepts either:

- `evidence`: `{ "numbers": [...], "tables": [...], "notes": "..." }`, or
- `data`: `{ "csv": "...", "table_name": "..." }` plus exactly one of
  `operation`: `{ "name": "...", "params": {...} }` or `sql`: `"SELECT ..."`.

It returns the certificate with HTTP 200 even when `verdict.overall == "fail"`;
a failed verification is a valid result, not a transport error. Set
`INSIGHT_VERIFY_AUTH_TOKEN` to require `Authorization: Bearer ...`. This is a
local/dev surface, not a multi-tenant service.

## Signing

Signing is optional and backward compatible. When `INSIGHT_SIGNING_KEY` is set
(or `signing_key` is passed to `AnalysisToolkit`), the certificate is signed
with HMAC-SHA256 over a deterministic JSON representation of every field except
`signature` itself. The signature is attached as:

```json
"signature": {
  "alg": "HMAC-SHA256",
  "key_id": "default",
  "value": "<base64 MAC>"
}
```

`subject.dataset_sha256` commits to the parsed root frame used for execution:
its column names, dtypes, row count, and row hashes. It does not embed the data.
The commitment is stable for a given pandas version.

Verify a signed certificate without re-running the analysis:

```bash
python -m insight.verify_cli --verify-certificate certificate.json \
    --signing-key-file signing-key.txt
```

The command exits `0` for a valid signature, `1` for an invalid/tampered
certificate, and `2` for a usage or key error. `INSIGHT_SIGNING_KEY` can be used
instead of `--signing-key-file`.

HMAC is symmetric: anyone who can verify also holds the signing key. Public-key
(Ed25519) receipts for external auditors are future work. A signature binds the
certificate fields; it does not prove the analysis is correct.

## Generating the JSON Schema

```python
import json
from insight.verification import VerificationCertificate

print(json.dumps(VerificationCertificate.model_json_schema(), indent=2))
```

## Design notes

- **Evidence is cited, not re-summarised.** A grounded claim points at a
  calculation, table cell, column, note, or warning. The certificate includes
  only the table cells that are actually referenced, so it stays small even
  when the source table is large.
- **Ignored tokens are explicit.** Years, ordinals, and date components are
  listed with a reason rather than silently dropped.
- **Skipped checks are visible.** `not_checked` is not a pass. `overall`
  becomes `pass_with_warnings` when a check was skipped.
- **Causal flags are advisory.** They are a prompt to inspect the analysis,
  not proof that the causal statement is false. The lexicon is intentionally
  small and deterministic; it is not a semantic entailment model.
- **Signing is opt-in.** With a configured key, the certificate binds its
  fields (including `subject.dataset_sha256`, plan/execution digests, checks,
  and claims) with HMAC-SHA256. Without a key, `signature` stays `null`. The
  signature proves integrity and provenance under a shared key; it does not
  prove the underlying analysis is correct.
