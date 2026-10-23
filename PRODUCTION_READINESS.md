# Production Readiness — Insight

This is the release gate for calling Insight **production-grade**. It is deliberately
tailored to Insight's contract — deterministic computation plus LLM narration — not a
generic SaaS checklist. Each criterion has an objective acceptance check; the status
column is evidence-backed and must be updated when behavior changes.

Last updated: 2026-10-23 · Scope: webapp + MCP server + toolkit · Out of scope: model quality.

Legend: **MET** = acceptance check passes and is covered by tests/docs · **PARTIAL** =
implemented but an acceptance check is unproven or a documented gap remains · **OPEN** =
not implemented.

> Note on evidence: the offline suite and local `pip-audit` ran on 2026-10-23; the new CI
> `security`/`docker` jobs land in the same commit, so their first hosted run is
> the final confirmation for C8.

## Status snapshot

| # | Criterion | Status | Evidence |
|---|---|---|---|
| C1 | Numbers are deterministic, auditable and verified | **MET** | `TEST.md` verification matrix; `verify_claims` + toolkit run records; grounding on by default |
| C2 | Provider failures are contained | **PARTIAL** | retry/fallback/budgets + typed errors; outage tests added in this pass; no live chaos drill |
| C3 | Authentication + per-client identity | **PARTIAL** | optional bearer/cookie auth gate added this pass; default stays open for trusted-LAN use; no user database/RBAC |
| C4 | Input and filesystem safety | **PARTIAL** | SQL sandbox + upload cap + non-root image; MCP file roots added this pass; archive-bomb limits not profiled |
| C5 | Rate and spend limits | **PARTIAL** | per-identity sliding-window limits for query/upload added; no provider-account spend cap |
| C6 | Observability + SLOs | **PARTIAL** | request IDs, JSON access log, `/metrics`, `/readyz` added; no alerting stack or published SLOs |
| C7 | Data lifecycle | **PARTIAL** | artifact retention sweeper added; session TTL added; no automatic disk quota |
| C8 | Delivery + supply chain | **PARTIAL** | CI now builds the image, scans it and audits deps; no lockfile/SBOM attestation |
| C9 | Scale / HA | **OPEN** | single-worker in-memory sessions; external session store still required for multi-worker |
| C10 | Release hygiene | **MET** | annotated tags + GitHub releases; `__version__` synced in this pass; changelog process documented |

## C1 — Correctness and auditability

The product promise: the LLM never executes code and never invents numbers.

Acceptance:
- Every displayed number is traceable to the execution that produced it, or is
  independently recomputed before the answer ships.
- The offline suite is green and is the release gate; a red suite blocks release.
- `verify_claims` rejects fabricated numbers and detects tampered run evidence.
- Determinism claim is scoped: numbers are deterministic given inputs; plans are not.

Current evidence: `TEST.md` → "Verification evidence" and "Toolkit + MCP real-data
smoke"; `tests/test_verification.py`, `tests/test_toolkit.py`, grounding/gateway tests.

Remaining: assert "zero ungrounded numbers shipped" as an end-to-end invariant across
all UI answer paths (today the guarantee is tested at the component level).

## C2 — Failure containment

Acceptance:
- Provider auth failure, 429 exhaustion, timeout or empty-content response always
  surfaces as a typed `InsightError`/`AllProvidersFailedError`; no partial answer.
- Fallback chain is exercised in tests (failover before first token, abort mid-stream).
- Bounded loops: planner repairs, agentic iterations and LLM calls have hard caps.

Current evidence: `insight/llm/providers/openai_compat.py` (retry/backoff/429),
`insight/orchestrator.py` (`_Chain` failover, stream abort), `tests/test_chain_fallback.py`
plus the outage tests added 2026-10-20.

Remaining: a live provider-outage drill against a staging deployment; queue-depth and
concurrency behavior under sustained throttle.

## C3 — Authentication and per-client identity

Acceptance:
- A deployment can require authentication for every non-health route with one env var.
- Auth supports (a) `Authorization: Bearer` for API/MCP-style clients, (b) a browser
  cookie established from a bootstrap token, (c) trusted-proxy identity headers.
- Session cookies are `HttpOnly` + `SameSite=Lax`; `Secure` is configurable for TLS.
- Artifacts are only served to the session that owns them.
- Default (unconfigured) mode is explicitly documented as trusted-network only.

Current evidence: `webapp/security.py`, `webapp/app.py`, settings
`INSIGHT_AUTH_TOKEN` / `INSIGHT_TRUST_PROXY_AUTH` / `INSIGHT_COOKIE_SECURE`;
tests in `tests/test_webapp_security.py`.

Remaining: user database/RBAC is out of scope; deployments needing SSO should put
Insight behind an authenticating reverse proxy and set `INSIGHT_TRUST_PROXY_AUTH=on`.

## C4 — Input and filesystem safety

Acceptance:
- SQL is read-only, single-statement, locked config, no external file access, bounded
  time and output rows.
- Uploads are size- and row-capped (per file, per request and in aggregate); rows are
  capped before a full parse and parsing failures are typed and safe.
- MCP/webapp path loading is restricted to configured roots; file reads are size-capped.
- The container runs unprivileged with no write access outside its artifact directory.

Current evidence: `insight/analytics/sql_engine.py` + `tests/test_sql_engine.py`; upload
caps in `webapp/app.py` (80 MB/file, 160 MB/request, 6 files, 256 MB normalized;
`INSIGHT_MAX_UPLOAD_ROWS`) and `insight/ingest.py` (64-sheet workbook cap); `AnalysisToolkit(file_roots=...)` +
`INSIGHT_MCP_FILE_ROOTS`; non-root `Dockerfile`. Tests: `tests/test_toolkit.py`,
`tests/test_mcp_server.py`, `tests/test_hardening.py`.

Remaining: archive/Excel bomb resource profiling; per-request total-memory ceiling.

## C5 — Rate and spend limits

Acceptance:
- Query and upload endpoints enforce per-identity sliding-window limits, configurable
  via env, returning 429 + `Retry-After`.
- Per-request LLM budgets are hard caps (planner repair, agentic iterations/calls).
- A deployment can cap aggregate spend independent of request count.

Current evidence: `webapp/ratelimit.py` middleware + unit/middleware tests; existing
budgets in `insight/settings.py` and agentic loop.

Remaining: provider-account spend caps and quota dashboards are deployment concerns
not implemented here; document the expected proxy/gateway controls.

## C6 — Observability and SLOs

Acceptance:
- Every request gets a request ID (inbound ID preserved), returned in `X-Request-ID`.
- One structured JSON access log line per request with method, path, status, duration.
- `/metrics` exposes request counts/duration and active sessions; `/readyz` distinguishes
  liveness from readiness (artifacts writable, a provider is usable).
- SLOs are published and measured before claiming a production SLA.

Current evidence: `webapp/telemetry.py`, `/metrics`, `/readyz`, tests in
`tests/test_webapp_telemetry.py`. Timings already exist in response meta.

Remaining: alert rules/error-budget policy and a load-test baseline on target hardware
(`tools/load_test.py` provides the harness; no published numbers yet); no
Prometheus/Grafana deployment is bundled.

## C7 — Data lifecycle

Acceptance:
- Sessions expire after a configurable idle TTL; the session store is bounded (LRU).
- Conversation history is bounded and busy sessions are never evicted or cleared.
- Generated artifacts older than a retention window are removable by a supported command
  and on startup; dry-run mode exists.
- No uploaded bytes are persisted to disk outside the process (documented); artifacts
  contain derived charts/reports only.

Current evidence: `WebSession` TTL/turn cap/busy flags in `webapp/sessions.py`; atomic
dataset replacement and query serialization in `webapp/app.py`; `insight/retention.py` +
`tools/retention.py`; startup sweep in `webapp/app.py`; tests in
`tests/test_sessions.py`, `tests/test_retention.py` and `tests/test_hardening.py`.

Remaining: automatic disk-quota enforcement; PII classification/erasure workflow.

## C8 — Delivery and supply chain

Acceptance:
- CI runs lint + offline suite on every push/PR.
- CI builds the container and fails on critical image vulnerabilities.
- CI audits declared dependencies and fails on known vulnerabilities.
- Releases are annotated tags + GitHub releases with honest evidence.

Current evidence: `.github/workflows/ci.yml` (test, security, docker jobs); releases
`v2.2.0`, `v2.3.0`. Remaining: lockfile + SBOM/attestation (image digest signing) is a
follow-up; current requirements use ranges.

## C9 — Scale and HA

Acceptance:
- More than one worker can serve requests without session loss or cross-session bleed.
- Session state is externalized or sticky sessions are a documented, tested deployment mode.
- A load test demonstrates the documented capacity envelope.

Current evidence: none — the Dockerfile and docs explicitly constrain deployments to one
worker. This is the largest remaining production gap.

Remaining: external session store (Redis or equivalent), multi-worker test, load baseline.
`SessionStore` already has the method surface (`get`/`get_or_create`/`drop`) to make this a
contained change; the blocker is serializing `WebSession` (DataFrames, responses) and
choosing a consistency model. Do not claim HA until this ships.

## C10 — Release hygiene

Acceptance:
- Semantic version tags, GitHub release notes with claims tied to `TEST.md` evidence.
- `insight.__version__` matches the latest release tag (checked at release time).
- Rollback = redeploy the previous git tag/image; DB/cache formats are versioned.

Current evidence: annotated tags + releases; version sync in this pass. Remaining: a
written rollback runbook and image digests in release notes.

## Release gate (must all be true before "production-ready")

- [x] Offline suite green + ruff clean (CI).
- [x] Dependency audit and image vulnerability scan are blocking CI jobs.
- [x] Auth gate, rate limits and artifact ownership checks exist and are tested.
- [x] Request IDs, access logs, `/metrics`, `/readyz` exist.
- [x] Session caps/TTL and artifact retention exist.
- [ ] External session store + multi-worker load test (C9) — **the only structural gate left**.
- [ ] Provider-outage drill on staging (C2) and SLO/alert policy (C6).
- [ ] Spend caps at the provider/gateway layer documented for the deployment (C5).

## Explicit non-goals

- Model-written code execution / sandboxed Python (no codegen by design).
- Multi-tenant RBAC, SSO, and user management inside the app (front with a proxy).
- SOC 2 / HIPAA certifications and formal data-processing agreements.
- Bundled Prometheus/Grafana/LangSmith stack (LangSmith tracing remains optional).

## How to verify this doc

```bash
pytest -q                                   # C1, C2, C3, C4, C5, C6, C7
python tools/smoke_toolkit.py               # C1 real-data validation
python -m pip_audit -r requirements.txt     # C8 dependency audit
docker build -t insight:local .             # C8 image build (then scan with trivy)
python tools/retention.py --days 14 --dry-run   # C7 cleanup preview
python tools/load_test.py --count 20 --workers 8   # C6/C9 upload-path load smoke
INSIGHT_AUTH_TOKEN=... uvicorn webapp.app:app --port 8000  # C3: / returns 401, Bearer works
```

Update the snapshot and the release-gate checkboxes whenever a criterion changes.
