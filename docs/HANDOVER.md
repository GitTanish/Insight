# INSIGHT - Session Handover

> Last updated: 2026-11-08 · Engine v2.4.0 · **offline suite green** · **20/20 live eval cases** · **MCP server + verification toolkit + versioned certificate (validated on real data)** · **standalone verifier CLI + local HTTP** · **production-hardening pass (auth, limits, telemetry, retention, CI scanning)** · Python 3.11 venv at `.venv`
> Branch note: `main` is protected and requires a PR. Work happens on short-lived branches; there are no long-lived feature branches.

---

## 1. What this project is

**Insight** is a validated, model-agnostic analytical agent for CSVs.
Core principle: **the LLM never executes code and never invents numbers.**

```
Upload → Profiler + Zero-Prompt Briefing (top-3 findings, no question needed)
      → Planner (LLM) emits a JSON AnalysisPlan → schema/column validated → bounded repair
        (or, in agent mode: tool-calling loop - ops as tools, bounded, observes each result)
      → [optional human gate: editable plan, Approve & run]
      → Deterministic engine executes 15 ops (chained via input_step DAG)
      → Statistical router runs scipy tests by column types (t-test / Mann-Whitney / chi²+Cramér's V)
      → Validator checks sanity + INDEPENDENTLY RECOMPUTES headline numbers (2nd code path)
      → Explainer (LLM) narrates using only computed values, STREAMED token-by-token
      → SSE to UI · DOCX report incl. briefing findings · optional LangSmith trace
```

Both front-ends share one engine (`insight/`): **FastAPI webapp** (primary, `webapp/`) and **Streamlit** (`legacy_streamlit/main.py`, maintained).

---

## 2. Repo map (what lives where)

| Path | Role |
|---|---|
| `insight/domain/` | Pydantic contracts: `DatasetProfile`+fingerprint, `AnalysisPlan`(steps+charts+input_step), `ExecutionResult`, `ChartSpec`, `Evidence`, typed errors |
| `insight/llm/` | `base.py` ABC; `providers/openai_compat.py` (retry/backoff/429-aware) + groq/mistral/openai/ollama/openrouter/anthropic; `registry.py`: curated catalog + live discovery + capability flags + fallback chain + custom env provider |
| `insight/profiling/profiler.py` | role inference (identifier-by-name heuristic!), missingness, fingerprint |
| `insight/planning/` | `prompts.py` (hard rules incl. title/DAG/binary rules), `planner.py` (JSON repair loop) |
| `insight/analytics/operations.py` | **the op registry** - 15 deterministic ops + `operation_catalog()` (auto-generates planner prompt) + `validate_plan_columns()` |
| `insight/tools/` | **agentic tool-use layer** - `tooling.py` (registry→provider tool schemas), `runtime.py` (dispatch through the executor path), `loop.py` (bounded observe/plan/act/refine), `pipeline.py` (plan/execution rebuild + chart synthesis + meta). Gated by `INSIGHT_AGENTIC` / `AnalysisRequest.agentic` |
| `insight/analytics/statistics.py` | scipy routing: group comparison / chi-square / correlation, effect sizes, plain-language interpretation |
| `insight/briefing.py` | zero-prompt detectors: trend/corr/outliers/imbalance/missing/dupes → ranked findings |
| `insight/execution/executor.py` | sequential executor; `frames` dict enables `input_step` DAG; filter_rows mutates implicit chain |
| `insight/validation/` | `result_validator.py` (sanity/small-sample/chart-consistency) + `verification.py` (**independent recomputation**) |
| `insight/verification/` | `certificate.py` (versioned certificate contract, evidence binding, per-aggregate recomputation status, verdict) + `causal.py` (advisory causal lexicon); JSON Schema at `schemas/verification_certificate.schema.json` |
| `insight/verify_cli.py` / `insight/verify_service.py` | Standalone verifier CLI (evidence, data + operation, data + SQL) and local FastAPI `POST /v1/verify`; no LLM |
| `insight/ingest.py` | shared CSV/Excel parsing (delimiter sniffing, encodings, row caps) used by the webapp and the toolkit |
| `insight/toolkit.py` | **agent-facing engine API** - load tables, run ops/plans/sandboxed SQL, and `verify_claims`; records runs for grounding, independent recomputation, and certificate evidence. No LLM |
| `insight/mcp_server.py` | MCP wrapper over the toolkit (`python -m insight.mcp_server`; stdio / sse / streamable-http) |
| `insight/visualization/` | `themes.py` (editorial/minimal/dark) + `renderer.py` (ChartSpec→PNG; adaptive bins live in distribution op; Other-bucket, horizontal-bar, tick formatting here) |
| `insight/orchestrator.py` | `analyze_stream()` async generator (stage / delta / plan_review events) wrapped by `analyze()`; `run_analysis_sync()` for sync UIs. Owns cache, timing meta (`planning_ms`, `repair_ms`, `first_token_ms`, …) and the HITL gate |
| `insight/observability.py` | LangSmith tracing (optional; `INSIGHT_TRACING=off` kills it) - root run + `trace_phase` spans |
| `insight/settings.py` | configuration model; operational settings via env (`.env` auto-loaded); model IDs and generation limits remain code defaults |
| `webapp/` | FastAPI: `app.py` routes, `sessions.py` cookie store, `services.py` thin webapp IO (delegates parsing to `insight/ingest.py`), Jinja `templates/`, `static/` (CSS + app.js SSE client w/ typing bubble + plan review + live agent tool trail) |
| `legacy_streamlit/` | Streamlit legacy UI (same engine via cached wrappers) |
| `tests/` | Offline suite, green in CI (fake providers / scripted chains). Key fixtures in `conftest.py`. Pass/skip split varies by platform and by which optional corpora are present - assert exit code, not count |
| `docs/` | Runbook (`HOW_TO_RUN.md`), handover, test manual, certificate contract (`CERTIFICATE.md`), and forward plans |

---

## 3. Run & test

```bash
# activate .venv first (Windows: .venv\Scripts\activate)

# primary UI
uvicorn webapp.app:app --port 8000        # → http://localhost:8000

# legacy UI
streamlit run legacy_streamlit/main.py     # → :8501

# tests (no network / no keys needed - tracing force-off in conftest)
pytest -q                                  # expect exit 0 (~30-70s on Windows)

# headless API
curl -X POST localhost:8000/api/query -H "Content-Type: application/json" \
     -d '{"question":"Which region has the highest revenue?"}'

# toolkit + MCP smoke on real public data (manual, not CI; first run downloads 4 CSVs
# into data/ - needs evaluation/data/, generated once via evaluation/generate_datasets.py)
python tools/smoke_toolkit.py               # 54 checks: 15 ops + SQL sandbox + DAG + verify_claims + MCP

# standalone verifier (no LLM): certificate demo, CLI, local HTTP
python tools/demo_epistemic_firewall.py
python -m insight.verify_cli --answer "North leads with 150." --data sales.csv \
  --operation group_aggregate --params-file params.json
uvicorn insight.verify_service:app --host 127.0.0.1 --port 8787

# production ops (see ../PRODUCTION_READINESS.md)
python tools/retention.py --days 14 --dry-run   # artifact retention preview
python tools/load_test.py --count 20 --workers 8  # upload-path load smoke (no LLM)
python tools/load_test.py --query --question "Which region leads revenue?" --count 8
# with auth enabled: add --auth-token ... ; monitor GET /readyz and GET /metrics
```

Env: copy `.env.example` → `.env`. Minimum `GROQ_API_KEY=gsk_...`.
Optional: MISTRAL/OPENAI/ANTHROPIC/OPENROUTER keys, `OLLAMA_BASE_URL`,
`INSIGHT_CUSTOM_BASE_URL/_API_KEY/_MODELS/_NAME` (any OpenAI-compatible service),
`LANGSMITH_API_KEY` (+`LANGSMITH_PROJECT`) for traces.

---

## 4. Verified-working highlights (do not regress)

- Live E2E vs Groq gpt-oss-120b: plan→execute→validate→chart→explain ≈ 2.5-6 s
- Mistral-small full pipeline incl. self-repair recovery
- OpenRouter key verified end-to-end (claude-sonnet-4, gpt-4o-mini)
- Webapp E2E smoke: upload → briefing on page → SSE stages → result HTML → history persist → DOCX(83 KB, w/ findings) → JSON API → clear
- Editorial charts render correctly (paper #f2ead8 / ink / accent red, serif)
- LangSmith hierarchy: `insight.analyze` → `planning.plan` → `provider.generate`
- Agentic pinned A/B on NVIDIA NIM gpt-oss-20b (k=3, `--no-fallback`): measured 58/60 (96.7%, 95% CI 88.6–99.1) vs single-shot 43/60 (71.7%, 59.2–81.5) - but the 2026-10-17 follow-up showed **most of that gap was a provider-side truncation bug that only hit the single-shot planner** (see TEST.md "What failed" #3); post-fix k=1 confirmations: 20/20 agentic vs 19/20 single-shot - **accuracy parity on this endpoint; the agentic edge is auditability + tool trail**. Raw numbers: TEST.md "Verification evidence". **Caveat (2026-11-03):** the historical agentic arm scored the loop's final answer directly and did not run `validate_execution` or the grounding tail, so those numbers are loop-only, not an end-to-end A/B. Use the Phase 5 harness for production-path claims.
- Toolkit + MCP server smoke-verified on real data (`tools/smoke_toolkit.py`, 2026-10-20): 54/54 checks across superstore / penguins / telco / temps + the generated corpora vs independent pandas/numpy/scipy and `ground_truth.json` (in-sample raw Brier 0.17746 / ECE 0.0613), including the MCP in-memory client flow; the pass caught + fixed a double-bracket `filters` shape hint in the auto-catalog and a `SettingWithCopyWarning` on ops mutating filtered frames (pinned by 3 new `test_operations.py` tests)
- **Verifier certificate + CLI + HTTP + MCP verified on real data (2026-11-07)**: superstore group aggregate grounds and recomputes; fabricated `999,999` exits 1; a causal narrative warns (exit 0, or 1 with `--fail-on-warn`); SQL claims ground while recomputation is correctly `not_checked`; Palmer penguins `statistical_test` grounds rounded means, derived `difference`, `d`, `t`, `p`, and `p < 0.001`, and recomputes them; live HTTP on :8787 covered evidence/execute/SQL/invalid-request paths; bearer auth on :8788 returned 401 without/with a wrong token and 200 with the right one; MCP `load_table → execute_operation → verify_claims` passed a correct answer and failed a fabricated one. Full offline suite green; ruff clean.

---

## 5. Known quirks / gotchas

1. **Windows console cp1252** chokes printing unicode (`…`, `\u202f`) - scripts need `sys.stdout.reconfigure(encoding="utf-8")`; the *library* is fine.
2. **PowerShell heredocs add BOM** with `-Encoding UTF8` on PS5.1 - after Set-Content edits, strip BOM or read as utf-8-sig when parsing.
3. **sse framing**: webapp hand-formats SSE (`event:\ndata:\n\n`, LF). Client normalizes CRLF and skips `:` comments anyway.
4. **uvicorn cold start ~8-12 s** (scipy+langsmith imports) - not hung.
5. **`st.cache_data` wrappers** in `legacy_streamlit/utils.py` are Streamlit-only; core logic duplicated cleanly in `webapp/services.py` (keep both in sync when changing CSV parsing).
6. **Tracing in tests**: conftest sets `INSIGHT_TRACING=off` BEFORE insight imports - preserve that line's position.
7. **Discovery cap**: `discovery_max_per_provider=40`; OpenRouter ranking = preferred vendors then shortest-name-first; variants (`:batch`) deduped.
8. **Planner quirk**: small models may set `top_n=1` (charts suffer but numbers stay correct); ChartSpec treats explicit `null` as "use default" (`resolved_top_n/resolved_sort_desc`).
9. **Artifacts**: plots under `.artifacts/web/<data_id>/` (webapp) vs `.artifacts/<session_id>/` (streamlit). Both gitignored.
10. **PowerShell env pinning is a trap**: `$env:FOO=''` *removes* the variable; dotenv then refills it from `.env`. That's why live A/B runs use `--no-fallback` (chain pinned in `build_llm_chain`) rather than blanking keys.
11. **Model-name collision**: Groq and NIM both serve a model literally named `openai/gpt-oss-20b`, so `models_used` alone cannot prove which provider served a run - pin with `--no-fallback` when it matters.
12. **Quota backoff shape differs**: NIM throttles in bursts (220–330 s single-case walls observed unpinned); Groq's free tier stretches every case (~178 s/case) via internal retry backoff, which leaves `throttled=0` even when the run is quota-bound.
13. **Reasoning-capable models need their capability flag declared**: NIM `gpt-oss-20b` without `supports_reasoning_effort` silently loses the configured `reasoning_effort="low"`; it then burns the whole `max_tokens` budget in the reasoning channel and returns `finish_reason=length` with empty content. Fixed 2026-10-17 (registry flag + `hit max_tokens=N` hint in the empty-content error) - when curating *new* NIM/reasoning models, set the flag deliberately.
14. **The public superstore / telco CSV copies are polluted (2026-10-20)**: `leonism/sample-superstore` concatenates unrelated sample tables after the real 9,994 rows of `data/superstore.csv` (a People/Region block, then Returns) - slice to the first block (`iloc[:9994]`) for any manual use, as `tools/smoke_toolkit.py` does. The `KimathiNewton/Telco-Customer-Churn` copy stores `SeniorCitizen` as `False/True/0/1` strings and blanks in `TotalCharges`; the ops' numeric coercion handles it, but pandas `== 1` comparisons silently match nothing.
15. **Middleware order is load-bearing (2026-10-20)**: the last middleware added is the outermost. `webapp/app.py` adds RateLimit → Auth → SecurityHeaders → Telemetry → GZip in that order, so telemetry sees the final status (including 401/429) and auth runs before routing. Tests force `INSIGHT_RATE_LIMIT=off` in `conftest.py`, so the main app never throttles under test - exercise limits on minimal apps with an injected fake-clock limiter.
16. **MCP HTTP transports require auth (2026-10-20)**: `sse` / `streamable-http` refuse to start without `INSIGHT_MCP_AUTH_TOKEN` (override only deliberately with `INSIGHT_MCP_ALLOW_UNAUTHENTICATED=1`). `load_table(path=...)` defaults to the CWD as its only file root; `INSIGHT_MCP_FILE_ROOTS=all` is unrestricted and `none` disables path loading entirely.
17. **PowerShell mangles inline JSON in the verifier CLI**: `--params '{"..."}'` can be rewritten by the shell. Write the parameters to a file and use `--params-file`; `--params` and `--params-file` are mutually exclusive by design.
18. **Evidence-only `verify_claims` uses the most recent run when `run_id` is omitted**: in a long-lived MCP session, an evidence-only call made after an execution is labelled `recomputed`. Pass an explicit `run_id` or use a fresh toolkit when you need `attestation_level: "grounding"`.

---

## 6. Remaining work - build queue (priority order)

> **Status 2026-11-07:** P1–P12 below are DONE (P8 verified by the pinned live
> A/B, P9 by the real-data smoke, P10 by the offline suite + `pip-audit`, P11 by
> the offline hardening suite, P12 by the certificate/CLI/HTTP/MCP trials - see
> TEST.md "Verification evidence", "Toolkit + MCP real-data smoke", and
> `docs/CERTIFICATE.md`). The remaining production gap is external
> session state for multi-worker (C9 in `../PRODUCTION_READINESS.md`); see
> `future_scope.md` for product work.

### P1 - Evaluation harness
`evaluation/generate_datasets.py` (seeded: sales/ecommerce/support + retail_250k;
diamonds.csv fetched separately) · `evaluation/cases.json` (20 cases w/ ground truth)
· `evaluation/run_eval.py` (ops superset + substrings + numeric ±tol scoring,
markdown report). pytest `eval` marker gated by `INSIGHT_RUN_EVAL=1`.
Live results: 20/20 (gpt-oss-120b 3/3 smoke; mistral-small 12/15 pre-fixes → fixed).

### P2 - Statistics depth
`anova_test` (F + η²), `regression` (OLS lstsq, t p-values, R², std betas,
listwise-drop reporting), compare_subsets 95% CIs (n≥30 gate). Router: cat>2×num → ANOVA;
binary-num × categorical → chi-square (fixed swapped-args bug found by eval).
`isotonic_calibration` (PAV, reliability table, Brier/ECE) added on top.

### P3 - Conversation state machine
`insight/conversation/state.py`; orchestrator `state=` param end-to-end; planner gets
serialized context; webapp + streamlit persist `analysis_state`; cache key includes it.

### P4 - Packaging
`Dockerfile` (non-root, single-worker note) · `.dockerignore` · `ruff.toml` +
`.github/workflows/ci.yml` (ruff+pytest, tracing off) · query cache in `insight/cache.py`
(TTL 14d / 512MB budget eviction; `INSIGHT_QUERY_CACHE*` envs; valid-results only).

### P5 - Smaller polish
Webapp `POST /models/refresh` + rail button · DOCX consolidated to `insight/reports/docx.py`
(both UIs delegate) · briefing LLM polish behind `INSIGHT_BRIEFING_LLM_POLISH` (default off)
· validator now emits MCAR caveats for referenced columns >5% missing · group_aggregate
reports `excluded_missing_<metric>` for mixed count bases.

### P6 - DuckDB relational analytics + artifact exports
`insight/analytics/sql_engine.py` (`DuckSession`: in-memory con, frames registered then
`enable_external_access=false` + `lock_configuration=true`, SELECT/WITH-only keyword
guards, chained-statement block, interrupt watchdog `INSIGHT_SQL_TIMEOUT_S=10`,
row cap). Op 15/15: `sql_query` - joins/windows/pivots; errors feed the existing
plan-repair loop. Multi-file upload (`files[]`, up to 6) → per-session tables;
planner gets exact table schemas via `DuckSession.describe()`. Exports:
`POST /export/dbt` (zip: model.sql + schema.yml w/ not_null tests) and
`GET /export/alert?step_id=` (Slack-ready monitor JSON), buttons render under SQL
answers. Tests: `test_sql_engine.py` (16), `test_sql_webapp.py` (2),
`test_artifacts.py` (2). Sandbox code-interpreter explicitly skipped by owner.

### P7 - Latency work (streaming, caching, plan approval)
**Text streaming**: `LLMProvider.stream()` default + real SSE parser in
`openai_compat.py` (yields `choices[].delta.content`, falls back to non-streaming
on endpoint rejection). `_Chain.stream()` will not splice two providers mid-answer.
Orchestrator emits `{"type":"delta"}`; the explainer AND the grounding rewrite
stream (the latter preceded by `delta_reset`). `app.js` renders a live typing
bubble with a caret (reduced-motion aware), replaced by the final HTML render.
`first_token_ms` measured ~1.0-2.0s vs a multi-second blank wait.

**Prefix caching**: session state/history/question moved OUT of the planner
system prompt into later messages so the prefix is byte-stable per dataset
(OpenRouter `prompt_cache_key`, Anthropic ephemeral breakpoint, cached-token
parsing → `meta.cached_prompt_tokens`).

**Cache on the SSE path** (was `/api/query`-only): ~0.23-0.39s repeats vs
7.8s cold. `_scope_signature` keys on ACTIVE FILTERS only, so a question repeats
into cache even though the session learned things meanwhile. Approved/plan-only
requests are excluded from cache read AND write.

**Timing telemetry**: `planning_ms`, `repair_ms`, `repair_attempts`,
`repair_issues`, `charts_ms`, `grounding_retry_ms`, `first_token_ms`,
`explaining_ms` - a repair loop was previously invisible (31s unaccounted).
`repair_issues` initially landed in the LangSmith span block instead of the
response meta; watch for that class of mistake.

**Repair budget**: `INSIGHT_MAX_REPAIR_ATTEMPTS` (default `1`, `0` disables)
bounds both planner-internal schema repair and the orchestrator's
execute/validate repair loop. The two paths share the same setting instead of
the result-repair path hardcoding a single retry.

**Background briefing**: LLM polish moved to `BackgroundTasks` so the upload
redirect never waits on it. `findings_status` guards with a `data_id`
generation check so a slow polish can't clobber a newer upload.

**HITL plan approval**: `INSIGHT_PLAN_APPROVAL=on` (or `plan_only` per request)
stops after planning, emits `plan_review` with editable JSON and a server-issued
approval (dataset id, generation, plan hash, issue/expiry). `approved_plan`
re-validates via `validate_plan_columns` rather than trusting the edit, and the
webapp/API require an approval matching the current dataset/generation and
unexpired window; library callers may omit the binding and are marked unbound.
`analyze()` raises on `plan_only` (it can't express a plan-only outcome).
`/api/query` drives `analyze_stream` internally for this.

**Evidence ledger**: `INSIGHT_AUDIT_LEDGER=on` appends hash-chained JSONL records
for plan reviews, completed analyses, and pre-result failures
(`insight/audit.py`). Verify with `python -m insight.audit`. Writes use a process
lock plus `flock`/`msvcrt` and `fsync`; a write failure is logged, never fatal to
the answer. Tail truncation is not detectable without an external head anchor.

**Non-decision - native engine (C++/Rust/Go) was measured and rejected**:
execution is 14-62ms of a ~30s query (~0.2%), even on 250K rows. A native
rewrite would save ~55ms. The LLM calls are 95%+ of latency. Documented in
`future_scope.md` so it isn't re-litigated.

### P8 - Agentic tool-use mode + alert-runner
**Agentic mode** (`INSIGHT_AGENTIC` / per-question toggle in both UIs): the LLM
calls ops *as tools*, observes real results, iterates. `insight/tools/`:
`tooling` (op params → provider tool schemas, `$defs` inlined, Anthropic +
OpenAI shapes), `runtime.dispatch` (validates through `validate_plan_columns`
then runs the same `OperationSpec.run`; chained calls skip profile
canonicalization because derived columns only exist in the chained frame),
`loop.run_loop` (bounded: `LoopBudget`, call delay, identical-failure abort,
`input_step` DAG mirroring the executor's working/frames semantics, `on_event`
for SSE), `pipeline` (rebuilds `AnalysisPlan`+`ExecutionResult` from the loop
so the **same** tail runs: validate → independent recompute → charts →
explainer → grounding). Anthropic provider now speaks tool_use/tool_result.
`_Chain.generate` tolerates tool-call-only (empty-content) responses.
Orchestrator: the shared tail was extracted to `_tail_events` - both modes
call it; agentic skips the plan-repair loop (the loop *is* the repair).
Streams `tool_call`/`tool_result` SSE events; `app.js` renders a live trail.
Cache key includes the mode. Tests: `test_agentic_pipeline.py` (20),
`test_webapp_agentic.py` (4).

**Verification (2026-10-16/17 deep pass)**: pinned k=3 A/B on NIM `gpt-oss-20b`
(no fallback) → measured 58/60 (96.7%) agentic vs 43/60 (71.7%) single-shot;
the 2026-10-17 diagnosis then showed 13 of the 17 baseline failures were a NIM
planner-truncation bug (missing `reasoning_effort` flag → reasoning consumed
`max_tokens` → empty content, reproduced 4/4), fixed same-day; post-fix k=1
confirmations: agentic 20/20, single-shot 19/20 (**parity**). Wire-format
fix + inbound-parser tests in `test_agentic_pipeline.py`; harmony-token
sanitizer for inbound tool names (also 2026-10-17); recorded replay
fixture (`tests/fixtures/agentic_replay_sales_top_region.json` →
`test_agentic_replay.py`); `tools/mutate.py` sampled mutations 11/11 killed;
`loop.py` 96% branch coverage; clean-clone bootstrap green (283 passed /
6 skipped before datasets, 286/3 after `generate_datasets.py`); live SSE trail
smoke `tools/sse_smoke.py` (tool_call → tool_result ordering, streamed deltas).
Details: TEST.md "Verification evidence".

**Gotchas**: (1) the SSE drain must `yield await queue.get()` - yielding the
bare coroutine never drains the queue and spins the generator forever.
(2) Stub chains in tests lack `.entries` - the tool-shape check uses
`getattr`. (3) `_FakeSettings` in `test_cache.py` needs an `agentic` field.
(4) Plan approval takes precedence over agentic mode: `plan_only` /
`INSIGHT_PLAN_APPROVAL` forces the single-shot planner, because the gate needs
a reviewable plan *before* anything executes.

**Alert-runner** (`tools/alert_runner.py`, roadmap #5): cron → DuckDB →
webhook worker for exported monitors. Reuses `DuckSession` (same sandbox:
statement validation, keyword guards, interrupt watchdog, row cap).
State-change alerting (post on trigger edge, `--notify-recovery`,
`--repeat`), `--dry-run`, `--once`, `--max-runs`, `--monitor <name>` filter,
`--webhook` override, URL from the monitor's `webhook.url_env`. Without
`--once`/`--interval`, each monitor's `schedule_cron` (five-field local time;
`*`, ranges, lists, `/step`) drives per-monitor due times, polled every
`--poll` seconds. Fired state and last-run times persist atomically to
`--state` (default `.artifacts/alert_state.json`; off by default under
`--dry-run`) so restarts do not re-post and recovery is detected. `--interval`
overrides cron. One worker per state file: there is no cross-instance lock.
Stdlib-only. Tests: `test_alert_runner.py`.

### P9 - MCP server + agent verification toolkit
`insight/toolkit.py`: agent-facing engine API (no LLM) - load tables (CSV/Excel
via shared `insight/ingest.py`), run ops / DAG plans / sandboxed SQL, and
`verify_claims` (number grounding in prose + independent recomputation + a
versioned certificate); every run recorded so evidence can be replayed and
tamper-checked.
`insight/mcp_server.py`: FastMCP wrapper, 8 tools (list_operations, load_table,
list_tables, profile_table, execute_operation, run_plan, run_sql, `verify_claims`),
`python -m insight.mcp_server`, stdio / sse / streamable-http via
`INSIGHT_MCP_*` envs; `webapp/services.py` now a thin re-export of the shared
parser. Tests: `test_toolkit.py` (25) + `test_mcp_server.py` (7, in-memory
client). **Verification (2026-10-20)**: `tools/smoke_toolkit.py` - 54/54 checks
on four public datasets + the generated corpora across all 15 ops, SQL sandbox
rejections, DAG chaining, verification gates and an MCP end-to-end flow, each
cross-checked against independent pandas/numpy/scipy results and
`ground_truth.json` (retail_250k load 1–15 s, filtered group_aggregate
0.04–0.12 s). The pass fixed a double-bracket `filters` shape hint in
`operation_catalog()` (models require a flat `list[Filter]`) and made
`apply_filters` results mutation-safe (was emitting `SettingWithCopyWarning` on
filtered group ops) + 3 regression tests. Details: TEST.md "Toolkit + MCP
real-data smoke".

### P10 - Production hardening
`../PRODUCTION_READINESS.md` is the gate (C1-C10, acceptance checks, non-goals).
Shipped in this pass: optional auth gate (`INSIGHT_AUTH_TOKEN`, `/auth` bootstrap
cookie, `INSIGHT_TRUST_PROXY_AUTH` identity, `Secure` cookies) + per-session
artifact ownership in `webapp/security.py` / `webapp/app.py`; per-identity
sliding-window rate limits (429 + `Retry-After`) in `webapp/ratelimit.py`;
request IDs + JSON access log + Prometheus `/metrics` in `webapp/telemetry.py`;
`/readyz` readiness probe in `webapp/readiness.py`; session LRU + idle TTL in
`webapp/sessions.py`; artifact retention (`insight/retention.py`,
`tools/retention.py`, startup sweep); MCP hardening (`INSIGHT_MCP_AUTH_TOKEN`,
Bearer middleware, `INSIGHT_MCP_FILE_ROOTS` allowlist + size cap, DNS-rebinding
host allowlist); CI dependency audit (`pip-audit`) + image build/Trivy jobs;
`tools/load_test.py`; `__version__` synced to 2.3.0. Tests:
`test_webapp_security.py`, `test_webapp_telemetry.py`, `test_sessions.py`,
`test_retention.py` plus additions to `test_toolkit.py`, `test_mcp_server.py`,
`test_chain_fallback.py`. **Not shipped:** external session store / multi-worker
(C9) - in-memory sessions remain the documented single-worker constraint.

### P11 - Memory / connection / race hardening
Follow-up audit pass on the P10 surface. Shipped: session generation counters, state
locks and per-session busy flags (`webapp/sessions.py`); atomic dataset replacement and
stale-safe background polish; 409/redirect guards for concurrent query, upload and clear;
snapshot-based exports; bounded conversation history (`INSIGHT_SESSION_MAX_TURNS=25`);
160 MB aggregate / 6-file upload caps and chunked reads; pre-parse row caps
(`INSIGHT_MAX_UPLOAD_ROWS=1000000`); atomic cache writes (`cache.py`); bounded toolkit
run history and table registry (`INSIGHT_MCP_MAX_RUNS` / `INSIGHT_MCP_MAX_TABLES`);
per-event-loop shared HTTP clients (`insight/llm/http_pool.py`); cached LangSmith client;
matplotlib `rc_context` isolation; rate-limit key pruning; temp chart-dir sweep for
library calls. Regression tests in `tests/test_hardening.py` and
`tests/test_profiler.py`.

### P12 - Standalone verifier and certificate (2026-11-07)
Shipped the unbundled "epistemic firewall": `verify_claims` now returns a
versioned `certificate` alongside its legacy summary. The certificate separates
`numeric_grounding`, `independent_recomputation`, `statistical_validity`,
`causal_language`, and `structural` checks, and states an explicit
`attestation_level` (`grounding` or `recomputed`) so callers can see whether the
verifier had an execution trace. Only grounding/recomputation can hard-fail;
statistical and causal findings are warnings, and skipped checks are
`not_checked`.

Grounding upgrades: scientific-notation tokens (`4.794e-16` stays one token),
precision-aware rounding (`8.55` grounds `8.554537`; `8.56` does not),
magnitude matching (`683` against `-683.411797`), p-value inequalities
(`p < 0.001`), and relative exactness for tiny values so `1e-9` cannot ground a
rounded-away `p_value = 0.0`.

Recomputation: `statistical_test` is now covered - `TestOutcome` carries
`group_means` and `difference`, the operation emits `mean_*`/`difference`
calculations, and the verifier re-routes the test and checks test name,
statistic, p-value, effect size, means, and difference. SQL and chained
(`input_step`) steps are reported `not_covered`.

Surfaces: `python -m insight.verify_cli` (evidence / data + operation / data +
SQL, `--params-file`, `--out`, `--fail-on-warn`) and
`uvicorn insight.verify_service:app` (`POST /v1/verify`, optional
`INSIGHT_VERIFY_AUTH_TOKEN`, `INSIGHT_VERIFY_MAX_CSV_MB`). Both are local/dev
surfaces. Contract: `docs/CERTIFICATE.md`; schema:
`schemas/verification_certificate.schema.json`; demo:
`tools/demo_epistemic_firewall.py`.

Tests: `tests/test_certificate.py` and `tests/test_verify_service.py`.
Real-data trials covered superstore group aggregates, fabricated values, causal
warnings, SQL `not_checked`, penguins `statistical_test`, live HTTP (including
bearer auth), and the full MCP `load_table → execute_operation → verify_claims`
flow. Remaining limits: grounding is value-based, not semantic-role-based;
`signature` is still `null`; no persistence/queues/tenancy.

### Explicitly deferred (per owner decision, do not start unprompted)
Container/subprocess sandboxing (no codegen exists), multi-tenant RBAC/SSO, PDF export, dashboards, external session store (C9 - owner-deferred; requires a consistency model for DataFrames/responses).

---

## 7. Conventions for next session

- **No comments in code** unless asked; short docstrings only on public classes where valuable.
- **Pydantic v2** everywhere (`model_validate`, `model_dump`); Optional fields tolerate LLM `null`s (see ChartSpec.resolved_*).
- **New analytics op checklist**: params BaseModel w/ Field descriptions (auto-catalog) → register_operation decorator → `_COLUMN_REF_FIELDS` entry → validator `TABLE_PRODUCING_OPS` if it emits tables → tests in `test_operations.py` → done (prompt updates free).
- **Test LLM calls** always through `stub_chain_factory` (conftest) monkeypatching `orch._build_chain_for`; never real network in CI.
- **Windows shell**: prefer writing temp .py scripts over multiline `python -c`; use `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` to avoid BOM.
- Keep `../README.md`, `HOW_TO_RUN.md`, and this file honest when scope changes.

---

## 8. Quick sanity checklist after any change

```bash
pytest -q                     # expect exit 0; skips vary by platform and optional corpora
python tools/smoke_toolkit.py # manual: toolkit + MCP on real public data (needs evaluation/data/)
python tools/demo_epistemic_firewall.py  # CLI certificate: correct pass, fabricated fail, causal warn
pytest -q tests/test_certificate.py tests/test_verify_service.py
uvicorn webapp.app:app --port 8000   # boot, upload any CSV, run one query, check SSE stages + View plan
curl -s localhost:8000/readyz        # readiness JSON (503 until a provider is configured)
python tools/retention.py --days 14 --dry-run
python tools/load_test.py --count 20 --workers 8   # upload-path load smoke (no LLM calls)
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b   # golden-question benchmark
```

The plan-review UI (`plan_review` → edit JSON → Approve & run) is unit-tested
but has **not** been exercised through a real browser round trip. Verify
manually before relying on it.
