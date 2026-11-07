# Insight v3 Future Scope

Status as of 2026-11-07. Shipped since v2.0: evaluation harness, statistics depth
(ANOVA/regression/CIs/isotonic calibration), conversation state machine, packaging
(Docker/CI/query cache), DOCX consolidation, embedded DuckDB relational analytics
(multi-file joins/windows via sandboxed `sql_query`) and dbt/alert artifact exports.
Recently: token-by-token answer streaming, prefix-cached planner prompts,
cache on the SSE path, background briefing polish, and human-in-the-loop plan
approval (`INSIGHT_PLAN_APPROVAL=on`). Latest: **agentic tool-use mode**
(`INSIGHT_AGENTIC` - ops as tools, bounded observe/plan/act/refine loop over the
same verified pipeline), the **alert-runner companion worker**
(`tools/alert_runner.py`), and the **versioned verification certificate** with a
standalone CLI (`insight.verify_cli`) and local HTTP surface
(`insight.verify_service`).
A Python code-interpreter sandbox was explicitly skipped by the owner - DuckDB
covers the analytical surface safely.

## Completed in v2.x (for reference)

1. **Automatic dataset profiling** - shipped as the profiler + zero-prompt briefing.
2. **Tool-based statistical analysis** - shipped as the deterministic op registry
   (15 ops incl. DuckDB SQL): the LLM plans, pandas/scipy/DuckDB compute.
3. **Persistent session memory** - partially shipped: conversation state machine
   persists analytical scope across turns; webapp sessions remain in-memory.
4. **Multi-file reasoning** - shipped for CSV sets (DuckDB joins); native SQLite/
   PostgreSQL connectors still open below.
5. **Human-in-the-loop** - shipped: every plan is reviewable and editable before
   execution; approved plans are re-validated against the schema.
6. **Standalone verification** - shipped: `verify_claims` returns a versioned
   certificate (grounding vs recomputed attestation, claim spans, evidence ids,
   per-aggregate status) through the toolkit/MCP server, CLI and local HTTP
   endpoint. Covered operations are recomputed; SQL and chained steps are
   explicitly `not_covered`.

## Next up

### 1. External session storage
Move `WebSession` (CSV bytes, profile, turns) to Redis/Valkey to unlock
multi-worker uvicorn and container restarts. The single-worker limitation is
currently documented in README/Dockerfile.

### 2. Live database connections
Beyond uploaded files: ATTACH-style read-only connections to user-provided
SQLite/Postgres DSNs, with credential handling and per-table grants.

### 3. Missingness mechanism diagnostics
Today listwise deletion assumes MCAR and the validator says so explicitly.
Next: detect MAR/MNAR signals per column and recommend imputation or exclusion
policies before aggregates are computed.

### 4. ~~Streaming-path caching~~ - shipped
The query-result cache now serves the SSE path too (~0.23-0.39s repeats vs
7.8s cold). Remaining work here is only cache-hit UI polish, not the plumbing.

### 5. ~~Alert-runner reference implementation~~ - shipped
`tools/alert_runner.py` runs exported monitor JSONs on a schedule through the
same sandboxed DuckSession and posts Slack-compatible payloads on threshold
state-change (see README "Alert runner").

### 6. Agentic mode - shipped; A/B measured on one endpoint
`INSIGHT_AGENTIC` routes questions through the tool-calling loop
(`insight/tools/`) with the same verification tail. Pinned k=3 A/B on NVIDIA
NIM `gpt-oss-20b` (2026-10-16, `--no-fallback`, same 20 cases/grader, production
budget 5/9) measured agentic **58/60 (96.7%, 95% CI 88.6–99.1)** vs single-shot
**43/60 (71.7%, 59.2–81.5)** at mean 2.17 vs 2.36 `llm_calls` - but the
2026-10-17 diagnosis showed most of that gap was a provider-side truncation bug
that only hit the single-shot planner (13 of 17 baseline failures), fixed the
same day; post-fix k=1 confirmations score 20/20 vs 19/20, i.e. **accuracy
parity on this endpoint - the agentic value is the auditable tool trail**.
**Caveat (2026-11-03):** the historical agentic arm scored the loop output
directly and did not run the validator/grounding tail; those numbers are
loop-only, not an end-to-end A/B. The shipped agentic path does run that tail.
The first live runs found and fixed an OpenAI wire-format bug that no offline
fake could reach; env-var provider pinning proved unreliable on Windows, so a
`--no-fallback` flag makes live runs provable. Offline assurance: 11/11 sampled
mutations killed, `loop.py` 96% branch coverage (see TEST.md "Verification
evidence"). Next: repeat the pinned A/B on a second endpoint (Groq) - more
important now that the first endpoint's gap collapsed, and re-measure k≥3
post-fix - and test whether a System One op-classifier router can cut calls
further (the `OperationRouter` seam is in place). The
`sup_chat_vs_email_resolution` failure was diagnosed: a gpt-oss harmony-token
leak into tool names (fixed - inbound names are sanitized; post-fix pinned run
20/20); the residual pattern is the model short-circuiting past the required
test op on that one case.

### 7. Semantic claim verification and signed certificates
The current certificate proves that a number appears in evidence and that covered
aggregates survive a second arithmetic path. It does not prove the number is
attached to the correct predicate, that caller-supplied evidence is true, or
that causal language is justified. `signature` is present but always `null`.
Next: evidence-replay control, a signing key managed by the verifier, and
claim-level checks for a narrow set of numeric predicates. SQL and chained
(`input_step`) steps also remain outside independent recomputation.

### Explicitly deferred (per owner decision)
Python code-interpreter sandboxing (DuckDB chosen instead), auth/multi-user,
PDF export, dashboards.

## Measured non-decisions (do not revisit without new data)

### Native execution engine (C++ / Rust / Go) - rejected on measurement
Profiling a ~30s question: the deterministic engine is **14-62ms** (≈0.2%),
while planning + explaining (LLM) is 95%+. Execution already handles 250K rows
with joins, OLS and charts. A Rust/C++ core would save ~55ms per query - under
0.2% - in exchange for an FFI/build-toolchain tax. The win is in the LLM calls:
prefix caching, fewer planning round-trips, and streaming (all shipped).
A native layer becomes worth revisiting only if execution ever dominates, e.g.
10M+ rows with heavy per-group work.

### Ground-truth-dependent validator tuning - blocked, not declined
The validator rejects roughly half of first-pass plans, costing a full extra LLM
round-trip (up to ~190s observed). `meta.repair_issues` now records the exact
rejection reasons, but tuning validation against them needs live provider runs
and the Groq developer tier hit its **daily** 200K token cap during the last
investigation. Re-run the repair-issue probe on a fresh quota before deciding
what to loosen. Note that observed "plan" times near 90-190s were throttle
backoff, not inference.
