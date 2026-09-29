# Insight v3 Future Scope

Status as of 2026-09-29. Shipped since v2.0: evaluation harness, statistics depth
(ANOVA/regression/CIs/isotonic calibration), conversation state machine, packaging
(Docker/CI/query cache), DOCX consolidation, embedded DuckDB relational analytics
(multi-file joins/windows via sandboxed `sql_query`) and dbt/alert artifact exports.
Recently: token-by-token answer streaming, prefix-cached planner prompts,
cache on the SSE path, background briefing polish, and human-in-the-loop plan
approval (`INSIGHT_PLAN_APPROVAL=on`).
A Python code-interpreter sandbox was explicitly skipped by the owner — DuckDB
covers the analytical surface safely.

## Completed in v2.x (for reference)

1. **Automatic dataset profiling** — shipped as the profiler + zero-prompt briefing.
2. **Tool-based statistical analysis** — shipped as the deterministic op registry
   (15 ops incl. DuckDB SQL): the LLM plans, pandas/scipy/DuckDB compute.
3. **Persistent session memory** — partially shipped: conversation state machine
   persists analytical scope across turns; webapp sessions remain in-memory.
4. **Multi-file reasoning** — shipped for CSV sets (DuckDB joins); native SQLite/
   PostgreSQL connectors still open below.
5. **Human-in-the-loop** — shipped: every plan is reviewable and editable before
   execution; approved plans are re-validated against the schema.

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

### 4. ~~Streaming-path caching~~ — shipped
The query-result cache now serves the SSE path too (~0.23-0.39s repeats vs
7.8s cold). Remaining work here is only cache-hit UI polish, not the plumbing.

### 5. Alert-runner reference implementation
The exported monitor JSON needs a tiny companion worker (cron → DuckDB → webhook)
to close the loop; ship it as `tools/alert_runner.py`.

### Explicitly deferred (per owner decision)
Python code-interpreter sandboxing (DuckDB chosen instead), auth/multi-user,
PDF export, dashboards.

## Measured non-decisions (do not revisit without new data)

### Native execution engine (C++ / Rust / Go) — rejected on measurement
Profiling a ~30s question: the deterministic engine is **14-62ms** (≈0.2%),
while planning + explaining (LLM) is 95%+. Execution already handles 250K rows
with joins, OLS and charts. A Rust/C++ core would save ~55ms per query — under
0.2% — in exchange for an FFI/build-toolchain tax. The win is in the LLM calls:
prefix caching, fewer planning round-trips, and streaming (all shipped).
A native layer becomes worth revisiting only if execution ever dominates, e.g.
10M+ rows with heavy per-group work.

### Ground-truth-dependent validator tuning — blocked, not declined
The validator rejects roughly half of first-pass plans, costing a full extra LLM
round-trip (up to ~190s observed). `meta.repair_issues` now records the exact
rejection reasons, but tuning validation against them needs live provider runs
and the Groq developer tier hit its **daily** 200K token cap during the last
investigation. Re-run the repair-issue probe on a fresh quota before deciding
what to loosen. Note that observed "plan" times near 90-190s were throttle
backoff, not inference.
