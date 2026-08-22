# Insight v3 Future Scope

Status as of 2026-08-22. Shipped since v2.0: evaluation harness, statistics depth
(ANOVA/regression/CIs/isotonic calibration), conversation state machine, packaging
(Docker/CI/query cache), DOCX consolidation, embedded DuckDB relational analytics
(multi-file joins/windows via sandboxed `sql_query`) and dbt/alert artifact exports.
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

### 4. Streaming-path caching
The query-result cache currently serves `/api/query`; extend it to warm the SSE
path so repeated UI questions skip planning without breaking live streaming UX.

### 5. Alert-runner reference implementation
The exported monitor JSON needs a tiny companion worker (cron → DuckDB → webhook)
to close the loop; ship it as `tools/alert_runner.py`.

### Explicitly deferred (per owner decision)
Python code-interpreter sandboxing (DuckDB chosen instead), auth/multi-user,
PDF export, dashboards.
