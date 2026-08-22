# INSIGHT — Session Handover

> Last updated: 2026-08-22 · Engine v2.1.0-alpha · **122 offline tests passing** · **20/20 live eval cases** · Python 3.11 venv at `.venv`

---

## 1. What this project is

**Insight** is a validated, model-agnostic analytical agent for CSVs.
Core principle: **the LLM never executes code and never invents numbers.**

```
Upload → Profiler + Zero-Prompt Briefing (top-3 findings, no question needed)
      → Planner (LLM) emits a JSON AnalysisPlan → schema/column validated → bounded repair
      → Deterministic engine executes 11 pandas ops (chained via input_step DAG)
      → Statistical router runs scipy tests by column types (t-test / Mann-Whitney / chi²+Cramér's V)
      → Validator checks sanity + INDEPENDENTLY RECOMPUTES headline numbers (2nd code path)
      → Explainer (LLM) narrates using only computed values
      → SSE-streamed to UI · DOCX report incl. briefing findings · optional LangSmith trace
```

Two front-ends share one engine (`insight/`): **FastAPI webapp** (primary, `webapp/`) and **Streamlit** (`main.py`, maintained).

---

## 2. Repo map (what lives where)

| Path | Role |
|---|---|
| `insight/domain/` | Pydantic contracts: `DatasetProfile`+fingerprint, `AnalysisPlan`(steps+charts+input_step), `ExecutionResult`, `ChartSpec`, `Evidence`, typed errors |
| `insight/llm/` | `base.py` ABC; `providers/openai_compat.py` (retry/backoff/429-aware) + groq/mistral/openai/ollama/openrouter/anthropic; `registry.py`: curated catalog + live discovery + capability flags + fallback chain + custom env provider |
| `insight/profiling/profiler.py` | role inference (identifier-by-name heuristic!), missingness, fingerprint |
| `insight/planning/` | `prompts.py` (hard rules incl. title/DAG/binary rules), `planner.py` (JSON repair loop) |
| `insight/analytics/operations.py` | **the op registry** — 11 deterministic ops + `operation_catalog()` (auto-generates planner prompt) + `validate_plan_columns()` |
| `insight/analytics/statistics.py` | scipy routing: group comparison / chi-square / correlation, effect sizes, plain-language interpretation |
| `insight/briefing.py` | zero-prompt detectors: trend/corr/outliers/imbalance/missing/dupes → ranked findings |
| `insight/execution/executor.py` | sequential executor; `frames` dict enables `input_step` DAG; filter_rows mutates implicit chain |
| `insight/validation/` | `result_validator.py` (sanity/small-sample/chart-consistency) + `verification.py` (**independent recomputation**) |
| `insight/visualization/` | `themes.py` (editorial/minimal/dark) + `renderer.py` (ChartSpec→PNG; adaptive bins live in distribution op; Other-bucket, horizontal-bar, tick formatting here) |
| `insight/orchestrator.py` | `analyze_stream()` async generator (stage events) wrapped by `analyze()`; `run_analysis_sync()` for sync UIs |
| `insight/observability.py` | LangSmith tracing (optional; `INSIGHT_TRACING=off` kills it) |
| `insight/settings.py` | all config via env; `.env` auto-loaded |
| `webapp/` | FastAPI: `app.py` routes, `sessions.py` cookie store, `services.py` streamlit-free IO, Jinja `templates/`, `static/` (CSS + app.js SSE client) |
| `ui_components.py` / `utils.py` / `main.py` | Streamlit legacy UI (same engine via cached wrappers) |
| `tests/` | **85 tests**, offline (fake providers / scripted chains). Key fixtures in `conftest.py` |

---

## 3. Run & test

```bash
# activate .venv first (Windows: .venv\Scripts\activate)

# primary UI
uvicorn webapp.app:app --port 8000        # → http://localhost:8000

# legacy UI
streamlit run main.py                      # → :8501

# tests (no network / no keys needed — tracing force-off in conftest)
pytest -q                                  # 85 passed expected (~10-30s on Windows)

# headless API
curl -X POST localhost:8000/api/query -H "Content-Type: application/json" \
     -d '{"question":"Which region has the highest revenue?"}'
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

---

## 5. Known quirks / gotchas

1. **Windows console cp1252** chokes printing unicode (`…`, `\u202f`) — scripts need `sys.stdout.reconfigure(encoding="utf-8")`; the *library* is fine.
2. **PowerShell heredocs add BOM** with `-Encoding UTF8` on PS5.1 — after Set-Content edits, strip BOM or read as utf-8-sig when parsing.
3. **sse framing**: webapp hand-formats SSE (`event:\ndata:\n\n`, LF). Client normalizes CRLF and skips `:` comments anyway.
4. **uvicorn cold start ~8-12 s** (scipy+langsmith imports) — not hung.
5. **`st.cache_data` wrappers** in `utils.py` are Streamlit-only; core logic duplicated cleanly in `webapp/services.py` (keep both in sync when changing CSV parsing).
6. **Tracing in tests**: conftest sets `INSIGHT_TRACING=off` BEFORE insight imports — preserve that line's position.
7. **Discovery cap**: `discovery_max_per_provider=40`; OpenRouter ranking = preferred vendors then shortest-name-first; variants (`:batch`) deduped.
8. **Planner quirk**: small models may set `top_n=1` (charts suffer but numbers stay correct); ChartSpec treats explicit `null` as "use default" (`resolved_top_n/resolved_sort_desc`).
9. **Artifacts**: plots under `.artifacts/web/<data_id>/` (webapp) vs `.artifacts/<session_id>/` (streamlit). Both gitignored.

---

## 6. Remaining work — build queue (priority order)

> **Status 2026-08-22:** P1–P5 below are DONE. See `future_scope.md` for what's next
> (multi-file reasoning, Redis sessions, MAR/MNAR diagnostics, SSE-path caching).

### P1 — Evaluation harness ✅
`evaluation/generate_datasets.py` (seeded: sales/ecommerce/support + retail_250k;
diamonds.csv fetched separately) · `evaluation/cases.json` (20 cases w/ ground truth)
· `evaluation/run_eval.py` (ops superset + substrings + numeric ±tol scoring,
markdown report). pytest `eval` marker gated by `INSIGHT_RUN_EVAL=1`.
Live results: 20/20 (gpt-oss-120b 3/3 smoke; mistral-small 12/15 pre-fixes → fixed).

### P2 — Statistics depth ✅
`anova_test` (F + η²), `regression` (OLS lstsq, t p-values, R², std betas,
listwise-drop reporting), compare_subsets 95% CIs (n≥30 gate). Router: cat>2×num → ANOVA;
binary-num × categorical → chi-square (fixed swapped-args bug found by eval).
`isotonic_calibration` (PAV, reliability table, Brier/ECE) added on top.

### P3 — Conversation state machine ✅
`insight/conversation/state.py`; orchestrator `state=` param end-to-end; planner gets
serialized context; webapp + streamlit persist `analysis_state`; cache key includes it.

### P4 — Packaging ✅
`Dockerfile` (non-root, single-worker note) · `.dockerignore` · `ruff.toml` +
`.github/workflows/ci.yml` (ruff+pytest, tracing off) · query cache in `insight/cache.py`
(TTL 14d / 512MB budget eviction; `INSIGHT_QUERY_CACHE*` envs; valid-results only).

### P5 — Smaller polish ✅
Webapp `POST /models/refresh` + rail button · DOCX consolidated to `insight/reports/docx.py`
(both UIs delegate) · briefing LLM polish behind `INSIGHT_BRIEFING_LLM_POLISH` (default off)
· validator now emits MCAR caveats for referenced columns >5% missing · group_aggregate
reports `excluded_missing_<metric>` for mixed count bases.

### P6 — DuckDB relational analytics + artifact exports ✅
`insight/analytics/sql_engine.py` (`DuckSession`: in-memory con, frames registered then
`enable_external_access=false` + `lock_configuration=true`, SELECT/WITH-only keyword
guards, chained-statement block, interrupt watchdog `INSIGHT_SQL_TIMEOUT_S=10`,
row cap). Op 15/15: `sql_query` — joins/windows/pivots; errors feed the existing
plan-repair loop. Multi-file upload (`files[]`, up to 6) → per-session tables;
planner gets exact table schemas via `DuckSession.describe()`. Exports:
`POST /export/dbt` (zip: model.sql + schema.yml w/ not_null tests) and
`GET /export/alert?step_id=` (Slack-ready monitor JSON), buttons render under SQL
answers. Tests: `test_sql_engine.py` (16), `test_sql_webapp.py` (2),
`test_artifacts.py` (2). Sandbox code-interpreter explicitly skipped by owner.

### Explicitly deferred (per owner decision, do not start unprompted)
Container/subprocess sandboxing (no codegen exists), FastAPI auth/multi-user, PDF export, dashboards.

---

## 7. Conventions for next session

- **No comments in code** unless asked; short docstrings only on public classes where valuable.
- **Pydantic v2** everywhere (`model_validate`, `model_dump`); Optional fields tolerate LLM `null`s (see ChartSpec.resolved_*).
- **New analytics op checklist**: params BaseModel w/ Field descriptions (auto-catalog) → register_operation decorator → `_COLUMN_REF_FIELDS` entry → validator `TABLE_PRODUCING_OPS` if it emits tables → tests in `test_operations.py` → done (prompt updates free).
- **Test LLM calls** always through `stub_chain_factory` (conftest) monkeypatching `orch._build_chain_for`; never real network in CI.
- **Windows shell**: prefer writing temp .py scripts over multiline `python -c`; use `[System.IO.File]::WriteAllText(..., UTF8Encoding($false))` to avoid BOM.
- Keep `README.md`, `HOW_TO_RUN.md`, and this file honest when scope changes.

---

## 8. Quick sanity checklist after any change

```bash
pytest -q                     # expect 122+
uvicorn webapp.app:app --port 8000   # boot, upload any CSV, run one query, check SSE stages + View plan
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b   # golden-question benchmark
```
