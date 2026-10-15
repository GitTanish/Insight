# INSIGHT — Comprehensive Testing & Quality Assurance Manual (TEST.md)

> **Document Purpose**: This document is the persistent testing brain and architectural handover for human engineers and autonomous coding agents (GPT, Gemini, Claude, Antigravity). A fresh agent reading this file without prior context can test, audit, validate, attack, and debug Insight systematically.
> **Source of Truth**: Implementation code in `insight/`, `webapp/`, `evaluation/`, and `tests/` overrides all marketing descriptions or historical notes.

---

## 1. System Architecture & Mental Model

### Core Philosophy
Insight operates on an absolute architectural boundary: **The LLM plans and narrates, but deterministic code executes, calculates, validates, and gates all numbers.** Under no circumstance does the LLM execute arbitrary code or invent numerical statistics.

```
                                  [ USER / CLIENT ]
                                    │             ▲
              1. Upload CSV / XLSX  │             │ 9. SSE Streaming / Web UI / DOCX
                                    ▼             │
                        ┌───────────────────────────────┐
                        │    FastAPI WebApp / Streamlit │
                        └───────────────┬───────────────┘
                                        │
                         2. Profiler & Zero-Prompt Briefing
                                        ▼
                        ┌───────────────────────────────┐
                        │      insight.profiler         │ ───► Heuristic Findings
                        │      insight.briefing         │      (Trend, Outliers, Skew)
                        └───────────────┬───────────────┘
                                        │
                         3. Question + Profile + Schema
                                        ▼
                        ┌───────────────────────────────┐
                        │      LLM Planner              │ ───► JSON AnalysisPlan
│ (insight.planning.planner)    │      (Bounded Self-Repair Loop)
                         └───────────────┬───────────────┘
                                         │
                        3b. AGENTIC MODE (opt-in — same guarantees)
                                         ▼
                         ┌───────────────────────────────┐
                         │ insight.tools.loop            │ ───► tool_call/tool_result
                         │ (bounded observe/plan/act)    │      streamed over SSE
                         │ • ops exposed as tools        │
                         │ • budgets/throttle/abort      │
                         │ • input_step DAG chaining     │
                         └───────────────┬───────────────┘
                                         │ rebuilt AnalysisPlan +
                                         │ ExecutionResult
                          4. Column & Schema Validation
                                         ▼
                        ┌───────────────────────────────┐
                        │    Deterministic Executor     │
                        │ (insight.execution.executor)  │
                        ├───────────────────────────────┤
                        │ • 15 Fixed Pandas/SciPy Ops   │ ───► Output DataTables &
                        │ • Sandboxed In-Memory DuckDB  │      Calculations
                        └───────────────┬───────────────┘
                                        │
                         5. Dual-Stage Validation
                                        ▼
                        ┌───────────────────────────────┐
                        │ • result_validator.py         │ ───► Sanity, MCAR warnings,
                        │ • verification.py             │      Independent Recomputation
                        └───────────────┬───────────────┘      (2nd Code Path)
                                        │
                         6. Render Editorial Charts
                                        ▼
                        ┌───────────────────────────────┐
                        │ insight.visualization.renderer│ ───► Matplotlib PNG Figures
                        └───────────────┬───────────────┘
                                        │
                         7. Evidence-Grounded Narration
                                        ▼
                        ┌───────────────────────────────┐
                        │      LLM Explainer            │ ───► Draft Answer Text
                        └───────────────┬───────────────┘
                                        │
                         8. Grounding Guard Verification
                                        ▼
                        ┌───────────────────────────────┐
                        │ insight.validation.grounding  │ ───► Strips / Retries any
                        └───────────────────────────────┘      ungrounded numeric claims
```

### Component Responsibility Map

| Subsystem | Key Files | Responsibility | Failure Mode |
|---|---|---|---|
| **Domain Contracts** | `insight/domain/dataset.py`<br>`insight/domain/analysis.py`<br>`insight/domain/query.py`<br>`insight/domain/visualization.py`<br>`insight/domain/errors.py` | Pydantic v2 schemas: `DatasetProfile`, `AnalysisPlan`, `PlanStep`, `ExecutionResult`, `DataTable`, `Calculation`, `ChartSpec`, `Evidence` | Schema validation error, typed `InsightError` hierarchy |
| **Profiling & Briefing** | `insight/profiling/profiler.py`<br>`insight/briefing.py` | Role inference (numeric, categorical, datetime, identifier), fingerprinting (schema/content SHA-256), heuristic anomaly briefing | Graceful skip of non-viable metrics, warning emission |
| **Planning Engine** | `insight/planning/planner.py`<br>`insight/planning/prompts.py` | Generates structured JSON plans with bounded self-repair (`max_repair_attempts = 1`). Injects SQL schema for multi-table queries. Separate `AGENTIC_SYSTEM` prompt for tool-calling mode | `PlanValidationError` (surfaced cleanly to user) |
| **Agentic Tool-Use Layer** | `insight/tools/tooling.py`<br>`insight/tools/runtime.py`<br>`insight/tools/loop.py`<br>`insight/tools/pipeline.py` | Op registry → provider tool schemas (`$defs` inlined; OpenAI + Anthropic shapes); `dispatch()` validates then runs the same `OperationSpec.run`; `run_loop()` is the bounded observe/plan/act/refine loop; `pipeline.py` rebuilds `AnalysisPlan`+`ExecutionResult` so the SAME validator/charts/explainer/grounding tail runs. Opt-in via `INSIGHT_AGENTIC` / `AnalysisRequest.agentic` | Bounded by `LoopBudget`; aborts on repeated identical failure; returns readable errors to the model for retry; never executes free-form code |
| **Op Registry** | `insight/analytics/operations.py` | 15 deterministic operations: `summarize`, `value_counts`, `top_n`, `group_aggregate`, `distribution`, `detect_outliers`, `trend`, `correlation`, `statistical_test`, `anova_test`, `regression`, `isotonic_calibration`, `sql_query`, `filter_rows`, `compare_subsets` | `OperationError` caught by executor, marked as failed step |
| **DuckDB SQL Sandbox** | `insight/analytics/sql_engine.py` | Embedded in-memory DuckDB with `enable_external_access=false`, `lock_configuration=true`, single SELECT/WITH guard, keyword blocklist, and 10s interrupt watchdog | `ValueError` with security or timeout explanation |
| **Statistical Engine** | `insight/analytics/statistics.py` | SciPy test routing (Welch t-test, Mann-Whitney U, Chi-Square + Cramér's V, ANOVA + $\eta^2$, OLS regression, Isotonic PAV) | Rejection of small samples, fallback to non-parametric tests |
| **Validation Engine** | `insight/validation/result_validator.py`<br>`insight/validation/verification.py`<br>`insight/validation/grounding.py` | Three-layer gate: (1) Sanity & NaN scan, (2) Independent pandas recomputation (second code path), (3) Answer number-level grounding verification | Triggers plan repair or explainer rewrite; flags unverified numbers |
| **LLM Provider Hub** | `insight/llm/registry.py`<br>`insight/llm/providers/*`<br>`insight/llm/base.py` | Multi-provider fallback chain (Groq, Mistral, OpenAI, Anthropic, Ollama, OpenRouter, Custom). Live model discovery and BYOK (`X-Insight-Key`) | `AllProvidersFailedError`, automatic fallback down provider list |
| **Conversation State** | `insight/conversation/state.py` | Cross-turn active filter persistence, tracked dimensions and metrics, session context serialization | State isolation per session cookie / session ID |
| **Query Cache** | `insight/cache.py` | SHA-256 fingerprint + question + model + state cache with TTL days (default 14) and max disk budget (default 512MB LRU) | Bypassed when `INSIGHT_QUERY_CACHE=off` or plan validation fails |
| **Export Engines** | `insight/reports/docx.py`<br>`insight/reports/artifacts.py` | Multi-turn `.docx` report generation, dbt model + schema.yml `.zip` export, Slack-ready alert JSON monitor | Empty table or file write error handled gracefully |
| **Companion Worker** | `tools/alert_runner.py` | Executes exported monitors on an interval: registers CSVs, runs each monitor's query through the SAME sandboxed `DuckSession`, applies the threshold, posts Slack-compatible payloads on state-change. Stdlib-only, `--dry-run` supported | Query rejection surfaces as a warning line; webhook failure does not crash the loop |
| **Web Applications** | `webapp/app.py`<br>`webapp/services.py`<br>`webapp/sessions.py`<br>`main.py` (Streamlit) | Primary FastAPI app with SSE streaming (`/query`), REST (`/api/query`), uploads, and healthcheck (`/healthz`). Secondary Streamlit UI | In-memory session expiry, HTTP 400/401/500 structured JSON |

---

## 2. Behavioral Invariants & Trust Boundaries

### Critical Invariants

| ID | Invariant | Where Enforced | How to Attack / Test | Authoritative Oracle |
|:--:|---|---|---|---|
| **INV-01** | **No Arbitrary Code Execution** | System Architecture | Prompt planner: *"Write python code to compute total"* | Plan must strictly contain registered operations (`operations.py`) or sandboxed SQL. |
| **INV-02** | **Deterministic Numerical Authority** | `insight/execution/executor.py`<br>`insight/analytics/operations.py` | Compare numbers in `ExecutionResult.calculations` against reference pandas/scipy math | Tolerance $\le 10^{-6}$ against standalone NumPy / SciPy computation. |
| **INV-03** | **Bounded Plan Repair** | `insight/planning/planner.py:52` | Force invalid plan (e.g. non-existent column) | Exactly `1 + max_repair_attempts` attempts made; halts on failure without looping. |
| **INV-04** | **Independent Recomputation Gate** | `insight/validation/verification.py`<br>`insight/validation/result_validator.py` | Tamper with operation output table rows | `validate_execution().valid == False`; mismatch issue emitted if values deviate $> 10^{-6}$. |
| **INV-05** | **Strict Number Grounding** | `insight/validation/grounding.py` | Explainer hallucinates a statistic (e.g., *"West had 99,999 units"* when not in table) | `find_ungrounded_numbers` catches the token; triggers retry; marks ungrounded in metadata. |
| **INV-06** | **DuckDB SQL Confinement** | `insight/analytics/sql_engine.py:62-105` | Inject `ATTACH`, `COPY`, `DROP`, `INSTALL`, `; SELECT ...`, or file reads | Rejected by `validate_statement` keyword blocklist, single-statement split, or engine sandboxing. |
| **INV-07** | **SQL Interrupt Watchdog** | `insight/analytics/sql_engine.py:112-127` | Execute long-running SQL query: `SELECT count(*) FROM range(10000000000)` | Interrupted at `timeout_s` (default 10s); raises `ValueError("query exceeded limit")`. |
| **INV-08** | **Plan DAG Acyclicity** | `insight/analytics/operations.py:validate_plan_columns` | Supply `input_step >= step_id` or circular chain | `validate_plan_columns` rejects the plan before execution. |
| **INV-09** | **Dirty Data Survival** | `insight/analytics/operations.py` | Pass DataFrame with 30%+ nulls, empty strings, NaNs | All 15 operations complete without crashing; emit MCAR warnings if missingness $> 5\%$. |
| **INV-10** | **Session Isolation** | `webapp/sessions.py` | Query data using a different session cookie | Session A data, tables, and turns must never leak to Session B. |
| **INV-11** | **Agentic Loop Is Bounded** | `insight/tools/loop.py` (`LoopBudget`, `run_loop`) | Router that *always* returns another tool call | Loop stops at `max_iterations` / `max_llm_calls`; `outcome.error` ∈ {`iteration_cap_reached`, `llm_call_budget_exhausted`, `repeated_failing_call`}; provider calls never exceed the budget. |
| **INV-12** | **Agentic Tool Calls Are Not Code Execution** | `insight/tools/runtime.py` (`dispatch`) | Ask the agent for `drop_table`, or a `group_aggregate` on a hallucinated column, or raw Python | Unknown op → readable error listing available ops; hallucinated column → rejected by `validate_plan_columns` before any op runs. Only registered `OperationSpec.run` bodies execute. |
| **INV-13** | **Agentic Answers Face The Same Gates** | `insight/orchestrator.py` (`_tail_events`), `insight/tools/pipeline.py` | Have the agentic loop produce a result, then tamper a headline number | `validate_execution()` runs (incl. `independently_verify`), `find_ungrounded_numbers` flags ungrounded figures — identical code path as a planned answer, because the loop outcome is rebuilt into plan + execution. |
| **INV-14** | **HITL Precedence Over Agentic** | `insight/orchestrator.py` (`analyze_stream`) | Request `agentic=True` together with `plan_only=True` (or `INSIGHT_PLAN_APPROVAL=on`) | Agentic is forced off; the request stops at a reviewable `plan_review` event and **no** `tool_call`/`tool_result` events are emitted. |

### Trust Boundaries

```
[ User Input (Question / Files) ]
      │ (Untrusted text, potential prompt injection, malformed CSV)
      ▼
┌───────────────────────┐
│ Boundary 1: Ingestion │ ──► CSV/XLSX sniffers, row cap (1M rows), 80MB upload limit
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│ Boundary 2: Planner   │ ──► LLM prompt sanitization, column canonicalization, strict Pydantic parsing
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│ Boundary 3: Engine    │ ──► Operations accept only validated typed models; DuckDB blocked keywords
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│ Boundary 4: Validator │ ──► Independent code path recomputes numbers; rejects NaN/inf
└───────────────────────┘
      │
      ▼
┌───────────────────────┐
│ Boundary 5: Explainer │ ──► Grounding guard filters all cited figures against computed evidence
└───────────────────────┘
```

**Boundary 3b — Tool calls (agentic mode only).** Inserted between Boundary 3
and Boundary 4 when `agentic` is enabled: the tool name must exist in
`OPERATIONS`, params are Pydantic-validated and column-canonicalised through
the same `validate_plan_columns` path the planner uses, and the loop is bounded
by `LoopBudget`. The model may only *select* registered operations and supply
their parameters — it never emits executable code. Failed calls return readable
errors to the model so it can correct itself, and the whole run is then rebuilt
into a plan so Boundaries 4 and 5 apply unchanged.

---

## 3. Test Oracles (Authoritative Correctness)

To verify whether Insight is *actually correct* rather than merely returning plausible-looking results, test agents must check against these explicit oracles:

### 1. Group Aggregations (`group_aggregate`)
* **Reference Implementation**: `df.groupby(group_by, observed=True)[metric.column].agg(metric.agg)`
* **Counts**: Non-null rows counted for metric; total size counted for `row_count`.
* **Folded Levels (`other_after`)**: Levels past `other_after - 1` by frequency must be folded into `"Other"`. The sum of `"Other"` plus remaining rows must equal the filtered population total.
* **Tolerance**: $10^{-6}$ relative tolerance for floats; exact match for integers.

### 2. OLS Regression (`regression`)
* **Reference Implementation**: `scipy.stats.linregress` for 1 feature or `numpy.linalg.lstsq` with design matrix $[1, X]$ on complete rows (`dropna()`).
* **$R^2$ Formula**: $1.0 - \frac{\text{RSS}}{\text{TSS}}$ where $\text{TSS} = \sum (y - \bar{y})^2$.
* **Adjusted $R^2$**: $1.0 - (1.0 - R^2) \frac{n - 1}{n - k - 1}$.
* **Standardized Beta**: $\beta_j \times \frac{\sigma_{X_j}}{\sigma_Y}$.
* **Listwise Deletion**: Must report `rows_excluded_incomplete` whenever nulls are dropped.

### 3. One-Way ANOVA (`anova_test`)
* **Reference Implementation**: `scipy.stats.f_oneway(*[group.dropna().values for _, group in df.groupby(group_col)[val_col]])`.
* **Effect Size ($\eta^2$)**: $\frac{\text{SS}_{\text{between}}}{\text{SS}_{\text{total}}} = \frac{F \times \text{df}_{\text{between}}}{F \times \text{df}_{\text{between}} + \text{df}_{\text{within}}}$.
* **Constraint**: Minimum 3 distinct groups, each with $n \ge 2$.

### 4. Correlation (`correlation` & `statistical_test`)
* **Reference Implementation**: `scipy.stats.pearsonr` or `scipy.stats.spearmanr`.
* **Invariants**: Matrix diagonal must be identically $1.000000$; matrix must be symmetric ($M_{ij} == M_{ji}$ within $10^{-6}$).

### 5. Outlier Detection (`detect_outliers`)
* **IQR Rule**: $\text{IQR} = Q_3 - Q_1$; Bounds: $[Q_1 - 1.5 \times \text{IQR},\; Q_3 + 1.5 \times \text{IQR}]$.
* **Z-Score Rule**: Mean $\mu$, Std $\sigma$; Flagged where $|\frac{x - \mu}{\sigma}| > 3.0$.
* **Outlier Share**: $\frac{\text{flagged rows}}{\text{total valid rows}}$.

### 6. Grounding Guard (`insight.validation.grounding`)
* **Extraction Pattern**: Regex `-?\d[\d,]*(?:\.\d+)?`.
* **Allowed Figures Without Exact Evidence Match**:
  * Year numbers: $1900 \le x \le 2100$.
  * Small calendar/rank integers: $1 \le |x| \le 31$.
  * Percentage variants: If $v \in \text{floats}$, $v \times 100 \pm 0.11$ is valid (e.g., $0.2853$ permits $28.53\%$).
* **Failure Condition**: Any figure outside these rules triggers a grounding violation and rewrite request.

---

## 4. Audit of Existing Test Suites & Evaluations

### Offline Unit & Integration Test Suite (`tests/`)

The repository contains **33 tracked test files** (`git ls-files tests/`) that execute completely offline without network access or live API keys (using `ScriptedLLM`/`StubChain` from `conftest.py`; the tool-calling fakes `ScriptedToolLLM`/`ToolStubChain` live in `tests/test_agentic_pipeline.py` itself). Assert `pytest -q` exits 0 — **never** an absolute test count (see Documentation Discrepancies).

| Test File | What It Actually Exercises | Weaknesses / False-Confidence Risks |
|---|---|---|
| `test_operations.py` | 15 deterministic operations under normal data | Does not test extreme null cascades (covered in `test_invariants.py`). |
| `test_statistics.py` | SciPy routing, t-test, ANOVA, OLS, isotonic PAV | Relies on synthetic distributions; extreme collinearity not fully stressed. |
| `test_sql_engine.py` | DuckDB session setup, table sanitization, keyword guards, watchdog interrupt, describe | Tests keyword blocklist, but does not test exotic DuckDB extensions. |
| `test_validator.py` | Sanity checks, empty table detection, MCAR caveats | Does not test multi-step DAG validation where intermediate steps fail. |
| `test_verification.py` | Independent recomputation for group_agg, top_n, value_counts, correlation | Only tests steps with `input_step=None`; skips explicit DAG inputs. |
| `test_executor.py` | Sequential executor: `filter_rows` mutation flows into the next step, unknown op recorded, invalid params reported, partial failure still yields surviving tables | Deliberately sequential; no concurrency or cancellation tests. |
| `test_chaining.py` | End-to-end `input_step` DAG: filter→group→top_n, chained statistical test on a filtered frame, unknown/forward/self `input_step` rejection, explicit `input_step` beating the implicit working frame | Scripted plans only; the planner's *choice* of a chain is not tested. |
| `test_orchestrator.py` | Full pipeline with scripted LLM responses | Uses `ScriptedLLM`; does not test real LLM reasoning variance. |
| `test_planner.py` | JSON extraction, markdown stripping, self-repair loop | Tests single repair; does not test malformed parameters within valid JSON. |
| `test_cache.py` | Cache key hashing (incl. agentic/planned separation), disk load/save, TTL & size budget sweeping | Fast local disk IO; does not test concurrent cache writes. |
| `test_conversation.py` | Session state extraction from plans, serialization | Tests state representation, not long multi-turn semantic drift. |
| `test_briefing.py` | Heuristic detectors (trend, imbalance, correlation, outliers) | Thresholds are hardcoded; edge cases with $N < 10$ need caution. |
| `test_artifacts.py` | dbt model generation and Slack alert monitor JSON | Validates string templates, not dbt compilation in a real dbt project. |
| `test_alert_runner.py` | Monitor loading, threshold evaluation, state-change alerting, webhook payload shape, destructive-query rejection | Webhook is faked (no live Slack call); no persistence of fired-state across restarts. |
| `test_sql_webapp.py` | FastAPI endpoints for `/export/dbt` and `/export/alert` | In-memory session fixture; single-threaded test client. |
| `test_webapp.py` | FastAPI routes (`/`, `/upload`, `/query`, `/healthz`, clear) | Tests SSE via TestClient; buffering behavior differs from real browsers. |
| `test_webapp_agentic.py` | `/query` + `/api/query` with `agentic`, SSE `tool_call`/`tool_result` frames, agent chip rendering, explicit `agentic=0` override | Stub chain, not a live provider; no browser round trip. |
| `test_agentic.py` | Deterministic follow-up suggestion ranking | Heuristic scoring only. |
| `test_agentic_tools.py` | Tool-schema generation (every op → a tool, self-contained JSON, Anthropic shape), dispatch rejection paths, loop budgets/throttle | Loop-only; no orchestrator integration. |
| `test_agentic_pipeline.py` | Loop→rebuild→shared-tail integration: multi-call turns, `input_step` chaining, repeated-failure abort, SSE event sink, plan/execution rebuild, chart synthesis, meta telemetry, orchestrator E2E, grounding flags invented numbers, throttle/provider-failure error paths, cache separation, `plan_only` precedence, `artifacts_dir=None` | Entirely scripted responses — offline by construction; live behavior is exercised separately by `evaluation/run_eval_agentic.py` (see the Golden-Question Evaluation Harness section for what has actually been measured). |
| `test_text_streaming.py` | Token-by-token SSE deltas, `first_token_ms`, live typing-bubble rendering, reduced-motion caret | Chunking and browser rendering are stubbed/asserted on markup, not observed live. |
| `test_prompt_cache.py` | Planner system-prompt byte stability (the prefix-cache prefix), `cache_prompt` flag set, `cached_prompt_tokens` accounting | Uses a stubbed usage block; real provider cache hits not asserted. |
| `test_hitl_plan_approval.py` | `plan_review` stop (never executes), editable plan payload, approved plan executes without re-planning, approved plan re-validated against the schema, user edits honored, `INSIGHT_PLAN_APPROVAL` forces the gate, plan review never cached, `analyze()` rejects `plan_only`, client controls exist | `analyze()` raising for `plan_only` is by design (it cannot express a plan-only outcome). |
| `test_upload_polish.py` | Background briefing polish: deterministic findings without an LLM round-trip, in-place session update, stale-generation guard, pending state in the template | Polishing itself is stubbed. |
| `test_byok.py` | `X-Insight-Key` header propagation, env fallback, `require_user_key` blocking/allowing | No live key validation. |
| `test_chain_fallback.py` | `_Chain` failover: empty-content response falls through to the next provider; all-empty raises `AllProvidersFailedError`. (Note: the empty-content guard now exempts tool-call-only responses, which is why this still passes in agent mode) | Simulated failures only; no real 429/5xx timing behaviour. |
| `test_custom_provider.py` | Custom OpenAI-compatible endpoint registration and the discovery toggle (`INSIGHT_CUSTOM_DISCOVERY`) | Does not contact the custom endpoint. |
| `test_invariants.py` | Stresses the analytics operations under 30% missing data; asserts grounding invariants | Property-style checks, not exhaustive. |
| `test_renderer.py` | Matplotlib chart generation across themes | Checks PNG creation and dimensions; does not check visual chart beauty. |
| `test_profiler.py` | Column role inference and dataset fingerprinting | Relies on column names for identifier heuristic (`order_id`, `uuid`). |
| `test_providers.py` | Provider URL construction, headers, retry backoff, tool-call payload/parse shapes | Mocks HTTP client; does not hit real provider endpoints. |
| `test_registry.py` | Model catalog, sorting, text-model filtering, custom provider env mapping | Relies on static catalog definitions. |
| `test_evaluation.py` | Scoring logic (`_numeric_close`, `_ops_satisfied`) | Tests harness itself, not agent performance. Needs `evaluation/data/`, which is gitignored — CI regenerates it. |

---

### Golden-Question Evaluation Harness (`evaluation/`)

* **Runner (single-shot planner)**: `evaluation/run_eval.py`
* **Runner (agentic A/B)**: `evaluation/run_eval_agentic.py` — runs the *same* cases through `insight.tools.loop` and scores them with the *same* `score_case`, so the two designs are directly comparable. Uses production wiring (the app's `AGENTIC_SYSTEM`/`build_agentic_messages`, a real `DuckSession` so `sql_query` works) and emits side-by-side telemetry: `llm_calls`, prompt/completion tokens, `wall_s`, throttling, which models actually served the calls, and per-case PASS/FAIL with identical reason strings. `--delay` spaces calls; `--max-iterations`/`--max-llm-calls` override budgets (smoke defaults 3/6; the app ships 5/9); `--record` keeps every raw request/response (`transcript.jsonl`) for offline replay; results are written incrementally so a throttled or interrupted run still yields analysable data. `evaluation/ab_stats.py` aggregates k repetitions per arm into per-case passes/k, pass rate with a 95% Wilson CI, mean calls/wall and token totals.
* **Test Cases**: `evaluation/cases.json` (20 curated cases)
* **Datasets**:
  * `sales.csv` (seeded synthetic)
  * `ecommerce.csv` (seeded synthetic)
  * `support.csv` (seeded synthetic)
  * `retail_250k.csv` (seeded 250,000-row stress dataset)
  * `diamonds.csv` (53,940-row GIA benchmark dataset)
* **Scoring Rules**:
  1. `ops_satisfied`: Plan must execute the required operations (or valid alternative).
  2. `contains_satisfied`: Explanation must contain key factual strings (e.g. `"West"`, `"Premium"`).
  3. `numeric_found`: Citations in calculations or result tables must match ground truth within `tol_pct` (default $\pm 0.5\%$).
  4. `validation_valid`: Result validator must report `valid == True`.
* **Historical Benchmark Results**:
  * **Groq (`openai/gpt-oss-120b`)**: 20/20 cases passed (Live run: 2026-08-22).
  * **OpenCode (`x-preview-f-free`)**: 8/8 real-world diamonds battery passed with 0 unverified figures (`evaluation/REAL_WORLD_LOG.md`).
  * **Agentic mode (live, in progress)**: first live runs on Groq `openai/gpt-oss-120b` exposed — and the fix landed for — an OpenAI wire-format bug in the assistant tool-call history (every 2nd turn 400'd; offline fakes never crossed the provider wire layer). After the fix: **17/20 at the eval smoke budget (3 iterations/6 calls)** — all 3 failures are `iteration_cap_reached` (the model was still mid-analysis when the cheap budget ran out; the app ships 5/9) — and **10/10 so far at the production budget (5/9)** before free-tier TPM backoff stopped the run (~178 s/case wall; `throttled=0` because internal retries eventually succeed). A full k≥3 A/B aggregated with `evaluation/ab_stats.py` is pending a higher-quota endpoint (NVIDIA NIM is wired via `INSIGHT_CUSTOM_*`). Do not quote a final agentic pass rate until that lands.

---

## 5. Master Test Matrix

| ID | Area | Scenario | Input / Payload | Procedure | Expected Outcome | Oracle | Severity |
|:--:|---|---|---|---|---|---|:--:|
| **ING-01** | Ingestion | Empty file upload | 0-byte `.csv` file | `POST /upload` with empty file | HTTP 400 with `"empty file"` error message; session unchanged. | API Contract | High |
| **ING-02** | Ingestion | Multi-file upload | 3 CSV files (`orders.csv`, `customers.csv`, `items.csv`) | `POST /upload` with multi-part files | Tables registered as `orders`, `customers`, `items` in DuckDB session; multi-table schema described. | `DuckSession.describe()` | High |
| **ING-03** | Ingestion | Non-standard encoding | CSV encoded in `ISO-8859-1` or `cp1252` | Upload via web UI | Auto-detected encoding or graceful UTF-8 decode with replacement characters. | `profile.row_count > 0` | Med |
| **OP-01** | Operations | Group Aggregate with `other_after` | Categorical column with 20 levels; `other_after=5` | Run `group_aggregate` | Top 4 levels retained; remaining 16 folded into `"Other"`; sum of counts matches total. | Reference groupby | High |
| **OP-02** | Operations | Missing data survival | 40% NaNs in numeric metric | Run `summarize` & `distribution` | Calculations report valid floats; MCAR warning emitted; no uncaught exceptions. | `np.isfinite` check | High |
| **OP-03** | Operations | Zero-variance numeric column | Column with all values equal to `100.0` | Run `distribution` & `correlation` | Distribution uses single bin or handles 0 width; correlation returns NaN/0 without crash. | No ZeroDivisionError | Med |
| **OP-04** | Operations | Trend on irregular dates | Datetime column with gaps and string dates | Run `trend(freq='month')` | Correct period bucketing; period-over-period deltas computed. | Reference resample | Med |
| **STAT-01**| Statistics | Welch t-test vs Mann-Whitney | Binary group $\times$ skewed numeric | Run `statistical_test` | Router checks normality/sample size; selects appropriate test; computes effect size. | Reference SciPy | High |
| **STAT-02**| Statistics | ANOVA with 3+ groups | Categorical (4 levels) $\times$ numeric | Run `anova_test` | Reports $F$-statistic, $p$-value, $\eta^2$ effect size, and per-group means table. | `scipy.stats.f_oneway` | High |
| **STAT-03**| Statistics | Multi-variable OLS regression | 1 target $\times$ 3 features | Run `regression` | Returns $R^2$, adjusted $R^2$, standardized betas, coefficients, and $p$-values. | `numpy.linalg.lstsq` | High |
| **STAT-04**| Statistics | Isotonic Probability Calibration | Predicted score $\times$ binary label | Run `isotonic_calibration` | Emits PAV reliability table, raw Brier, calibrated Brier, and ECE. | Scikit-learn isotonic | Med |
| **SQL-01** | SQL Engine | Multi-table relational JOIN | SQL query joining `orders` and `customers` on `id` | Run `sql_query` | Executes in DuckDB; returns merged table within timeout limit. | Reference SQL result | High |
| **SQL-02** | SQL Sandbox | Adversarial `ATTACH` attempt | `ATTACH 'test.db' AS db;` | Pass to `sql_query` | Blocked immediately with `"'ATTACH' statements are not permitted"`. | Invariant INV-06 | Critical |
| **SQL-03** | SQL Sandbox | File system access attempt | `SELECT * FROM read_csv('/etc/passwd')` | Pass to `sql_query` | DuckDB error: external access disabled (`enable_external_access=false`). | Invariant INV-06 | Critical |
| **SQL-04** | SQL Sandbox | Infinite loop / runaway query | `SELECT count(*) FROM range(10000000000)` | Pass to `sql_query` | Watchdog thread interrupts at `timeout_s` (10s); raises timeout error. | Invariant INV-07 | High |
| **PLN-01** | Planner | Hallucinated column name | Question asks for `"profit"` (dataset only has `"revenue"`) | Send to `Planner.plan` | First attempt fails validation; repair prompt asks LLM to correct using valid columns. | Invariant INV-03 | High |
| **PLN-02** | Planner | Circular DAG reference | Step 1 specifies `input_step: 2` | Send to `validate_plan_columns` | Plan rejected with DAG validation error (`input_step >= step_id`). | Invariant INV-08 | High |
| **GRD-01** | Grounding | Fabricated summary total | Answer cites `"Total revenue was $1,500,000"` when table shows `1,200,000` | Run `find_ungrounded_numbers` | `"1,500,000"` flagged as ungrounded; triggers rewrite prompt to LLM. | Invariant INV-05 | Critical |
| **GRD-02** | Grounding | Legitimate percentage formatting | Table shows `0.2853`; answer cites `"28.5%"` | Run `find_ungrounded_numbers` | Accepted as valid scaled representation ($ candidate \times 100 $). | Grounding Oracle | Med |
| **LLM-01** | Provider | Primary provider rate limit (429) | Provider A raises HTTP 429 | Run query via `_Chain` | Automatically catches 429 and fails over to Provider B in chain. | `test_chain_fallback` | High |
| **LLM-02** | Provider | Bring-Your-Own-Key header | Request contains `X-Insight-Key: sk-user-test` | `POST /query` | Provider initialized with visitor's key instead of environment default. | `test_byok` | High |
| **CONV-01**| State | Follow-up query filter preservation | Turn 1: filter `region == 'West'`; Turn 2: *"What about laptops?"* | Run consecutive turns | Session state retains `region == 'West'`; adds `category == 'laptops'`. | `AnalysisSessionState` | Med |
| **EXP-01** | Exports | dbt model artifact bundle | Successful SQL analysis step | `POST /export/dbt?step_id=0` | Returns valid `.zip` containing `model.sql`, `schema.yml`, and `README.md`. | Zipfile inspection | Med |
| **EXP-02** | Exports | Full DOCX analysis report | Session with 2 turns + charts | `GET /export/docx` | Returns valid `.docx` with briefing findings, headings, tables, and embedded PNGs. | `docx.Document` parse | Med |
| **CCH-01** | Cache | Exact query hit | Repeated identical question on same dataset | Run `analyze(use_cache=True)` | Second call returns in $< 100$ms with `meta['cache_hit'] == True`. | Invariant | Med |
| **CCH-02** | Cache | LRU & TTL cache eviction | Cache folder exceeds `query_cache_max_mb` | Run `sweep_cache()` | Oldest entries removed until total size is within budget. | `sweep_cache` return | Low |
| **AGENT-01** | Agentic | Iteration budget honoured | Router that always returns another tool call | `run_loop(..., budget=LoopBudget(max_iterations=2))` | `outcome.iterations == 2`; `outcome.error == "iteration_cap_reached"`. | INV-11 | High |
| **AGENT-02** | Agentic | LLM-call budget honoured | Router that always returns another tool call | `run_loop(..., budget=LoopBudget(max_iterations=99, max_llm_calls=2))` | `outcome.llm_calls <= 2`; `outcome.error == "llm_call_budget_exhausted"`. | INV-11 | High |
| **AGENT-03** | Agentic | Identical-failure abort | Tool call with a hallucinated column, repeated | `run_loop(..., max_iterations=10)` | Stops after 3 identical failures; `outcome.error == "repeated_failing_call"`. | INV-11 | Med |
| **AGENT-04** | Agentic | Unknown op / hallucinated column | Call `drop_table`; call `group_aggregate` on `"profit"` | `insight.tools.runtime.dispatch` | `ToolResult.ok is False` with a message listing available ops/columns; **no op executes**. | INV-12 | Critical |
| **AGENT-05** | Agentic | DAG chaining via `input_step` | `group_aggregate(region, sum revenue)` then `top_n(sort_by="total_revenue", input_step=0)` | `run_loop` | Second call `ok is True`, `input_step == 0`, and its table has 2 rows from the *derived* column. | INV-12 | High |
| **AGENT-06** | Agentic | Agentic answer is grounded | Explainer answers citing `999999` (never computed) | `analyze_stream(..., AnalysisRequest(agentic=True))` | `"999999"` in `meta["unverified_figures"]`; warning Evidence emitted; rewrite attempted. | INV-05, INV-13 | Critical |
| **AGENT-07** | Agentic | Provider 429 during the loop | First `generate` raises rate-limit | `analyze_stream(..., agentic=True)` | Response `error` starts with `"throttled"`; `meta["agentic_throttled"] is True`; no fabricated answer. | INV-11 | High |
| **AGENT-08** | Agentic | `plan_only` wins over agentic | `AnalysisRequest(agentic=True, plan_only=True)` | `analyze_stream` | No `tool_call`/`tool_result` events; a `plan_review` event is emitted. | INV-14 | Critical |
| **AGENT-09** | Agentic | Cache does not cross modes | Same question, `agentic=None` vs `agentic=True` | `cache_key_for(...)` | Different keys. | — | Med |
| **AGENT-10** | Agentic | Live SSE trail | `POST /query` with `agentic=1` | Parse SSE frames | `tool_call` frame carries `name`/`arguments`; `tool_result` frame carries `ok`; result HTML shows the agent chip. | INV-12 | High |
| **AGENT-11** | Agentic | Anthropic wire shape | Loop request when the chain's primary provider is Anthropic | `AnthropicProvider._payload(request)` | `tool_use` blocks for assistant turns; `tool_result` in a user turn; `input_schema` tool definitions; `tool_choice={"type":"auto"}`. | — | High |
| **ALERT-01** | Alert runner | Destructive monitor query | Monitor JSON with `"query": "DROP TABLE sales;"` | `Runner.cycle()` | Query rejected by `DuckSession.validate_statement`; nothing posted. | INV-06 | Critical |
| **ALERT-02** | Alert runner | State-change alerting | Monitor that stays triggered across 2 cycles | `Runner(repeat=False).cycle()` ×2 | Posts once; second cycle logs "still triggered" and does not re-post. | — | High |
| **ALERT-03** | Alert runner | Threshold semantics | `{"metric_column":"total","op":">","value":100}` over a total of 770 | `Runner.evaluate` | `triggered is True`; threshold note records the compared value. | — | High |
| **ALERT-04** | Alert runner | Webhook payload | Triggered monitor with a URL | `Runner.post` (urlopen faked) | POST body is Slack-compatible JSON with `text` containing monitor name + row count. | — | Med |

---

## 6. How Another LLM Should Test This Repository

When an autonomous agent is tasked with testing, validating, or attacking this codebase, follow this strict operating procedure:

```
[ Step 1: Read TEST.md & settings.py ]
               │
[ Step 2: Locate Subsystem in Repo Map ]
               │
[ Step 3: Identify Authoritative Oracle ]
               │
[ Step 4: Execute Smallest Offline Test First ]
               │
[ Step 5: Capture Exact Evidence (Logs / Values) ]
               │
[ Step 6: Compare Observed vs. Oracle ]
               │
[ Step 7: Classify Result (See Taxonomy) ]
               │
[ Step 8: Run Full Offline Regression Suite ]
```

### Failure Classification Taxonomy
When an anomaly occurs, classify it into exactly one category:
1. **Application Defect**: Implementation deviates from its contract or invariant (e.g. OLS calculates incorrect standard error, grounding guard misses a hallucinated figure, SQL sandbox allows `ATTACH`).
2. **Test Defect**: The test expectation is flawed (e.g. wrong tolerance, outdated schema expectations).
3. **Environment / Configuration Issue**: Missing `.env`, missing Python virtual environment, unavailable port.
4. **External Provider Failure**: Live LLM rate limit (429), upstream timeout (504), or empty response from external model provider.
5. **Expected Behavior / Intentional Rejection**: System correctly rejects an invalid plan, malicious SQL statement, or ungrounded draft.
6. **Known Limitation**: Pre-documented constraint (e.g. single-worker memory limitation in Docker container).

### Standard Evidence Template for Reporting Tests
```markdown
Test ID: <e.g. SQL-02>
Date: <YYYY-MM-DD>
Commit: <git-sha>
Subsystem: <e.g. DuckDB SQL Sandbox>

Objective: Verify blocked keyword enforcement
Input: "SELECT * FROM t; DROP TABLE t;"
Procedure: Pass input to DuckSession.validate_statement()
Expected: ValueError: exactly one SQL statement is allowed
Observed: ValueError: exactly one SQL statement is allowed
Oracle: Invariant INV-06
Result: PASS
Classification: Expected Behavior
Regression Impact: None
```

### Reusable Diagnostic Prompts for Agents

#### Prompt 1: Investigating Numerical Discrepancies
> *"You are auditing a numerical discrepancy between an Insight analysis result and a reference calculation. Read `insight/validation/verification.py` and the target operation in `insight/analytics/operations.py`. Extract the exact formula used for the metric. Identify if missing values were dropped listwise, whether ddof=1 was used for sample standard deviation, and whether group counts reflect valid rows or total rows."*

#### Prompt 2: Auditing Answer Grounding
> *"You are a strict numerical auditor. Review the draft answer text alongside the calculations list and tabular outputs. Extract every integer and floating-point number from the text. Check whether each number exists in the evidence, represents a valid percentage transformation ($x \times 100$), or represents an allowed year/date. Flag any figure that cannot be strictly verified."*

#### Prompt 3: Adversarial SQL Sandbox Testing
> *"You are performing security testing on `insight/analytics/sql_engine.py`. Inspect `DuckSession.validate_statement()` and the DuckDB initialization settings. Construct 5 adversarial payloads designed to bypass the single-statement split, comment stripping, keyword blocklist, or external access restrictions. Test each against the validator and record the rejection reason."*

---

## 7. Baseline Handover & Regression

### Minimal Smoke Test
To verify the application is alive and healthy:
```bash
# 1. Verify healthcheck endpoint
curl -s http://127.0.0.1:8000/healthz
# Expected output: {"status":"ok"}

# 2. Run core offline test suite
pytest tests/test_operations.py tests/test_sql_engine.py tests/test_verification.py -q
# Expected output: All tests pass in < 5 seconds

# 3. Run the agentic layer (offline, no key needed)
pytest tests/test_agentic_tools.py tests/test_agentic_pipeline.py tests/test_webapp_agentic.py tests/test_alert_runner.py -q
```

### Core Regression Commands
```bash
# Full offline test suite (tracing & query cache forced off)
pytest -q

# Run local invariants & robustness tests (if present locally)
pytest tests/test_invariants.py tests/test_byok.py tests/test_chain_fallback.py -q

# Run live golden-question evaluation harness (requires live GROQ_API_KEY in .env)
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b --limit 5

# Agentic A/B against the SAME golden cases and grader (live, needs quota)
python evaluation/run_eval_agentic.py --model groq/openai/gpt-oss-120b --limit 5 --delay 2

# Aggregate repeated runs into stats (per-case passes/k, 95% Wilson CI, cost)
python evaluation/ab_stats.py --agentic DIR1,DIR2 --baseline DIR3,DIR4
```

### Manual Agentic Smoke Test
1. Start the app (`uvicorn webapp.app:app --port 8000`), upload a CSV.
2. Tick **Agent mode** next to the composer.
3. Ask a question that needs more than one lookup, e.g. *"Which region leads revenue, and how far ahead is it?"*
4. Confirm a **live tool trail** appears (chips turn from dashed to solid as each call returns), then the answer + charts.
5. Open **View calculation** — the steps shown are the agent's actual tool calls; numbers must match a manual pandas computation.
6. Untick Agent mode and repeat: the trail must disappear (single-shot planner path).

### Known Limitations & Deferred Capabilities
1. **Single-Worker Container Constraint**: In-memory `WebSession` state (uploaded CSV bytes, profile, conversation history) lives in Python process memory. The Docker container must run exactly **one** Uvicorn worker. Horizontal multi-worker scaling requires externalizing session state (e.g. to Redis).
2. **Windows cp1252 Unicode**: Windows terminal printing Unicode characters (e.g. `…`, `\u202f`) can crash scripts unless `sys.stdout.reconfigure(encoding="utf-8")` is invoked.
3. **PowerShell UTF-8 BOM**: PowerShell 5.1 `Set-Content` can inject a Byte Order Mark (BOM). All parsers must handle UTF-8-SIG or strip BOMs.
4. **Agentic mode's live coverage is partial**: the tool loop, plan rebuild, shared tail, SSE trail, budgets and failure paths are covered offline by `test_agentic_pipeline.py` / `test_webapp_agentic.py`; live golden-case runs exist (17/20 @ 3/6; 10/10 @ 5/9, partial — see the harness section) but the full k≥3 A/B has not completed. Do not quote a final agentic pass rate as if it were measured.
5. **Agentic cost/latency is only partially measured**: first live runs averaged ~2.0–2.2 `llm_calls` per case, with wall time dominated by free-tier TPM backoff (minutes per case on the Groq free tier, invisible to `throttled`). `INSIGHT_AGENTIC_MAX_LLM_CALLS` (default 9) is still a safety bound, not a tuned budget — tune it against the k≥3 run before recommending a default.
6. **Alert-runner state is per-process**: fired-monitor state lives in memory, so a restart re-posts an already-triggering monitor, and multi-instance deployments have no shared state. It is a reference worker, not a durable scheduler; cron/interval supervision is the caller's job.
7. **Deferred Features (Do Not Implement Unprompted)**: Python code-interpreter sandboxes (omitted by design; DuckDB covers analytics safely), multi-user authentication, and native PDF exports.

### Documentation Discrepancies
* **Test Counts**: Do not assert an absolute test count anywhere in the docs. The suite's pass/skip split varies by platform and by which optional corpora are present — `diamonds.csv` is a manual download and `evaluation/data/` is gitignored, so a bare clone legitimately skips tests that a fully populated checkout runs. **Assert the outcome (`pytest -q` exits 0), never the number.** Hardcoded counts in `README.md`, `HOW_TO_RUN.md`, `HANDOVER.md` and this file were removed on 2026-09-30 after drifting stale twice — and this very bullet was left holding a stale "217 collected tests", which is the exact failure it warns about. Counts of *tracked files* (`git ls-files tests/`) are stable and can be stated.
* **CI was silently red**: Until 2026-09-30 the GitHub Actions workflow had never passed. `evaluation/data/` is gitignored, so `tests/test_evaluation.py` failed on any fresh clone with `missing dataset sales.csv`; `pyyaml` was imported by `tests/test_artifacts.py` but never declared in `requirements.txt` (it was only present transitively via Streamlit). CI now runs `python evaluation/generate_datasets.py` before pytest, and PyYAML is a declared dependency. **Lesson: a test suite that only passes on the author's machine is not a safety net.** If CI is red, treat it as a code bug, not a flaky environment.
* **Streamlit vs. WebApp**: `README.md` initially presented the Streamlit UI as primary. The primary production interface is now the FastAPI web application in `webapp/` (Uvicorn port 8000), with Streamlit maintained as a secondary interface.
* **GitHub repo metadata**: The repository description claimed "Built with Streamlit, LangChain, and Groq API". LangChain was never a dependency (zero imports; only `langsmith` for tracing) and the LLM layer is hand-rolled over `httpx`. Corrected 2026-09-30.
* **A whole feature was gitignored, not absent (2026-10-13)**: the agentic tool-use layer (`insight/tools/`), its tests, and `evaluation/run_eval_agentic.py` existed in the working tree but sat under a "Temporarily ignored untracked files" block in `.gitignore` — so no branch, CI run or fresh clone contained them, and `git status` was the only place they showed up. Worse, the scaffolding was **incomplete in a way no doc said**: there was no `INSIGHT_AGENTIC` setting, and nothing referenced `insight.tools` from the orchestrator or webapp. Lesson: *a module that no caller imports is not a feature*. Grep for callers, not for files, when auditing scope claims.
* **An SSE drain bug that only a real hang reveals (2026-10-13)**: the agentic branch streamed loop events with `yield queue.get()` — yielding an **un-awaited coroutine**. The queue therefore never drained and the generator yielded fresh coroutines forever: the request hung until timeout, and the offline suite had no test that exercised a successful agentic SSE run to completion. Fixed to `yield await queue.get()`. Lesson: generator pipelines that *forward* another queue/task need an end-to-end drain test, not just per-event assertions.
* **`_Chain.generate` rejected tool-calling responses**: the empty-content guard (`raise InsightError("returned empty content")`) fired on legitimate tool-call turns, where content is `""` and the call lives in `tool_calls`. Any provider abstraction with tool support must exempt tool-only responses from text-emptiness checks.
* **A wire-format bug no offline fake could see (2026-10-16)**: assistant `tool_calls` were serialised in the internal shape `{id, name, arguments: dict}` instead of OpenAI's `{id, type: "function", function: {name, arguments: <json string>}}`. Every *second* turn of an agentic run was a hard HTTP 400 on every OpenAI-compatible provider (first live smoke, Groq: `'messages.1.tool_calls.0.type' : property 'type' is missing`). The scripted-LLM suite could not catch it because tool turns never crossed the provider HTTP layer. Lesson: fakes that bypass a serialization boundary do not test that boundary — keep one test per provider that goes through `httpx.MockTransport` and asserts the exact wire shape, plus at least one live multi-turn smoke before trusting a tool-calling path.
