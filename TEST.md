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
| **Planning Engine** | `insight/planning/planner.py`<br>`insight/planning/prompts.py` | Generates structured JSON plans with bounded self-repair (`max_repair_attempts = 1`). Injects SQL schema for multi-table queries | `PlanValidationError` (surfaced cleanly to user) |
| **Op Registry** | `insight/analytics/operations.py` | 15 deterministic operations: `summarize`, `value_counts`, `top_n`, `group_aggregate`, `distribution`, `detect_outliers`, `trend`, `correlation`, `statistical_test`, `anova_test`, `regression`, `isotonic_calibration`, `sql_query`, `filter_rows`, `compare_subsets` | `OperationError` caught by executor, marked as failed step |
| **DuckDB SQL Sandbox** | `insight/analytics/sql_engine.py` | Embedded in-memory DuckDB with `enable_external_access=false`, `lock_configuration=true`, single SELECT/WITH guard, keyword blocklist, and 10s interrupt watchdog | `ValueError` with security or timeout explanation |
| **Statistical Engine** | `insight/analytics/statistics.py` | SciPy test routing (Welch t-test, Mann-Whitney U, Chi-Square + Cramér's V, ANOVA + $\eta^2$, OLS regression, Isotonic PAV) | Rejection of small samples, fallback to non-parametric tests |
| **Validation Engine** | `insight/validation/result_validator.py`<br>`insight/validation/verification.py`<br>`insight/validation/grounding.py` | Three-layer gate: (1) Sanity & NaN scan, (2) Independent pandas recomputation (second code path), (3) Answer number-level grounding verification | Triggers plan repair or explainer rewrite; flags unverified numbers |
| **LLM Provider Hub** | `insight/llm/registry.py`<br>`insight/llm/providers/*`<br>`insight/llm/base.py` | Multi-provider fallback chain (Groq, Mistral, OpenAI, Anthropic, Ollama, OpenRouter, Custom). Live model discovery and BYOK (`X-Insight-Key`) | `AllProvidersFailedError`, automatic fallback down provider list |
| **Conversation State** | `insight/conversation/state.py` | Cross-turn active filter persistence, tracked dimensions and metrics, session context serialization | State isolation per session cookie / session ID |
| **Query Cache** | `insight/cache.py` | SHA-256 fingerprint + question + model + state cache with TTL days (default 14) and max disk budget (default 512MB LRU) | Bypassed when `INSIGHT_QUERY_CACHE=off` or plan validation fails |
| **Export Engines** | `insight/reports/docx.py`<br>`insight/reports/artifacts.py` | Multi-turn `.docx` report generation, dbt model + schema.yml `.zip` export, Slack-ready alert JSON monitor | Empty table or file write error handled gracefully |
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

The repository contains **25 test files** designed to execute completely offline without network access or live API keys (using `ScriptedLLM` and `StubChain` fixtures in `conftest.py`).

| Test File | Lines | What It Actually Exercises | Weaknesses / False-Confidence Risks |
|---|:---:|---|---|
| `test_operations.py` | 206 | 15 deterministic operations under normal data | Does not test extreme null cascades (covered in local `test_invariants.py`). |
| `test_statistics.py` | 452 | SciPy routing, t-test, ANOVA, OLS, isotonic PAV | Relies on synthetic distributions; extreme collinearity not fully stressed. |
| `test_sql_engine.py` | 176 | DuckDB session setup, table sanitization, keyword guards, watchdog interrupt, describe | Tests keyword blocklist, but does not test exotic DuckDB extensions. |
| `test_validator.py` | 115 | Sanity checks, empty table detection, MCAR caveats | Does not test multi-step DAG validation where intermediate steps fail. |
| `test_verification.py` | 95 | Independent recomputation for group_agg, top_n, value_counts, correlation | Only tests steps with `input_step=None`; skips explicit DAG inputs. |
| `test_orchestrator.py`| 99 | Full pipeline with scripted LLM responses | Uses `ScriptedLLM`; does not test real LLM reasoning variance. |
| `test_planner.py` | 73 | JSON extraction, markdown stripping, self-repair loop | Tests single repair; does not test malformed parameters within valid JSON. |
| `test_cache.py` | 184 | Cache key hashing, disk load/save, TTL & size budget sweeping | Fast local disk IO; does not test concurrent cache writes. |
| `test_conversation.py`| 156 | Session state extraction from plans, serialization | Tests state representation, not long multi-turn semantic drift. |
| `test_briefing.py` | 83 | Heuristic detectors (trend, imbalance, correlation, outliers) | Thresholds are hardcoded; edge cases with $N < 10$ need caution. |
| `test_artifacts.py` | 50 | dbt model generation and Slack alert monitor JSON | Validates string templates, not dbt compilation in a real dbt project. |
| `test_sql_webapp.py` | 104 | FastAPI endpoints for `/export/dbt` and `/export/alert` | In-memory session fixture; single-threaded test client. |
| `test_webapp.py` | 191 | FastAPI routes (`/`, `/upload`, `/query`, `/healthz`, clear) | Tests SSE via TestClient; buffering behavior differs from real browsers. |
| `test_renderer.py` | 70 | Matplotlib chart generation across themes | Checks PNG creation and dimensions; does not check visual chart beauty. |
| `test_profiler.py` | 67 | Column role inference and dataset fingerprinting | Relies on column names for identifier heuristic (`order_id`, `uuid`). |
| `test_providers.py` | 167 | Provider URL construction, headers, retry backoff | Mocks HTTP client; does not hit real provider endpoints. |
| `test_registry.py` | 111 | Model catalog, sorting, custom provider env mapping | Relies on static catalog definitions. |
| `test_evaluation.py` | 133 | Scoring logic (`_numeric_close`, `_ops_satisfied`) | Tests test harness itself, not agent performance. Needs `evaluation/data/`, which is gitignored — CI regenerates it. |

#### Local-Only Test Suites (Excluded from Git via `.gitignore`)
* `tests/test_invariants.py`: Stresses all 14 analytics operations under 30% missing data and asserts grounding invariants.
* `tests/test_byok.py`: Tests `X-Insight-Key` header propagation and `require_user_key` enforcement.
* `tests/test_chain_fallback.py`: Tests failover from empty/flaky providers to healthy providers.
* `tests/test_custom_provider.py`: Tests custom OpenAI-compatible endpoint registration and discovery toggle.

---

### Golden-Question Evaluation Harness (`evaluation/`)

* **Runner**: `evaluation/run_eval.py`
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
```

### Core Regression Commands
```bash
# Full offline test suite (215 unit & integration tests, tracing & query cache forced off)
pytest -q

# Run local invariants & robustness tests (if present locally)
pytest tests/test_invariants.py tests/test_byok.py tests/test_chain_fallback.py -q

# Run live golden-question evaluation harness (requires live GROQ_API_KEY in .env)
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b --limit 5
```

### Known Limitations & Deferred Capabilities
1. **Single-Worker Container Constraint**: In-memory `WebSession` state (uploaded CSV bytes, profile, conversation history) lives in Python process memory. The Docker container must run exactly **one** Uvicorn worker. Horizontal multi-worker scaling requires externalizing session state (e.g. to Redis).
2. **Windows cp1252 Unicode**: Windows terminal printing Unicode characters (e.g. `…`, `\u202f`) can crash scripts unless `sys.stdout.reconfigure(encoding="utf-8")` is invoked.
3. **PowerShell UTF-8 BOM**: PowerShell 5.1 `Set-Content` can inject a Byte Order Mark (BOM). All parsers must handle UTF-8-SIG or strip BOMs.
4. **Deferred Features (Do Not Implement Unprompted)**: Python code-interpreter sandboxes (omitted by design; DuckDB covers analytics safely), multi-user authentication, and native PDF exports.

### Documentation Discrepancies
* **Test Counts**: Environment-dependent by design. `README.md`, `HOW_TO_RUN.md`, `HANDOVER.md` and this file all state **215 offline tests**, matching `pytest -q` on a clone where `evaluation/data/` has been generated. The count is 216 passed / 1 skipped once the optional third-party `diamonds.csv` corpus is present, and 215 passed / 2 skipped without it. Do not treat a single absolute number as canonical — assert the pass/fail outcome instead.
* **CI was silently red**: Until 2026-09-30 the GitHub Actions workflow had never passed. `evaluation/data/` is gitignored, so `tests/test_evaluation.py` failed on any fresh clone with `missing dataset sales.csv`; `pyyaml` was imported by `tests/test_artifacts.py` but never declared in `requirements.txt` (it was only present transitively via Streamlit). CI now runs `python evaluation/generate_datasets.py` before pytest, and PyYAML is a declared dependency. **Lesson: a test suite that only passes on the author's machine is not a safety net.** If CI is red, treat it as a code bug, not a flaky environment.
* **Streamlit vs. WebApp**: `README.md` initially presented the Streamlit UI as primary. The primary production interface is now the FastAPI web application in `webapp/` (Uvicorn port 8000), with Streamlit maintained as a secondary interface.
* **GitHub repo metadata**: The repository description claimed "Built with Streamlit, LangChain, and Groq API". LangChain was never a dependency (zero imports; only `langsmith` for tracing) and the LLM layer is hand-rolled over `httpx`. Corrected 2026-09-30.
