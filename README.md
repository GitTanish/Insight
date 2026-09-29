# INSIGHT ðŸ“Š
### A validated, model-agnostic analytical agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev)
[![Tests](https://img.shields.io/badge/tests-172%20passing-brightgreen.svg)](#5-run-the-test-suite-optional-but-encouraged)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Upload a CSV â€” or drop four related ones. Ask anything in plain English. Get verified answers, newspaper-styled charts, a formatted Word report, and production-ready dbt/alert exports.

**[â†’ Try the live demo](https://the-insight-ai.streamlit.app/)**

---

## Proof, not promises

Measured on this repo during development â€” not aspirational numbers:

| Anchor metric | What it proves |
|---|---|
| **20/20** golden-question eval cases passed live (Groq Â· Mistral Â· opencode) | the no-invented-numbers policy holds end-to-end |
| **250K rows**: profiled in **2.0s**, heaviest op **<2s**, bivariate OLS in **125ms** | deterministic engine at real scale, not toy data |
| **âˆ’31.7% Brier score** after isotonic recalibration; over-confidence flagged at **ECE = 0.245** | calibration audits quantify, not guess |
| **â‰¤0.13% aggregate drift** under 5% MCAR missingness â†’ graceful **0.60% @ 35%**, with explicit MCAR caveats | honest missing-data behavior, surfaced not hidden |
| Engine answered diamonds' top cut as **Premium ($4,584 avg)** where LLM priors say *Ideal* | computed evidence beats model priors |
| **8/8 malicious SQL patterns blocked** (`DROP`, `COPY TO`, `ATTACH`, `PRAGMA`, chained statementsâ€¦) + hard **10s interrupt** on runaway queries | read-only relational analytics you can hand to an LLM |
| **16-question real-world battery** (54K-row diamonds + a messy 4-sheet clinical workbook with case-mismatched join keys) logged in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md), each answer cross-checked against independent pandas/scipy ground truth | survives data the way it actually arrives |
| **172 offline tests**, CI-green with zero API keys, every headline number independently recomputed through a second arithmetic path â€” and every number in the *narrative* traced back to computed evidence | verified by construction |

---

## How it works

Insight v2 is built on one principle: **the LLM is not the analytics engine.**

```
Upload CSV
    â†“
Dataset Profiler â”€â”€â–º types, missingness, cardinality â†’ dataset fingerprint
    â†“
Zero-Prompt Briefing â”€â”€â–º top-3 findings immediately, no question needed
    â†“                    (outliers Â· trends Â· correlations Â· imbalance Â· quality)
Planner (LLM) â”€â”€â”€â”€â–º JSON AnalysisPlan â”€â”€â–º schema + column validated
    â†“                                   â””â”€ invalid? bounded repair loop
Deterministic Engine â”€â”€â–º 15 operations (14 fixed pandas/scipy ops + embedded
    â†“                     DuckDB SQL) chained via input_step DAG
Statistical Router â”€â”€â–º Welch t-test / Mann-Whitney U / chi-square + CramÃ©r's V
    â†“                    / one-way ANOVA + Î·Â² â€” auto-routed by column types
Result Validator â”€â”€â–º non-finite values, empty results, small samples,
    â†“                 missingness caveats (MCAR assumption) + INDEPENDENT
    â†“                 RECOMPUTATION of every headline number through a
    â†“                 second arithmetic path
Conversation State â”€â”€â–º active filters/dims/metrics persist across turns;
    â†“                    follow-ups become filter patches, not re-interpretation
Explainer (LLM) â”€â”€â–º narrative citing only computed values
    â†“
Answer + Evidence + Charts â”€â”€â–º streamed live (SSE) Â· exportable as .docx
```

Every claim in an answer traces back to a calculation you can inspect via **View calculation**, and every plan step is auditable via **View plan**. If a plan fails validation or execution, the orchestrator repairs and retries â€” once â€” then tells you exactly what went wrong instead of hallucinating.

---

## Features

- **Zero-prompt briefing** â€” the moment a CSV lands, deterministic detectors surface the three most important findings (trend breaks, outliers, strong correlations, class imbalance, data quality), each with an *Investigate* button.
- **Verified answers** â€” a validator checks sanity and small-sample caveats; an independent recomputation layer re-derives aggregates through a second code path and flags any mismatch before you see the answer.
- **Statistical tests without statistics mistakes** â€” binary-vs-numeric questions route to Welch's t-test or Mannâ€“Whitney U with Cohen's d and plain-language group differences ("Class 1 averages 32% lower"); categorical pairs get chi-square with CramÃ©r's V; 3+ groups vs numeric route to one-way ANOVA with Î·Â². Point-biserial correlation traps are actively blocked.
- **Regression & calibration** â€” OLS via `np.linalg.lstsq` with t-stat p-values, RÂ² and standardized betas; `isotonic_calibration` audits whether a score truly tracks a binary outcome (PAV fit, reliability table, Brier before/after recalibration, ECE).
- **Honest missing-data handling** â€” every op coerces/drops explicitly; listwise deletions are reported (`rows_excluded_incomplete`, mixed count bases flagged), and referenced columns >5% missing raise an explicit MCAR-assumption caveat in the answer.
- **Conversation state machine** â€” filters applied in earlier turns persist as structured state; "now only 2025" becomes a filter patch instead of the model re-guessing context.
- **Relational analytics on embedded DuckDB** â€” drop 2â€“6 related CSVs, or a multi-sheet Excel workbook where **every sheet becomes its own table**. Ask for JOINs, window functions, per-group rankings and pivots via the sandboxed `sql_query` op: in-memory only, external access disabled, config locked, SELECT/WITH-only with keyword guards, interrupt-based timeout, hard row caps. Runtime errors flow straight into the existing plan-repair loop â€” self-correction without a code interpreter.
- **Actionable artifact exports** â€” any SQL answer ships two buttons: **Export dbt model** (a `.sql` model + `schema.yml` with column tests, zipped) and **Alert monitor** (a Slack-ready webhook monitor descriptor with the query, cron schedule and payload template).
- **Chained "why" analysis** â€” plans are DAGs: `filter_rows â†’ group_aggregate â†’ statistical_test` compose via explicit `input_step` references.
- **Editorial charts that never look broken** â€” adaptive Freedmanâ€“Diaconis/Sturges binning, sparse-bin merging, cardinality-aware titles, automatic "Other" bucketing for long tails, horizontal-bar fallback for long labels, human tick formatting (12.5K / 3.2M).
- **Model-agnostic plug-and-play** â€” Groq, Mistral, OpenAI, Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint via env vars. Automatic cross-provider fallback on rate limits, 5xx â€” *and empty responses*, which reasoning endpoints occasionally emit mid-plan.
- **Live pipeline streaming** â€” watch *Planning â†’ Executing â†’ Validating â†’ Explaining* happen in real time over SSE.
- **Headless JSON API** â€” `POST /api/query` exposes the same verified pipeline for scripts and integrations.
- **Query-result cache** â€” identical (dataset hash, question, model, temperature, session state) hits are served from `.artifacts/cache/` with TTL + size-budget eviction (`INSIGHT_QUERY_CACHE=off` to disable).
- **Golden-question eval harness** â€” 20 seeded/real benchmark cases (incl. the 54K-row diamonds dataset and a 250K-row synthetic retail set) with exact ground truth; `python evaluation/run_eval.py --model <id>` produces a per-case PASS/FAIL markdown report.
- **Observability** â€” optional LangSmith tracing of every plan, step, and LLM call.

---

## Screenshots

| Upload & Configure | Ask Questions |
|---|---|
| ![Upload](image/1.png) | ![Query](image/2.png) |

| Generated Charts | Exported Report |
|---|---|
| ![Charts](image/4.png) | ![Report](image/download%20report%20demonstration.png) |

---

## Quickstart

**Prerequisites:** Python 3.10+ (3.11 recommended) and a free [Groq API key](https://console.groq.com).

### 1. Clone and set up a virtual environment

```bash
git clone https://github.com/GitTanish/Insight.git
cd Insight

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API key (or keys)

Copy the template and fill in whichever providers you use:

```bash
cp .env.example .env
```

Minimum viable setup:

```bash
# Required â€” pick at least one provider
GROQ_API_KEY=gsk_your_key_here

# Optional â€” any of these unlock additional models + cross-provider fallback
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

Or export directly:

```bash
# Windows
set GROQ_API_KEY=gsk_your_key_here
# macOS / Linux
export GROQ_API_KEY=gsk_your_key_here
```

### 4. Run

**Primary â€” FastAPI web server:**

```bash
uvicorn webapp.app:app --port 8000
```

Open `http://localhost:8000`. You get the full editorial UI plus live pipeline streaming (Planning â†’ Executing â†’ Validating â†’ Explaining) and a JSON API at `POST /api/query`.

**Alternative â€” classic Streamlit UI:**

```bash
streamlit run main.py
```

Both front-ends share the same `insight/` engine.

### 5. Run the test suite (optional but encouraged)

```bash
pytest
```

85 unit/integration tests cover the profiler, every analytics operation (including statistical routing), the DAG executor, the validator + independent recomputation layer, planner repair loops, chart rendering rules, the briefing engine, conversation state, the query cache, and the FastAPI webapp end-to-end â€” all via a scripted fake LLM, no API key needed. A separate `eval`-marked live suite (`INSIGHT_RUN_EVAL=1 pytest -m eval`) exercises the golden-question harness against a real provider.

### Evaluation harness (optional)

Benchmark the full pipeline against deterministic datasets with known ground truth:

```bash
python evaluation/generate_datasets.py          # writes sales/ecommerce/support/retail_250k CSVs
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b
# â†’ per-case PASS/FAIL + markdown report under .artifacts/eval/
```

Real-world runs are recorded in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md):
the question battery, ops executed per answer, timings, grounding flags, plus an
independent ground-truth verification section (including the data-quality traps the
engine had to navigate â€” case-mismatched join keys, inconsistent state codes).

---

## Providers (plug and play)

Insight is **model-agnostic**: every provider sits behind one `LLMProvider` interface, and the app auto-detects whichever you configure. Set any subset of these in `.env` â€” missing keys simply hide that provider.

| Provider | Key | Models |
|---|---|---|
| **Groq** *(default)* | `GROQ_API_KEY` | gpt-oss-120b / gpt-oss-20b, qwen3.6-27b |
| **Mistral** | `MISTRAL_API_KEY` | mistral-small / medium-latest |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o family + live discovery |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude models + live discovery |
| **OpenRouter** | `OPENROUTER_API_KEY` | 400+ aggregated models, curated discovery |
| **Ollama** (local) | `OLLAMA_BASE_URL` or `INSIGHT_ENABLE_OLLAMA=1` | whatever you have pulled |
| **Any OpenAI-compatible service** | `INSIGHT_CUSTOM_BASE_URL` + `INSIGHT_CUSTOM_API_KEY` (+ `INSIGHT_CUSTOM_MODELS`) | DeepSeek, Together, Fireworks, vLLM, LM Studioâ€¦ |

See [.env.example](.env.example) for the full template. Model capabilities (JSON mode, reasoning effort) are declared per model; the orchestrator strips unsupported parameters automatically and falls back across providers when a 429/5xx hits. Copy `.env.example` to `.env` and fill in what you have:

```bash
cp .env.example .env
```

---

## Configuration

File limits: up to 1,000,000 rows Â· CSV (UTF-8/Latin-1/CP1252, delimiters `,` `;` `\t`) Â· Excel `.xlsx`/`.xls`/`.xlsm` with one table registered per non-empty sheet.

| Variable | Default | Purpose |
|---|---|---|
| `INSIGHT_SQL` | `on` | enable the embedded DuckDB `sql_query` op |
| `INSIGHT_SQL_TIMEOUT_S` | `10` | hard interrupt for runaway queries |
| `INSIGHT_SQL_MAX_ROWS` | `10000` | result-set cap before display truncation |
| `INSIGHT_QUERY_CACHE` | `on` | cache identical (dataset, question, model, temp, state) results under `.artifacts/cache/` |
| `INSIGHT_QUERY_CACHE_TTL_DAYS` | `14` | cache entries older than this are evicted |
| `INSIGHT_QUERY_CACHE_MAX_MB` | `512` | size budget; oldest entries evicted first |
| `INSIGHT_BRIEFING_LLM_POLISH` | `off` | one extra fast LLM call to sharpen briefing headlines (numbers frozen) |

**Deployment note:** the webapp is single-worker by design â€” sessions (CSV bytes, profile, turns) live in process memory. Run exactly one uvicorn worker; horizontal scaling requires externalizing session state first. Put a TLS-terminating reverse proxy in front before exposing beyond localhost; there is no built-in auth or rate limiting.

---

## Project structure

```
webapp/                  FastAPI UI (primary)
â”œâ”€â”€ app.py               routes: upload / SSE query stream / export / JSON API / models refresh
â”œâ”€â”€ sessions.py          cookie-keyed server-side sessions (single-worker)
â”œâ”€â”€ services.py          Streamlit-free CSV/profile helpers
â”œâ”€â”€ templates/           Jinja2 (base, index, partials)
â””â”€â”€ static/              editorial CSS + SSE client glue
main.py                  Streamlit entry point (legacy UI, same engine)
â”œâ”€â”€ ui_components.py     Streamlit rendering layer
â”œâ”€â”€ utils.py             cached wrappers (delegates DOCX to insight/reports)
â”œâ”€â”€ style.css            editorial styling
â”œâ”€â”€ insight/             â† the engine (UI-agnostic)
â”‚   â”œâ”€â”€ domain/          Pydantic v2 contracts: profiles, plans, results, errors
â”‚   â”œâ”€â”€ llm/             Provider ABC, OpenAI-compatible base, Groq/Mistral/OpenRouter/..., registry
â”‚   â”œâ”€â”€ profiling/       Dataset profiler + fingerprinting
â”‚   â”œâ”€â”€ planning/        Planner (structured output + repair), prompt builders
â”‚   â”œâ”€â”€ analytics/       Deterministic operation library (15 ops incl. DuckDB sql_query)
â”‚   â”œâ”€â”€ briefing.py      zero-prompt top-findings engine (+ optional LLM polish flag)
â”‚   â”œâ”€â”€ conversation/    AnalysisSessionState â€” filters/dims/metrics across turns
â”‚   â”œâ”€â”€ execution/       Plan executor (sync core, async wrapper, input_step DAG)
â”‚   â”œâ”€â”€ validation/      Result validator + independent recomputation checks
â”‚   â”œâ”€â”€ cache.py         query-result cache (TTL + size-budget eviction)
â”‚   â”œâ”€â”€ reports/         shared DOCX builder + dbt/alert artifact generators
â”‚   â”œâ”€â”€ visualization/   Editorial theme + chart renderer from ChartSpec
â”‚   â””â”€â”€ orchestrator.py  plan â†’ execute â†’ validate â†’ repair â†’ render â†’ explain (+ streaming variant)
â”œâ”€â”€ evaluation/          golden-question harness: generator, 20 cases, scorer/reporter
â””â”€â”€ tests/               pytest suite (fake LLM, zero network needed)
```

The `insight/` package is deliberately UI-agnostic â€” both front-ends call it through one function. The FastAPI app additionally exposes the pipeline as an SSE stream and a JSON API, so CLIs and future clients reuse the exact same verified engine.

---

## Example queries

```
"Which region has the highest revenue?"
"Why did revenue fall between Q2 and Q3?"            â† chained DAG plan
"Join orders to customers and rank each customer's biggest order"  â† DuckDB window fn
"Is the conversion difference between groups real?"  â† auto t-test / Mann-Whitney
"Do handle times differ across priority levels?"     â† auto one-way ANOVA + Î·Â²
"How strongly do price and carat weight correlate?"  â† Pearson/Spearman
"Audit whether lead_score predicts conversion well"  â† isotonic calibration + ECE
"Are treatment and churn associated?"                â† chi-square + CramÃ©r's V
"Create 2-3 visualizations of this data"
"Are there outliers in the price column?"
"What correlates most strongly with quantity?"
"Compare average order value between 2024 and 2025"
```

Every response includes:
- **Answer** â€” narrative citing only computed numbers, with significance stated where tests ran
- **Charts** â€” consistent ink-on-paper editorial style
- **View calculation** â€” the exact result tables behind the claims
- **View plan** â€” the validated step DAG with parameters and reasons
- **Warnings** â€” consolidated caveats (small samples, sparse bins, excluded binary correlations)

Export everything â€” including the upload briefing â€” as a formatted Word report.

---

## Methods & references

Every statistical claim above is a textbook method executed by deterministic code (`insight/analytics/`). The papers behind them:

**Classical inference** â€” *Welch's t-test* (Welch, 1947, *Biometrika*), *Mannâ€“Whitney U* (Mann & Whitney, 1947), *chi-square independence* (Pearson, 1900), *CramÃ©r's V* with bias correction (CramÃ©r, 1946), *one-way ANOVA F-test* (Fisher, 1925), t-distribution p-values (Student, 1908). Effect sizes follow Cohen's conventions â€” Cohen's d (Cohen, 1988) and Î·Â² benchmarks (Cohen, 1973).

**Regression** â€” OLS via least squares (Legendre, 1805; Gauss), coefficient standard errors from ÏƒÂ²(Xáµ€X)â»Â¹, standardized betas and adjusted RÂ² per Montgomery, Peck & Vining, *Introduction to Linear Regression Analysis*.

**Distribution & outliers** â€” adaptive binning blends Sturges' rule (Sturges, 1926, *JASA*) with the Freedmanâ€“Diaconis rule (Freedman & Diaconis, 1981); outlier fences use Tukey's 1.5Ã—IQR rule (*Exploratory Data Analysis*, 1977). Rank-based association via Spearman (1904).

**Probability calibration** â€” `isotonic_calibration` implements the pool-adjacent-violators algorithm (Ayer et al., 1955; Robertson, Wright & Dykstra, *Order Restricted Statistical Inference*, 1988), scored by the Brier score (Brier, 1950) and expected calibration error in the style of Guo et al. (2017), following the score-to-probability calibration literature (Zadrozny & Elkan, 2002; Niculescu-Mizil & Caruana, 2005).

**LLM reliability patterns** â€” the planner-emits-plan / engine-computes split follows program-aided language models (Gao et al., *PAL*, ICML 2023); bounded plan-repair mirrors iterative self-refinement (Madaan et al., *Self-Refine*, NeurIPS 2023); independent recomputation of every headline number is an application of chain-of-verification ideas (Dhuliawala et al., 2023).

## Security note

Your CSV contents are sent to the configured LLM provider as context. The LLM **never executes code**: all computation happens through a fixed library of pandas operations inside your own process. Don't upload datasets containing sensitive personal information you wouldn't paste into a chat model.

---

## Troubleshooting

**`No provider available`** â€” no `GROQ_API_KEY`/`MISTRAL_API_KEY` found in `.env` or environment.

**`unknown column 'x'` in the answer** â€” the model referenced a column that doesn't exist; the repair loop already retried. Rephrase using exact column names from the profile expander.

**Rate limit errors** â€” Groq free tier allows 30 requests/min on gpt-oss models. Wait a moment, switch to `gpt-oss-20b`, or add a Mistral key for fallback.

**Charts not appearing** â€” check the `.artifacts/web/<session>/` folder is writable; render warnings appear under answers.

**Slow first load** â€” provider model catalogs are scanned on startup (~10s, cached); the dataset profile and briefing compute once per upload.

**Streamlit vs webapp sessions** â€” each front-end keeps its own session state; uploads don't transfer between them.

---

## Observability

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) to enable full tracing via [LangSmith](https://smith.langchain.com). Every question produces a root `insight.analyze` run containing:

- the validated **plan** JSON (with repair attempts visible as failedâ†’retried children),
- one **LLM span per provider call** (model, tokens, latency, retries),
- step-level execution results and validation outcomes.

Tracing is fully optional â€” with no key the engine runs identically and posts nothing.

---

## Roadmap

Completed: zero-prompt briefing âœ… Â· statistical engine (t-test / Mann-Whitney / chi-square / Cramer's V) âœ… Â· chained DAG plans âœ… Â· independent result verification âœ… Â· FastAPI + SSE front-end âœ… Â· LangSmith observability âœ… Â· statistics depth (ANOVA + eta-squared, OLS regression, confidence intervals) âœ… Â· isotonic calibration audits âœ… Â· conversation state machine âœ… Â· golden-question evaluation harness âœ… Â· Dockerfile + CI + query-result cache with eviction âœ… Â· embedded DuckDB relational analytics + dbt/alert artifact exports âœ…

**Up next** (detailed notes in [HANDOVER.md](HANDOVER.md)):

- External session storage (Redis) for multi-worker deployments
- MAR/MNAR-aware missingness diagnostics (currently MCAR assumption is surfaced as a caveat)
- Streaming cache for the SSE path; PDF export remains deferred

---

## Contributing

```bash
git checkout -b feature/your-feature
git commit -m 'add: your feature'
git push origin feature/your-feature
# open a pull request
```

Run `pytest` before submitting â€” CI-friendly, no keys required.

---

## Author

**Tanish Saroj** Â· [github.com/GitTanish](https://github.com/GitTanish) Â· [linkedin.com/in/tanishsaroj](https://linkedin.com/in/tanishsaroj)

---

*Built with [Groq](https://groq.com) Â· [Pydantic](https://docs.pydantic.dev) Â· [FastAPI](https://fastapi.tiangolo.com) Â· pandas Â· scipy*
