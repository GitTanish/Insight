# INSIGHT 📊
### A validated, model-agnostic analytical agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev)
[![Tests](https://img.shields.io/badge/tests-CI%20green-brightgreen.svg)](#5-run-the-test-suite-optional-but-encouraged)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Upload a CSV — or drop four related ones. Ask anything in plain English. Get verified answers, newspaper-styled charts, a formatted Word report, and production-ready dbt/alert exports.

**[→ Try the live demo](https://the-insight-ai.streamlit.app/)**

---

## Proof, not promises

Measured on this repo during development — not aspirational numbers:

| Anchor metric | What it proves |
|---|---|
| **20/20** golden-question eval cases passed live (Groq · Mistral · opencode) | the no-invented-numbers policy holds end-to-end |
| **Agentic vs single-shot, pinned live A/B** (k=3, production budget, one model, no fallback): measured **58/60 (96.7%)** vs **43/60 (71.7%)** on the same 20 golden cases — follow-up diagnosis showed the gap was mostly a provider truncation bug that only hit the single-shot planner; post-fix k=1 confirmations: **20/20 vs 19/20 (parity)** | the tool loop matches the single-shot planner on accuracy at comparable call counts (2.17 vs 2.36) while adding an auditable tool trail, all backed by mutation-sampled tests (`loop.py` at 96% branch coverage) |
| **250K rows**: profiled in **2.0s**, heaviest op **<2s**, bivariate OLS in **125ms** | deterministic engine at real scale, not toy data |
| **−31.7% Brier score** after isotonic recalibration; over-confidence flagged at **ECE = 0.245** | calibration audits quantify, not guess |
| **≤0.13% aggregate drift** under 5% MCAR missingness → graceful **0.60% @ 35%**, with explicit MCAR caveats | honest missing-data behavior, surfaced not hidden |
| Engine answered diamonds' top cut as **Premium ($4,584 avg)** where LLM priors say *Ideal* | computed evidence beats model priors |
| **8/8 malicious SQL patterns blocked** (`DROP`, `COPY TO`, `ATTACH`, `PRAGMA`, chained statements…) + hard **10s interrupt** on runaway queries | read-only relational analytics you can hand to an LLM |
| **16-question real-world battery** (54K-row diamonds + a messy 4-sheet clinical workbook with case-mismatched join keys) logged in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md), each answer cross-checked against independent pandas/scipy ground truth | survives data the way it actually arrives |
| **Full offline suite green in CI** with zero API keys, every headline number independently recomputed through a second arithmetic path — and every number in the *narrative* traced back to computed evidence | verified by construction |

---

## How it works

Insight v2 is built on one principle: **the LLM is not the analytics engine.**

```
Upload CSV
    ↓
Dataset Profiler ──► types, missingness, cardinality → dataset fingerprint
    ↓
Zero-Prompt Briefing ──► top-3 findings immediately, no question needed
    ↓                    (outliers · trends · correlations · imbalance · quality)
Planner (LLM) ────► JSON AnalysisPlan ──► schema + column validated
    ↓                                   └─ invalid? bounded repair loop
Deterministic Engine ──► 15 operations (14 fixed pandas/scipy ops + embedded
    ↓                     DuckDB SQL) chained via input_step DAG
Statistical Router ──► Welch t-test / Mann-Whitney U / chi-square + Cramér's V
    ↓                    / one-way ANOVA + η² — auto-routed by column types
Result Validator ──► non-finite values, empty results, small samples,
    ↓                 missingness caveats (MCAR assumption) + INDEPENDENT
    ↓                 RECOMPUTATION of every headline number through a
    ↓                 second arithmetic path
Conversation State ──► active filters/dims/metrics persist across turns;
    ↓                    follow-ups become filter patches, not re-interpretation
Explainer (LLM) ──► narrative citing only computed values
    ↓
Answer + Evidence + Charts ──► streamed live (SSE) · exportable as .docx
```

Every claim in an answer traces back to a calculation you can inspect via **View calculation**, and every plan step is auditable via **View plan**. If a plan fails validation or execution, the orchestrator repairs and retries — once — then tells you exactly what went wrong instead of hallucinating.

With `INSIGHT_PLAN_APPROVAL=on` the pipeline inserts a human gate between the planner and the engine: you see the exact operations, can edit the JSON, and only then approve. Nothing touches your data until you do.

In **agent mode** steps 3–5 are replaced by a bounded loop: the model calls the
engine's ops *as tools*, reads each real result, and decides what to do next —
then its findings are rebuilt into a plan so every box below (validation,
recomputation, charts, narration, grounding) runs unchanged. See
[Agent mode](#agent-mode-opt-in).

---

## Agent mode (opt-in)

The default planner asks the model for one complete `AnalysisPlan` up front. **Agent mode** (`INSIGHT_AGENTIC=on`, or the per-question *Agent mode* toggle next to the composer) lets the model instead call the engine's operations **as tools, one at a time**:

```
question → [call op → read real computed result] ×n → final answer
                │                                    │
                └─ live tool trail over SSE          └─ rebuilt into a plan →
                                                          validator · independent
                                                          recompute · charts ·
                                                          explainer · grounding
```

- **It observes before it decides.** A failed call (unknown column, bad params) comes back as a readable error, so the model fixes it instead of the whole plan dying on one mistake. It can also chain: pass `input_step` to run on an earlier step's output table, or let `filter_rows` narrow the working row-set for later calls.
- **It is bounded.** `INSIGHT_AGENTIC_MAX_ITERATIONS` / `INSIGHT_AGENTIC_MAX_LLM_CALLS` cap turns and provider calls; three identical failing calls abort the run; rate limits surface as a visible "throttled" outcome, never a wrong answer.
- **The guarantees do not move.** Every tool call is validated into the op's Pydantic params and executed by the same `OperationSpec.run` the executor uses — no free-form code, ever. Afterwards the run is *rebuilt* into an `AnalysisPlan` + `ExecutionResult`, so validation, the independent recomputation of every headline number, chart rendering, the explainer and the number-grounding guard are the exact same code the single-shot path uses. What changes is how the evidence is gathered, not how it is verified.
- **Chart planning is deterministic in this mode:** the two most recent table-producing steps are rendered with the same editorial rules, and any bad chart spec is rejected by the ordinary validator rather than trusted.
- Provider-agnostic: OpenAI-shaped tools on every OpenAI-compatible endpoint, Anthropic-shaped tools when the chain's primary provider is Anthropic.
- `INSIGHT_PLAN_APPROVAL=on` (or `plan_only`) takes precedence: that gate needs a reviewable plan before anything executes, so those requests use the single-shot planner.
- Benchmark it against the single-shot planner with the same golden cases and grader: `python evaluation/run_eval_agentic.py --model <id> --no-fallback --record`, then aggregate k repetitions per arm with `evaluation/ab_stats.py`. **Measured so far** (k=3, NVIDIA NIM `gpt-oss-20b`, production budget, no fallback): 58/60 (96.7%, 95% CI 88.6–99.1) vs 43/60 (71.7%, 59.2–81.5) for single-shot, at 2.17 vs 2.36 mean LLM calls — one endpoint, and the 2026-10-17 diagnosis showed most of that gap was a NIM planner-truncation bug that only hit the single-shot arm (fixed same day); post-fix k=1 confirmations score 20/20 vs 19/20, i.e. **accuracy parity on this endpoint** (the agentic edge is the auditable tool trail). The harness paid for itself on its first live runs: it exposed an OpenAI wire-format bug in the assistant tool-call history that no scripted fake could see (every second turn was a hard HTTP 400; fixed and pinned by tests), and later proved that env-var provider pinning is unreliable on Windows — hence `--no-fallback`.

---

## Alert runner

A SQL answer exports an **Alert monitor** (query + cron hint + threshold + Slack-ready webhook template). `tools/alert_runner.py` is the companion worker that closes the loop:

```bash
python -m tools.alert_runner --data data --monitors .artifacts/monitors --interval 300
python -m tools.alert_runner --once --dry-run     # render payloads, post nothing
```

It registers the CSVs as DuckDB tables, runs each monitor's query through the *same sandboxed session* the webapp uses (SELECT/WITH-only, keyword guards, interrupt watchdog, row cap), applies the threshold, and posts to the webhook when the condition **changes state** (recovery notifications via `--notify-recovery`, `--repeat` to post every cycle instead). Stdlib-only; the URL comes from the monitor's `webhook.url_env` (default `INSIGHT_ALERT_WEBHOOK_URL`) or `--webhook`.

---

## Features

- **Zero-prompt briefing** — the moment a CSV lands, deterministic detectors surface the three most important findings (trend breaks, outliers, strong correlations, class imbalance, data quality), each with an *Investigate* button.
- **Agentic tool-use mode (opt-in)** — flip on **Agent mode** and the model stops writing one big plan: it calls analysis ops *as tools, one at a time*, reads each real computed result, and decides what to do next (retry a bad call, chain from an earlier step's table, or conclude). It is bounded (iteration + call budgets, throttle-aware, identical-failure abort), DAG-chained via `input_step`, and — crucially — it runs on the *same* guarantees: every call routes through the deterministic op registry and column validator, and the answer is rebuilt into a plan so the validator, independent recomputation, charts, explainer and grounding guard all still apply. The tool trail streams live into the UI.
- **Verified answers** — a validator checks sanity and small-sample caveats; an independent recomputation layer re-derives aggregates through a second code path and flags any mismatch before you see the answer.
- **Statistical tests without statistics mistakes** — binary-vs-numeric questions route to Welch's t-test or Mann–Whitney U with Cohen's d and plain-language group differences ("Class 1 averages 32% lower"); categorical pairs get chi-square with Cramér's V; 3+ groups vs numeric route to one-way ANOVA with η². Point-biserial correlation traps are actively blocked.
- **Regression & calibration** — OLS via `np.linalg.lstsq` with t-stat p-values, R² and standardized betas; `isotonic_calibration` audits whether a score truly tracks a binary outcome (PAV fit, reliability table, Brier before/after recalibration, ECE).
- **Honest missing-data handling** — every op coerces/drops explicitly; listwise deletions are reported (`rows_excluded_incomplete`, mixed count bases flagged), and referenced columns >5% missing raise an explicit MCAR-assumption caveat in the answer.
- **Conversation state machine** — filters applied in earlier turns persist as structured state; "now only 2025" becomes a filter patch instead of the model re-guessing context.
- **Relational analytics on embedded DuckDB** — drop 2–6 related CSVs, or a multi-sheet Excel workbook where **every sheet becomes its own table**. Ask for JOINs, window functions, per-group rankings and pivots via the sandboxed `sql_query` op: in-memory only, external access disabled, config locked, SELECT/WITH-only with keyword guards, interrupt-based timeout, hard row caps. Runtime errors flow straight into the existing plan-repair loop — self-correction without a code interpreter.
- **Actionable artifact exports** — any SQL answer ships two buttons: **Export dbt model** (a `.sql` model + `schema.yml` with column tests, zipped) and **Alert monitor** (a Slack-ready webhook monitor descriptor with the query, cron schedule and payload template).
- **Chained "why" analysis** — plans are DAGs: `filter_rows → group_aggregate → statistical_test` compose via explicit `input_step` references.
- **Editorial charts that never look broken** — adaptive Freedman–Diaconis/Sturges binning, sparse-bin merging, cardinality-aware titles, automatic "Other" bucketing for long tails, horizontal-bar fallback for long labels, human tick formatting (12.5K / 3.2M).
- **Model-agnostic plug-and-play** — Groq, Mistral, OpenAI, Anthropic, OpenRouter, Ollama, or any OpenAI-compatible endpoint via env vars. Automatic cross-provider fallback on rate limits, 5xx — *and empty responses* (the classic "empty content" case on reasoning models was diagnosed on 2026-10-17: a missing `reasoning_effort` capability flag let the model consume the whole token budget reasoning, so the error now names the suspicion — `hit max_tokens=N`).
- **Token-by-token answer streaming** — the explainer streams over SSE into a live typing bubble, so the first words land in ~1s instead of after a multi-second blank wait. Time-to-first-token is measured and shown in the timing chip.
- **Human-in-the-loop plan approval** — with `INSIGHT_PLAN_APPROVAL=on`, every question stops at an editable plan (the exact operations, as JSON) that you approve before anything runs. Approved plans are *re-validated against your schema*, so a hand-edit naming a nonexistent column is rejected rather than silently trusted.
- **Live pipeline streaming** — watch *Planning → Executing → Validating → Explaining* happen in real time over SSE.
- **Headless JSON API** — `POST /api/query` exposes the same verified pipeline for scripts and integrations. Send `plan_only: true` to get a reviewable plan back instead of an answer, or `agentic: true` for one tool-calling run.
- **Query-result cache** — identical (dataset hash, question, model, temperature, session state) hits are served from `.artifacts/cache/` with TTL + size-budget eviction (`INSIGHT_QUERY_CACHE=off` to disable). Serves the streaming UI path too: ~20× faster on repeats.
- **Prefix-cached prompts** — the planner's system prompt stays byte-stable per dataset (per-turn state lives in later messages, not the prefix), so providers serve it from their prompt cache. Reported as `cached_prompt_tokens`.
- **Golden-question eval harness** — 20 seeded/real benchmark cases (incl. the 54K-row diamonds dataset and a 250K-row synthetic retail set) with exact ground truth; `evaluation/run_eval.py` scores the single-shot path, `run_eval_agentic.py` runs the same cases and grader through the agentic loop (token/wall/model telemetry, optional raw transcripts via `--record`, `--no-fallback` for provable single-model runs), and `ab_stats.py` aggregates k repetitions into per-case pass counts and Wilson CIs. Pinned k=3 A/B (NIM `gpt-oss-20b`): agentic 96.7% vs single-shot 71.7% — but the 2026-10-17 follow-up showed most of that gap was a provider-side truncation bug that hit the single-shot planner (13 of 17 baseline failures); post-fix pinned k=1 confirmations score **20/20 (agentic) vs 19/20 (single-shot): accuracy parity on this endpoint — the agentic edge is auditability and the tool trail, not accuracy**. The agentic loop additionally carries sampled mutation tests (11/11 killed) and 96% branch coverage.
- **Observability** — optional LangSmith tracing of every plan, step, and LLM call.

---

## Screenshots

| Upload & Configure | Ask Questions |
|---|---|
| ![Upload](image/1.png) | ![Query](image/2.png) |

| Generated Charts | Exported Report |
|---|---|
| ![Charts](image/4.png) | ![Report](image/download-report-demo.png) |

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
# Required — pick at least one provider
GROQ_API_KEY=gsk_your_key_here

# Optional — any of these unlock additional models + cross-provider fallback
MISTRAL_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
NVIDIA_API_KEY=your_key_here        # free NIM credits at build.nvidia.com
```

Or export directly:

```bash
# Windows
set GROQ_API_KEY=gsk_your_key_here
# macOS / Linux
export GROQ_API_KEY=gsk_your_key_here
```

### 4. Run

**Primary — FastAPI web server:**

```bash
uvicorn webapp.app:app --port 8000
```

Open `http://localhost:8000`. You get the full editorial UI plus live pipeline streaming (Planning → Executing → Validating → Explaining) and a JSON API at `POST /api/query`.

**Alternative — classic Streamlit UI:**

```bash
streamlit run main.py
```

Both front-ends share the same `insight/` engine.

**Alternative — Docker:**

```bash
docker build -t insight .
docker run -p 8000:8000 --env-file .env insight
```

No local Python or virtualenv needed. The image runs unprivileged as a non-root user, binds to `0.0.0.0:8000`, and ships a `/healthz` healthcheck. Keys stay outside the image — they're read from `--env-file` at runtime.

### 5. Run the test suite (optional but encouraged)

```bash
pytest
```

The offline unit/integration suite covers the profiler, every analytics operation (including statistical routing), the DAG executor, the validator + independent recomputation layer, planner repair loops, chart rendering rules, the briefing engine, conversation state, the query cache, text streaming, plan approval, prompt-cache stability, and the FastAPI webapp end-to-end — all via a scripted fake LLM, no API key needed. It runs green in CI on every push. CI generates the deterministic eval datasets first (`python evaluation/generate_datasets.py`); a handful of tests additionally exercise the optional third-party `diamonds.csv` corpus when it is present, so the exact pass/skip split varies by platform and by which corpora you have. A separate `eval`-marked live suite (`INSIGHT_RUN_EVAL=1 pytest -m eval`) exercises the golden-question harness against a real provider.

### Evaluation harness (optional)

Benchmark the full pipeline against deterministic datasets with known ground truth:

```bash
python evaluation/generate_datasets.py          # writes sales/ecommerce/support/retail_250k CSVs
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b
# → per-case PASS/FAIL + markdown report under .artifacts/eval/
```

The agentic loop has a matching runner over the **same cases and the same grader**:

```bash
python evaluation/run_eval_agentic.py --model groq/openai/gpt-oss-120b --record
# → same pass/fail, plus cost telemetry: llm_calls, prompt/completion tokens,
#   wall_s, whether the run was throttled, and which models actually served
#   the calls (fallback-chain use is visible in the data, never silent)
#   --record keeps every raw request/response as transcript.jsonl for replay
```

Aggregate repeated runs into publishable A/B numbers:

```bash
python evaluation/ab_stats.py --agentic DIR1,DIR2,DIR3 --baseline DIR4,DIR5,DIR6
# → per-case passes/k, overall pass rate with a 95% Wilson CI, mean llm_calls
#   and wall time, token totals, throttled counts, failure-reason rollup
```

Handy flags: `--limit N`, `--dataset sales.csv`, `--delay S` (spacing between provider calls to stay under free-tier TPM), `--max-iterations`/`--max-llm-calls` (eval default 3/6 for cheap smokes; the app ships 5/9), `--no-fallback` (pin the chain to the primary model — required for a provable single-model A/B; env-var pinning is unreliable on Windows). Results are written incrementally, so an interrupted or throttled run still yields analysable data.

Pinned, fallback-free A/B — the exact recipe behind the numbers above (needs `NVIDIA_API_KEY`):

```bash
python evaluation/run_eval_agentic.py --model nvidia/openai/gpt-oss-20b --no-fallback --delay 2 --record
python evaluation/run_eval.py        --model nvidia/openai/gpt-oss-20b --no-fallback
# → pre-fix k=3 on 20 cases: agentic 58/60 (96.7%, 95% CI 88.6–99.1)
#                             single-shot 43/60 (71.7%, 95% CI 59.2–81.5)
# (mostly a NIM planner-truncation artifact on the single-shot arm — fixed
#  2026-10-17; post-fix k=1: 20/20 agentic vs 19/20 single-shot → parity)
```

Cross-arm caveat: token totals are not comparable (the agentic runner sums every provider call; the single-shot runner records the meta usage block) — compare `llm_calls` and wall time instead.

Real-world runs are recorded in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md):
the question battery, ops executed per answer, timings, grounding flags, plus an
independent ground-truth verification section (including the data-quality traps the
engine had to navigate — case-mismatched join keys, inconsistent state codes).

---

## Providers (plug and play)

Insight is **model-agnostic**: every provider sits behind one `LLMProvider` interface, and the app auto-detects whichever you configure. Set any subset of these in `.env` — missing keys simply hide that provider.

| Provider | Key | Models |
|---|---|---|
| **Groq** *(default)* | `GROQ_API_KEY` | gpt-oss-120b / gpt-oss-20b, qwen3.8-27b |
| **Groq (2nd account)** *(optional)* | `GROQ_API_KEY2` | same models via a separate quota pool (`groq2/...` ids) — doubles fallback + rate-limit headroom |
| **Mistral** | `MISTRAL_API_KEY` | mistral-small / medium-latest |
| **OpenAI** | `OPENAI_API_KEY` | gpt-4o family + live discovery |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude models + live discovery |
| **OpenRouter** | `OPENROUTER_API_KEY` | 400+ aggregated models, curated discovery |
| **NVIDIA NIM** | `NVIDIA_API_KEY` | gpt-oss-20b + live discovery (40 req/min tool-calling endpoints) |
| **Ollama** (local) | `OLLAMA_BASE_URL` or `INSIGHT_ENABLE_OLLAMA=1` | whatever you have pulled |
| **Any OpenAI-compatible service** | `INSIGHT_CUSTOM_BASE_URL` + `INSIGHT_CUSTOM_API_KEY` (+ `INSIGHT_CUSTOM_MODELS`) | DeepSeek, Together, Fireworks, vLLM, LM Studio… |

See [.env.example](.env.example) for the full template. Model capabilities (JSON mode, reasoning effort) are declared per model; the orchestrator strips unsupported parameters automatically and falls back across providers when a 429/5xx hits. Copy `.env.example` to `.env` and fill in what you have:

```bash
cp .env.example .env
```

Example — NVIDIA NIM (40 req/min, tool calling included):

```bash
NVIDIA_API_KEY=nvapi-...
# then use it anywhere a model id is accepted:
# python evaluation/run_eval_agentic.py --model nvidia/openai/gpt-oss-20b
```

---

## Configuration

File limits: up to 1,000,000 rows · CSV (UTF-8/Latin-1/CP1252, delimiters `,` `;` `\t`) · Excel `.xlsx`/`.xls`/`.xlsm` with one table registered per non-empty sheet.

| Variable | Default | Purpose |
|---|---|---|
| `INSIGHT_SQL` | `on` | enable the embedded DuckDB `sql_query` op |
| `INSIGHT_SQL_TIMEOUT_S` | `10` | hard interrupt for runaway queries |
| `INSIGHT_SQL_MAX_ROWS` | `10000` | result-set cap before display truncation |
| `INSIGHT_QUERY_CACHE` | `on` | cache identical (dataset, question, model, temp, state) results under `.artifacts/cache/` |
| `INSIGHT_QUERY_CACHE_TTL_DAYS` | `14` | cache entries older than this are evicted |
| `INSIGHT_QUERY_CACHE_MAX_MB` | `512` | size budget; oldest entries evicted first |
| `INSIGHT_BRIEFING_LLM_POLISH` | `off` | one extra fast LLM call to sharpen briefing headlines (numbers frozen). Runs in the background — the upload response never waits on it. |
| `INSIGHT_PLAN_APPROVAL` | `off` | stop at an editable plan for review before executing (`on`) |
| `INSIGHT_AGENTIC` | `off` | default question mode: agentic tool-calling loop instead of the single-shot planner (per-question override in both UIs) |
| `INSIGHT_AGENTIC_MAX_ITERATIONS` | `5` | max model turns per agentic question |
| `INSIGHT_AGENTIC_MAX_LLM_CALLS` | `9` | hard cap on provider calls per agentic question |
| `INSIGHT_AGENTIC_DELAY_S` | `0` | seconds between agentic provider calls (throttle headroom on free tiers) |

**Deployment note:** the webapp is single-worker by design — sessions (CSV bytes, profile, turns) live in process memory. Run exactly one uvicorn worker; horizontal scaling requires externalizing session state first. Put a TLS-terminating reverse proxy in front before exposing beyond localhost; there is no built-in auth or rate limiting.

---

## Project structure

```
webapp/                  FastAPI UI (primary)
├── app.py               routes: upload / SSE query stream / export / JSON API / models refresh
├── sessions.py          cookie-keyed server-side sessions (single-worker)
├── services.py          Streamlit-free CSV/profile helpers
├── templates/           Jinja2 (base, index, partials)
└── static/              editorial CSS + SSE client glue
main.py                  Streamlit entry point (legacy UI, same engine)
├── ui_components.py     Streamlit rendering layer
├── utils.py             cached wrappers (delegates DOCX to insight/reports)
├── style.css            editorial styling
├── insight/             ← the engine (UI-agnostic)
│   ├── domain/          Pydantic v2 contracts: profiles, plans, results, errors
│   ├── llm/             Provider ABC, OpenAI-compatible base, Groq/Mistral/OpenRouter/..., registry
│   ├── profiling/       Dataset profiler + fingerprinting
│   ├── planning/        Planner (structured output + repair), prompt builders
│   ├── analytics/       Deterministic operation library (15 ops incl. DuckDB sql_query)
│   ├── tools/           Opt-in agentic tool-use layer: op→tool schemas, dispatch, bounded loop, plan rebuild
│   ├── briefing.py      zero-prompt top-findings engine (+ optional LLM polish flag)
│   ├── conversation/    AnalysisSessionState — filters/dims/metrics across turns
│   ├── execution/       Plan executor (sync core, async wrapper, input_step DAG)
│   ├── validation/      Result validator + independent recomputation checks
│   ├── cache.py         query-result cache (TTL + size-budget eviction)
│   ├── reports/         shared DOCX builder + dbt/alert artifact generators
│   ├── visualization/   Editorial theme + chart renderer from ChartSpec
│   └── orchestrator.py  plan → execute → validate → repair → render → explain (+ streaming variant)
├── evaluation/          golden-question harness: generator, 20 cases, single-shot + agentic runners, k-rep aggregator, scorer/reporter
├── tools/               companion workers (alert_runner: cron → DuckDB → webhook)
└── tests/               pytest suite (fake LLM + adversarial agentic suite, zero network needed)
```

The `insight/` package is deliberately UI-agnostic — both front-ends call it through one function. The FastAPI app additionally exposes the pipeline as an SSE stream and a JSON API, so CLIs and future clients reuse the exact same verified engine.

---

## Example queries

```
"Which region has the highest revenue?"
"Why did revenue fall between Q2 and Q3?"            ← chained DAG plan
"Join orders to customers and rank each customer's biggest order"  ← DuckDB window fn
"Is the conversion difference between groups real?"  ← auto t-test / Mann-Whitney
"Do handle times differ across priority levels?"     ← auto one-way ANOVA + η²
"How strongly do price and carat weight correlate?"  ← Pearson/Spearman
"Audit whether lead_score predicts conversion well"  ← isotonic calibration + ECE
"Are treatment and churn associated?"                ← chi-square + Cramér's V
"Create 2-3 visualizations of this data"
"Are there outliers in the price column?"
"What correlates most strongly with quantity?"
"Compare average order value between 2024 and 2025"
```

Every response includes:
- **Answer** — narrative citing only computed numbers, with significance stated where tests ran
- **Charts** — consistent ink-on-paper editorial style
- **View calculation** — the exact result tables behind the claims
- **View plan** — the validated step DAG with parameters and reasons
- **Warnings** — consolidated caveats (small samples, sparse bins, excluded binary correlations)

In agent mode you additionally get the **live tool trail** (each op as it was
called and whether it succeeded), and a `mode: agent` chip showing how many
model turns and provider calls the run cost.

Export everything — including the upload briefing — as a formatted Word report.

---

## Methods & references

Every statistical claim above is a textbook method executed by deterministic code (`insight/analytics/`). The papers behind them:

**Classical inference** — *Welch's t-test* (Welch, 1947, *Biometrika*), *Mann–Whitney U* (Mann & Whitney, 1947), *chi-square independence* (Pearson, 1900), *Cramér's V* with bias correction (Cramér, 1946), *one-way ANOVA F-test* (Fisher, 1925), t-distribution p-values (Student, 1908). Effect sizes follow Cohen's conventions — Cohen's d (Cohen, 1988) and η² benchmarks (Cohen, 1973).

**Regression** — OLS via least squares (Legendre, 1805; Gauss), coefficient standard errors from σ²(XᵀX)⁻¹, standardized betas and adjusted R² per Montgomery, Peck & Vining, *Introduction to Linear Regression Analysis*.

**Distribution & outliers** — adaptive binning blends Sturges' rule (Sturges, 1926, *JASA*) with the Freedman–Diaconis rule (Freedman & Diaconis, 1981); outlier fences use Tukey's 1.5×IQR rule (*Exploratory Data Analysis*, 1977). Rank-based association via Spearman (1904).

**Probability calibration** — `isotonic_calibration` implements the pool-adjacent-violators algorithm (Ayer et al., 1955; Robertson, Wright & Dykstra, *Order Restricted Statistical Inference*, 1988), scored by the Brier score (Brier, 1950) and expected calibration error in the style of Guo et al. (2017), following the score-to-probability calibration literature (Zadrozny & Elkan, 2002; Niculescu-Mizil & Caruana, 2005).

**LLM reliability patterns** — the planner-emits-plan / engine-computes split follows program-aided language models (Gao et al., *PAL*, ICML 2023); bounded plan-repair mirrors iterative self-refinement (Madaan et al., *Self-Refine*, NeurIPS 2023); independent recomputation of every headline number is an application of chain-of-verification ideas (Dhuliawala et al., 2023).

## Security & privacy

- **No arbitrary code execution:** the LLM does not generate or execute arbitrary Python code. All operations route through a fixed library of audited pandas/scipy functions or a strictly sandboxed, read-only embedded DuckDB engine — everything executes inside your own process. **Agent mode changes none of this:** a "tool call" can only *select* one of those registered operations and supply its parameters, which are schema-validated and column-checked before anything runs; the model still never writes code.
- **Context protection:** full tabular datasets are never dumped into the LLM context window. The provider receives only schema definitions, statistical fingerprints (column types, cardinality, missingness, sampled summaries), and the small set of computed aggregate values a given answer actually cites.
- **Sanitized SQL:** DuckDB queries are accepted only as a single read-only `SELECT`/`WITH` statement. Destructive and administrative keywords (`ATTACH`, `COPY`, `DROP`, `PRAGMA`, …) are rejected, the connection runs in-memory with external access disabled and configuration locked, results are row-capped, and a hard 10-second interrupt aborts runaway queries.

Still, aggregates and schemas leave your machine: whatever you ask, the numbers behind the answer are sent to your configured provider. Don't upload datasets whose derived statistics you wouldn't paste into a chat model.

---

## Troubleshooting

**`No provider available`** — no `GROQ_API_KEY`/`MISTRAL_API_KEY` found in `.env` or environment.

**`unknown column 'x'` in the answer** — the model referenced a column that doesn't exist; the repair loop already retried. Rephrase using exact column names from the profile expander.

**Rate limit errors** — Groq free tier allows 30 requests/min on gpt-oss models. Wait a moment, switch to `gpt-oss-20b`, or add a Mistral key for fallback.

**Agentic runs feel slow on free tiers** — each turn re-sends the tool history, and Groq's free TPM can spend minutes in silent retry backoff before a call succeeds (the outcome reports `throttled=0`; the wall clock tells the truth). Space calls (`--delay` in the eval runners, `INSIGHT_AGENTIC_DELAY_S` in the app), keep budgets sane, or switch to the built-in NVIDIA NIM endpoint (`NVIDIA_API_KEY`, model `nvidia/openai/gpt-oss-20b`) — it throttles in bursts rather than stretching every call.

**Charts not appearing** — check the `.artifacts/web/<session>/` folder is writable; render warnings appear under answers.

**Slow first load** — provider model catalogs are scanned on startup (~10s, cached); the dataset profile and briefing compute once per upload.

**Streamlit vs webapp sessions** — each front-end keeps its own session state; uploads don't transfer between them.

---

## Observability

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) to enable full tracing via [LangSmith](https://smith.langchain.com). Every question produces a root `insight.analyze` run containing:

- the validated **plan** JSON (with repair attempts visible as failed→retried children),
- one **LLM span per provider call** (model, tokens, latency, retries),
- step-level execution results and validation outcomes.

Agent-mode runs add an `agentic.run_loop` span plus one span per model turn,
with the tool calls it made.

Tracing is fully optional — with no key the engine runs identically and posts nothing.

---

## Roadmap

Completed: zero-prompt briefing ✅ · statistical engine (t-test / Mann-Whitney / chi-square / Cramer's V) ✅ · chained DAG plans ✅ · independent result verification ✅ · FastAPI + SSE front-end ✅ · token-by-token answer streaming & prefix caching ✅ · human-in-the-loop plan approval ✅ · LangSmith observability ✅ · statistics depth (ANOVA + eta-squared, OLS regression, confidence intervals) ✅ · isotonic calibration audits ✅ · conversation state machine ✅ · golden-question evaluation harness ✅ · Dockerfile + CI + query-result cache with eviction (API & SSE streaming paths) ✅ · embedded DuckDB relational analytics + dbt/alert artifact exports ✅ · agentic tool-use mode + alert-runner companion worker ✅ · agentic A/B eval harness + k-rep aggregator ✅

**Up next** (detailed notes in [HANDOVER.md](HANDOVER.md) and [future_scope.md](future_scope.md)):

- External session storage (Redis / Valkey) for multi-worker deployments
- Live database connections (read-only SQLite/Postgres DSN connectors)
- MAR/MNAR-aware missingness diagnostics (currently MCAR assumption is surfaced as a caveat)
- Published k≥3 agentic-vs-single-shot A/B numbers (harness, transcript recording and aggregator already landed)

*Explicitly deferred per architectural decision:* Python code-interpreter sandboxing (DuckDB covers relational analytics safely), auth / multi-user, PDF export, and persistent dashboards.

---

## Contributing

```bash
git checkout -b feature/your-feature
git commit -m 'add: your feature'
git push origin feature/your-feature
# open a pull request
```

Run `pytest` before submitting — CI-friendly, no keys required.

---

## Author

**Tanish Saroj** · [github.com/GitTanish](https://github.com/GitTanish) · [linkedin.com/in/tanishsaroj](https://linkedin.com/in/tanishsaroj)

---

*Built with [Groq](https://groq.com) · [Pydantic](https://docs.pydantic.dev) · [FastAPI](https://fastapi.tiangolo.com) · pandas · scipy*
