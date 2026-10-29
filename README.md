# INSIGHT

### An analytical agent that shows its work

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-teal.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/pydantic-v2-green.svg)](https://docs.pydantic.dev)
[![Tests](https://img.shields.io/badge/tests-CI%20green-brightgreen.svg)](#5-run-the-test-suite-optional-but-encouraged)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/GitTanish/badge.svg)](https://hub.xerj.org/contributors/GitTanish)

Upload a CSV — or four related ones. Ask anything in plain English. Get verified answers, newspaper-styled charts, a formatted Word report, and production-ready dbt and alert exports.

---

## The idea

The LLM is not the analytics engine.

That sounds obvious. It isn't. Most AI analytics tools let the model do the math, wrap the output in confident prose, and hope you don't check. That's how you get dashboards that look right and numbers that aren't.

Insight splits the job in two. The model reads your question, plans the analysis, and explains the result. Everything in between — every average, regression, significance test, and join — runs as deterministic pandas, scipy, and sandboxed DuckDB code. Before an answer reaches you, a second arithmetic path recomputes the headline group-by aggregates and cross-checks rankings, counts, and correlations; every number in the narrative is also traced back to computed evidence. If the narrative and the math disagree, you find out.

We think this is the only honest way to build an analytical agent. It's also why we can say something you can rarely say about an LLM product: the numbers are right by construction. You can click **View calculation** and inspect every one of them. You can click **View plan** and audit every step. When a plan fails validation or execution, the orchestrator repairs and retries — once — then tells you exactly what went wrong instead of hallucinating a result.

Set `INSIGHT_PLAN_APPROVAL=on` and a human gate appears between the planner and the engine: you see the exact operations, you can edit the JSON, and nothing touches your data until you approve.

## Proof, not promises

Everything here was measured on this repo during development. Nothing is aspirational.

| Anchor metric | What it proves |
|---|---|
| **20/20** golden-question eval cases passed live across three independent provider setups | the no-invented-numbers policy holds end to end |
| **Agentic vs single-shot A/B** — apparent pre-fix advantage, mostly explained by a provider bug; post-fix near parity | the edge that survives scrutiny is the **auditable tool trail**, not a proven accuracy gain — resolved in full below |
| **250K rows**: profiled in **2.0s**, heaviest op **<2s**, bivariate OLS in **125ms** | the deterministic engine holds up at real scale, not toy data |
| **−31.7% Brier score** after isotonic recalibration; over-confidence flagged at **ECE = 0.245** | calibration audits quantify rather than guess |
| **Explicit MCAR caveats** when referenced columns exceed 5% missingness, with row-exclusion counts and mixed count bases surfaced | missing-data behavior is honest and surfaced, not hidden |
| Engine answered diamonds' top cut as **Premium ($4,584 avg)** where LLM priors say *Ideal* | computed evidence beats model priors |
| **Sandboxed read-only SQL** — `DROP`, `COPY TO`, `ATTACH`, `PRAGMA`, chained statements and file reads all blocked; external access disabled and configuration locked; hard **10s interrupt** on runaway queries | read-only relational analytics you can safely hand to an LLM |
| **16-question real-world battery** (54K-row diamonds plus a messy 4-sheet clinical workbook with case-mismatched join keys), logged in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md), every answer cross-checked against independent pandas/scipy ground truth | it survives data the way data actually arrives |
| **Full offline suite green in CI** with zero API keys, group-by aggregates independently recomputed through a second arithmetic path, rankings/counts/correlations cross-checked, every number in the narrative traced back to computed evidence | verified by construction |

### The agentic benchmark, resolved

The agentic-vs-single-shot comparison ran in two phases on one pinned endpoint (NVIDIA NIM `gpt-oss-20b`, production budget, no fallback) over the same 20 golden cases. This is the definitive reading:

- **Pre-fix** — agentic **58/60 (96.7%, 95% CI 88.6–99.1)** vs single-shot **43/60 (71.7%, 59.2–81.5)**, at 2.17 vs 2.36 mean LLM calls. A substantial apparent advantage — but the follow-up diagnosis traced 13 of the 17 single-shot failures to a provider truncation bug that hit only the single-shot planner. Most of that gap was an artifact, not capability.
- **Post-fix** — **20/20 vs 19/20** in a single confirmation run (k=1): near parity. One run is a signal, not a margin.
- **The advantage that remains** — the **auditable tool trail**: every op call is visible, validated, and replayable. That is what agent mode demonstrably adds today, not measured accuracy.
- **Next step** — repeated, pinned post-fix comparisons (k≥3). Until those land, no accuracy-superiority claim gets made.

Both phases stay published because the story is the point: the first number made a claim the second didn't support, and the honest response was to say so.

## How it works

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
    ↓                 RECOMPUTATION of headline aggregates through a
    ↓                 second arithmetic path
Conversation State ──► active filters/dims/metrics persist across turns;
    ↓                    follow-ups become filter patches, not re-interpretation
Explainer (LLM) ──► narrative citing only computed values
    ↓
Answer + Evidence + Charts ──► streamed live (SSE) · exportable as .docx
```

In **agent mode**, steps 3–5 are replaced by a bounded loop: the model calls the engine's ops *as tools*, reads each real result, and decides what to do next — then its findings are rebuilt into a plan so every box below (validation, recomputation, charts, narration, grounding) runs unchanged. See [Agent mode](#agent-mode-opt-in).

## Agent mode (opt-in)

The default planner asks the model for one complete `AnalysisPlan` up front. **Agent mode** (`INSIGHT_AGENTIC=on`, or the per-question *Agent mode* toggle next to the composer in both UIs) lets the model instead call the engine's operations **as tools, one at a time**:

```
question → [call op → read real computed result] ×n → final answer
                │                                    │
                └─ live tool trail over SSE          └─ rebuilt into a plan →
                                                          validator · independent
                                                          recompute · charts ·
                                                          explainer · grounding
```

- **It observes before it decides.** A failed call (unknown column, bad params) comes back as a readable error, so the model fixes it instead of the whole plan dying on one mistake. It can chain: pass `input_step` to run on an earlier step's output table, or let `filter_rows` narrow the working row-set for later calls.
- **It is bounded.** `INSIGHT_AGENTIC_MAX_ITERATIONS` and `INSIGHT_AGENTIC_MAX_LLM_CALLS` cap turns and provider calls. Three identical failing calls abort the run. Rate limits surface as a visible "throttled" outcome, never a wrong answer.
- **The guarantees do not move.** Every tool call is validated into the op's Pydantic params and executed by the same `OperationSpec.run` the executor uses — no free-form code, ever. Afterwards the run is *rebuilt* into an `AnalysisPlan` and `ExecutionResult`, so validation, the independent recomputation of headline aggregates, chart rendering, the explainer, and the number-grounding guard are the exact same code the single-shot path uses. What changes is how the evidence is gathered, not how it is verified.
- **Chart planning is deterministic in this mode.** The two most recent table-producing steps are rendered with the same editorial rules, and any bad chart spec is rejected by the ordinary validator rather than trusted.
- **Provider-agnostic.** OpenAI-shaped tools on every OpenAI-compatible endpoint; Anthropic-shaped tools when the chain's primary provider is Anthropic.
- **`INSIGHT_PLAN_APPROVAL=on` (or `plan_only`) takes precedence.** That gate needs a reviewable plan before anything executes, so those requests use the single-shot planner.

Benchmark it against the single-shot planner with the same golden cases and grader: `python evaluation/run_eval_agentic.py --model <id> --no-fallback --record`, then aggregate k repetitions per arm with `evaluation/ab_stats.py`. The measured results — the pre-fix advantage, the provider bug behind most of it, and the near-parity post-fix confirmation — are resolved once, near the top of this file: [The agentic benchmark, resolved](#the-agentic-benchmark-resolved).

The harness paid for itself on its first live runs. It exposed an OpenAI wire-format bug in the assistant tool-call history that no scripted fake could see (every second turn was a hard HTTP 400; fixed and pinned by tests). It also proved that env-var provider pinning is unreliable on Windows — hence `--no-fallback` for provable single-model runs.

## MCP server (use the engine from any agent)

Insight ships an MCP server so any MCP-capable client (Claude Desktop, Cursor, opencode, a cron worker, your own agent) can load tables, run the registered operations, sandboxed SQL, or full DAG plans — and **verify any narrative against computed evidence** before it reaches a user. No LLM calls happen inside the server; everything is pandas/scipy/sandboxed DuckDB.

```bash
python -m insight.mcp_server                        # stdio (default)
INSIGHT_MCP_TRANSPORT=streamable-http python -m insight.mcp_server
```

```jsonc
// mcp client config (stdio example)
{
  "mcpServers": {
    "insight": {
      "command": "C:\\path\\to\\Insight\\.venv\\Scripts\\python.exe",
      "args": ["-m", "insight.mcp_server"]
    }
  }
}
```

| Tool | What it does |
|---|---|
| `load_table` | register inline CSV text or a file path (`.xlsx` expands to one table per sheet) |
| `list_tables` / `profile_table` | row counts, schemas, column-level profiles |
| `list_operations` | the 15 registered ops with their parameter schemas |
| `execute_operation` | run one op through the same registry + column validation as the webapp |
| `run_plan` | validate and execute a full `AnalysisPlan` DAG (`input_step` chaining included) |
| `run_sql` | one read-only SELECT/WITH statement in the DuckDB sandbox |
| `verify_claims` | check every number in prose against computed evidence; returns `ungrounded_numbers` plus independent-recomputation issues |

The typical agent flow is `load_table → execute_operation/run_plan → write the answer → verify_claims` with the returned `run_id`; anything flagged as ungrounded or mismatched is fixed before the user sees it. The toolkit that backs the server (`insight/toolkit.py`) is importable directly for code-first callers.

Validated against real public data by `tools/smoke_toolkit.py` — 54 checks across superstore, Palmer penguins, telco churn, daily temperatures and the generated evaluation corpora: all 15 ops, SQL sandbox rejections, DAG chaining, verification gates and the MCP flow, each cross-checked against independent pandas/numpy/scipy results and the seeded ground truth. Full evidence: `TEST.md` → "Toolkit + MCP real-data smoke".

Config: `INSIGHT_MCP_TRANSPORT` (`stdio` | `sse` | `streamable-http`), `INSIGHT_MCP_HOST` (default `127.0.0.1`), `INSIGHT_MCP_PORT` (default `8765`). HTTP transports require `INSIGHT_MCP_AUTH_TOKEN` (clients send `Authorization: Bearer ...`) and refuse to start without one; `INSIGHT_MCP_FILE_ROOTS` restricts `load_table(path=...)` (default: the current directory; `all` = unrestricted; `none` = disable), `INSIGHT_MCP_MAX_FILE_MB` caps file reads (default 80 MB), and `INSIGHT_MCP_MAX_RUNS` / `INSIGHT_MCP_MAX_TABLES` bound retained run snapshots (default 100) and registered table names (default 64). DNS-rebinding protection allows localhost hosts by default; extend with `INSIGHT_MCP_ALLOWED_HOSTS`.

## Alert runner

A SQL answer exports an **Alert monitor**: query + cron hint + threshold + Slack-ready webhook template. `tools/alert_runner.py` is the companion worker that closes the loop:

```bash
python -m tools.alert_runner --data data --monitors .artifacts/monitors --interval 300
python -m tools.alert_runner --once --dry-run     # render payloads, post nothing
```

It registers the CSVs as DuckDB tables, runs each monitor's query through the *same sandboxed session* the webapp uses (SELECT/WITH-only, keyword guards, interrupt watchdog, row cap), applies the threshold, and posts to the webhook when the condition **changes state** (recovery notifications via `--notify-recovery`, `--repeat` to post every cycle instead). Stdlib-only. The URL comes from the monitor's `webhook.url_env` (default `INSIGHT_ALERT_WEBHOOK_URL`) or `--webhook`.

## Bring your own key (BYOK)

Set `INSIGHT_REQUIRE_USER_KEY=on` and the deployment stops using your keys on other people's questions. Every query must carry the user's own provider key.

- The web UI shows a **Your API key** field in the left rail. It's stored only in the user's browser and sent per-request as an `x-insight-key` header. Leave it empty on hosted deployments and the request is refused with a clear 401.
- The JSON API accepts the same header, or an `api_key` field in the request body.
- The user's key is used for that request only. It is never persisted server-side.
- Uploads don't need a key — but background briefing polish is skipped when BYOK is on, since there is no user key yet at upload time.

This is the only built-in access gate. See the [deployment note](#configuration) before exposing an instance beyond localhost.

## Features

- **Zero-prompt briefing** — the moment a CSV lands, deterministic detectors surface the three most important findings (trend breaks, outliers, strong correlations, class imbalance, data quality), each with an *Investigate* button.
- **Next-best-action suggestions** — `insight/agentic.py` derives follow-up questions from the profile, what you've already analyzed, and columns shared across tables. Zero LLM cost, zero latency; the UI renders them as one-click chips.
- **Agentic tool-use mode (opt-in)** — the model calls analysis ops as tools one at a time, reads each real computed result, and decides what to do next. Bounded, throttle-aware, identical-failure abort, DAG-chained via `input_step` — and it runs on the *same* guarantees: every call routes through the deterministic op registry and column validator, and the answer is rebuilt into a plan so validation, recomputation, charts, explainer, and grounding still apply. The tool trail streams live into the UI.
- **Verified answers** — a validator checks sanity and small-sample caveats; an independent recomputation layer re-derives group-by aggregates through a second code path and cross-checks rankings, counts, and correlations before you see the answer.
- **Statistical tests without statistics mistakes** — binary-vs-numeric questions route to Welch's t-test or Mann–Whitney U with Cohen's d and plain-language group differences ("Class 1 averages 32% lower"); categorical pairs get chi-square with Cramér's V; 3+ groups vs numeric route to one-way ANOVA with η². Point-biserial correlation traps are actively blocked.
- **Regression & calibration** — OLS via `np.linalg.lstsq` with t-stat p-values, R² and standardized betas; `isotonic_calibration` audits whether a score truly tracks a binary outcome (PAV fit, reliability table, Brier before/after recalibration, ECE).
- **Honest missing-data handling** — every op coerces and drops explicitly; listwise deletions are reported (`rows_excluded_incomplete`, mixed count bases flagged), and referenced columns >5% missing raise an explicit MCAR-assumption caveat in the answer.
- **Conversation state machine** — filters applied in earlier turns persist as structured state; "now only 2025" becomes a filter patch instead of the model re-guessing context.
- **Relational analytics on embedded DuckDB** — drop 2–6 related CSVs, or a multi-sheet Excel workbook where every sheet becomes its own table. JOINs, window functions, per-group rankings and pivots via the sandboxed `sql_query` op: in-memory only, external access disabled, config locked, SELECT/WITH-only with keyword guards, interrupt-based timeout, hard row caps. Runtime errors flow straight into the existing plan-repair loop — self-correction without a code interpreter.
- **Actionable artifact exports** — any SQL answer ships two buttons: **Export dbt model** (a `.sql` model + `schema.yml` with column tests, zipped) and **Alert monitor** (a Slack-ready webhook monitor descriptor with the query, cron schedule, and payload template).
- **Chained "why" analysis** — plans are DAGs: `filter_rows → group_aggregate → statistical_test` compose via explicit `input_step` references.
- **Editorial charts that never look broken** — adaptive Freedman–Diaconis/Sturges binning, sparse-bin merging, cardinality-aware titles, automatic "Other" bucketing for long tails, horizontal-bar fallback for long labels, human tick formatting (12.5K / 3.2M).
- **Model-agnostic plug-and-play** — Groq, Mistral, OpenAI, Anthropic, OpenRouter, Ollama, NVIDIA NIM, or any OpenAI-compatible endpoint via env vars. Automatic cross-provider fallback on rate limits, 5xx — *and empty responses* (the classic "empty content" case on reasoning models was diagnosed on 2026-10-17: a missing `reasoning_effort` capability flag let the model consume its whole token budget reasoning, so the error now names the suspicion — `hit max_tokens=N`).
- **Token-by-token answer streaming** — the explainer streams over SSE into a live typing bubble, so the first words land in ~1s instead of after a multi-second blank wait. Time-to-first-token is measured and shown in the timing chip.
- **Human-in-the-loop plan approval** — with `INSIGHT_PLAN_APPROVAL=on`, every question stops at an editable plan (the exact operations, as JSON) that you approve before anything runs. Approved plans are *re-validated against your schema*, so a hand-edit naming a nonexistent column is rejected rather than silently trusted.
- **Live pipeline streaming** — watch *Planning → Executing → Validating → Explaining* happen in real time over SSE.
- **Headless JSON API** — `POST /api/query` exposes the same verified pipeline for scripts and integrations. Send `plan_only: true` to get a reviewable plan back instead of an answer, or `agentic: true` for one tool-calling run.
- **MCP server + verification toolkit** — `python -m insight.mcp_server` exposes the deterministic ops, sandboxed SQL, DAG plans, and `verify_claims` (number grounding + independent recomputation for any prose) to MCP clients like Claude Desktop, Cursor, or opencode. `insight/toolkit.py` is the same API as a plain Python import; no LLM inside.
- **Query-result cache** — identical (dataset hash, question, model, temperature, session state) hits are served from `.artifacts/cache/` with TTL + size-budget eviction (`INSIGHT_QUERY_CACHE=off` to disable). It serves the streaming UI path too: ~20× faster on repeats.
- **Prefix-cached prompts** — the planner's system prompt stays byte-stable per dataset (per-turn state lives in later messages, not the prefix), so providers serve it from their prompt cache. Reported as `cached_prompt_tokens`.
- **Golden-question eval harness** — 20 seeded and real benchmark cases (including the 54K-row diamonds dataset and a 250K-row synthetic retail set) with exact ground truth. `evaluation/run_eval.py` scores the single-shot path; `run_eval_agentic.py` runs the same cases and grader through the agentic loop (token/wall/model telemetry, optional raw transcripts via `--record`, `--no-fallback` for provable single-model runs); `ab_stats.py` aggregates k repetitions into per-case pass counts and Wilson CIs. The agentic loop additionally carries sampled mutation tests (11/11 killed) and 96% branch coverage.
- **Observability** — optional LangSmith tracing of every plan, step, and LLM call.

## Screenshots

| Upload & Configure | Ask Questions |
|---|---|
| ![Upload](image/1.png) | ![Query](image/2.png) |

| Generated Charts | Exported Report |
|---|---|
| ![Charts](image/4.png) | ![Report](image/download-report-demo.png) |

## Quickstart

**Prerequisites:** Python 3.10+ (3.11 recommended) and an API key for at least one supported provider. Groq is the shipped default and its free tier is enough to start.

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
# Example — Groq is the shipped default; any one supported provider key works
GROQ_API_KEY=gsk_your_key_here

# Optional — add more for extra models + cross-provider fallback
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

No local Python or virtualenv needed. The image runs unprivileged as a non-root user, binds to `0.0.0.0:8000`, and ships a `/healthz` healthcheck. Keys stay outside the image — they're read from `--env-file` at runtime. Step-by-step setup notes live in [HOW_TO_RUN.md](HOW_TO_RUN.md).

### 5. Run the test suite (optional but encouraged)

```bash
pytest
```

The offline unit/integration suite covers the profiler, every analytics operation (including statistical routing), the DAG executor, the validator + independent recomputation layer, planner repair loops, chart rendering rules, the briefing engine, conversation state, the query cache, text streaming, plan approval, prompt-cache stability, BYOK enforcement, and the FastAPI webapp end to end — all via a scripted fake LLM, no API key needed. It runs green in CI on every push. CI generates the deterministic eval datasets first (`python evaluation/generate_datasets.py`); a handful of tests additionally exercise the optional third-party `diamonds.csv` corpus when it is present, so the exact pass/skip split varies by platform and by which corpora you have. A separate `eval`-marked live suite (`INSIGHT_RUN_EVAL=1 pytest -m eval`) exercises the golden-question harness against a real provider.

### Evaluation harness (optional)

Benchmark the full pipeline against deterministic datasets with known ground truth (`--model` accepts any configured `<provider>/<model-id>`; the examples use the shipped default):

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

Pinned, fallback-free A/B — the recipe behind the results in [The agentic benchmark, resolved](#the-agentic-benchmark-resolved) (needs `NVIDIA_API_KEY`):

```bash
python evaluation/run_eval_agentic.py --model nvidia/openai/gpt-oss-20b --no-fallback --delay 2 --record
python evaluation/run_eval.py        --model nvidia/openai/gpt-oss-20b --no-fallback
# → reproduces the pinned A/B; the definitive reading of its two phases
#   is published above — "The agentic benchmark, resolved"
```

Cross-arm caveat: token totals are not comparable (the agentic runner sums every provider call; the single-shot runner records the meta usage block) — compare `llm_calls` and wall time instead.

Real-world runs are recorded in [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md): the question battery, ops executed per answer, timings, grounding flags, plus an independent ground-truth verification section — including the data-quality traps the engine had to navigate (case-mismatched join keys, inconsistent state codes).

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

See [.env.example](.env.example) for the full template. Model capabilities (JSON mode, reasoning effort) are declared per model; the orchestrator strips unsupported parameters automatically and falls back across providers when a 429/5xx hits.

Example — NVIDIA NIM (40 req/min, tool calling included):

```bash
NVIDIA_API_KEY=nvapi-...
# then use it anywhere a model id is accepted:
# python evaluation/run_eval_agentic.py --model nvidia/openai/gpt-oss-20b
```

## Configuration

File limits: up to 1,000,000 rows per table (`INSIGHT_MAX_UPLOAD_ROWS`) · CSV (UTF-8/Latin-1/CP1252, delimiters `,` `;` `\t`) · Excel `.xlsx`/`.xls`/`.xlsm` with one table registered per non-empty sheet (max 64 sheets). The web UI accepts at most 6 files per upload, 80 MB per file and 160 MB total; normalized CSVs are capped at 256 MB. Rows are capped before a full parse, and each session allows one in-flight query or upload at a time.

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
| `INSIGHT_ANSWER_GROUNDING` | `on` | every number in the narrative is checked against computed evidence; ungrounded figures trigger one explainer retry, and anything still unverified is surfaced as a warning |
| `INSIGHT_AGENTIC` | `off` | default question mode: agentic tool-calling loop instead of the single-shot planner (per-question override in both UIs) |
| `INSIGHT_AGENTIC_MAX_ITERATIONS` | `5` | max model turns per agentic question |
| `INSIGHT_AGENTIC_MAX_LLM_CALLS` | `9` | hard cap on provider calls per agentic question |
| `INSIGHT_AGENTIC_DELAY_S` | `0` | seconds between agentic provider calls (throttle headroom on free tiers) |
| `INSIGHT_REQUIRE_USER_KEY` | `off` | BYOK mode: every query must carry the user's own provider key (`x-insight-key` header / `api_key` body); requests without one get a 401 |
| `INSIGHT_AUTH_TOKEN` | — | require Bearer auth (API) or the `/auth?token=...` browser cookie on every non-health route |
| `INSIGHT_TRUST_PROXY_AUTH` | `off` | trust `X-Forwarded-User` from an authenticating reverse proxy instead of the shared token |
| `INSIGHT_COOKIE_SECURE` | `off` | mark session/auth cookies `Secure` (set when serving over TLS) |
| `INSIGHT_RATE_LIMIT` | `on` | per-identity sliding-window limits on query/upload routes |
| `INSIGHT_RATE_LIMIT_PER_MINUTE` | `60` | query requests/minute per identity |
| `INSIGHT_UPLOAD_RATE_LIMIT_PER_MINUTE` | `10` | uploads/minute per identity |
| `INSIGHT_SESSION_MAX` | `64` | LRU cap on in-memory sessions |
| `INSIGHT_SESSION_TTL_MINUTES` | `240` | idle TTL after which a session is dropped |
| `INSIGHT_SESSION_MAX_TURNS` | `25` | retained conversation turns per session (oldest dropped) |
| `INSIGHT_MAX_UPLOAD_ROWS` | `1000000` | rows accepted per uploaded table, enforced before full parsing |
| `INSIGHT_ARTIFACT_RETENTION_DAYS` | `14` | startup sweep of `.artifacts/web/*` older than this (`tools/retention.py` for cron) |
| `INSIGHT_CUSTOM_BASE_URL` | — | base URL for any OpenAI-compatible service |
| `INSIGHT_CUSTOM_API_KEY` | — | key for that service |
| `INSIGHT_CUSTOM_MODELS` | — | comma-separated model ids; empty = auto-discover via `GET {base_url}/models` |
| `INSIGHT_CUSTOM_DISCOVERY` | `on` | off = show only the curated `INSIGHT_CUSTOM_MODELS` list |
| `INSIGHT_CUSTOM_NAME` | `custom` | display name for the custom provider in the UI |

**Deployment note:** the webapp is single-worker by design — sessions (CSV bytes, profile, turns) live in process memory. Run exactly one uvicorn worker; horizontal scaling requires externalizing session state first. Sessions allow one in-flight query/upload at a time (concurrent work gets 409/redirect), retain at most 25 turns, and are never evicted while busy. Cache publishes are atomic. Set `INSIGHT_AUTH_TOKEN` (or run behind an authenticating proxy with `INSIGHT_TRUST_PROXY_AUTH=on`) plus `INSIGHT_COOKIE_SECURE=on` under TLS before exposing it beyond localhost. Query/upload rate limits are on by default; `/metrics` and `/readyz` support monitoring. The full production criteria, current gaps (multi-worker is the big one) and acceptance checks are in [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

## Project structure

```
webapp/                  FastAPI UI (primary)
├── app.py               routes: upload / SSE query stream / export / JSON API / models refresh
├── sessions.py          cookie-keyed server-side sessions (TTL + LRU, single-worker)
├── security.py          optional auth gate, identity, response hardening
├── ratelimit.py         per-identity sliding-window limits (429 + Retry-After)
├── telemetry.py         request IDs, JSON access log, Prometheus metrics
├── readiness.py         /readyz checks (artifacts writable + provider configured)
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
│   ├── agentic.py       deterministic next-best-action suggestions (zero LLM, zero latency)
│   ├── conversation/    AnalysisSessionState — filters/dims/metrics across turns
│   ├── execution/       Plan executor (sync core, async wrapper, input_step DAG)
│   ├── validation/      Result validator + independent recomputation checks
│   ├── ingest.py        shared CSV/Excel parsing (webapp + toolkit)
│   ├── toolkit.py       agent-facing engine API: load/run/verify, no LLM
│   ├── mcp_server.py    MCP wrapper exposing the toolkit (stdio/HTTP)
│   ├── cache.py         query-result cache (TTL + size-budget eviction)
│   ├── retention.py     artifact retention sweeper (startup + CLI)
│   ├── reports/         shared DOCX builder + dbt/alert artifact generators
│   ├── visualization/   Editorial theme + chart renderer from ChartSpec
│   ├── observability.py optional LangSmith tracing glue (no key → no-op)
│   ├── settings.py      env-var config model — every INSIGHT_* default lives here
│   └── orchestrator.py  plan → execute → validate → repair → render → explain (+ streaming variant)
├── evaluation/          golden-question harness: generator, 20 cases, single-shot + agentic runners, k-rep aggregator, scorer/reporter
├── tools/               companion workers (alert_runner, load_test, retention) + mutation sampler + SSE smoke test + real-data toolkit/MCP smoke (smoke_toolkit)
├── tests/               pytest suite (fake LLM + adversarial agentic suite, zero network needed)
├── promo/               standalone 30s product film, rendered frame-by-frame from code (see promo/README.md)
├── data/                sample datasets used by tests and logs
├── assets/              brand assets
└── image/               README screenshots
```

The `insight/` package is deliberately UI-agnostic — both front-ends call it through one function. The FastAPI app additionally exposes the pipeline as an SSE stream and a JSON API, so CLIs and future clients reuse the exact same verified engine.

### More docs

| Document | What it is |
|---|---|
| [HOW_TO_RUN.md](HOW_TO_RUN.md) | step-by-step local setup for testing |
| [TEST.md](TEST.md) | test-suite map and the story of how it got honest |
| [AUDIT.md](AUDIT.md) | first-principles audit of the architecture, claims checked against code |
| [evaluation/AUDIT_MATH_AND_ORCHESTRATION.md](evaluation/AUDIT_MATH_AND_ORCHESTRATION.md) | math + orchestration audit with known weaknesses stated, not hidden |
| [evaluation/REAL_WORLD_LOG.md](evaluation/REAL_WORLD_LOG.md) | the 16-question real-world battery with independent ground truth |
| [HANDOVER.md](HANDOVER.md) · [future_scope.md](future_scope.md) | engineering handover and forward plans |
| [promo/README.md](promo/README.md) | how the 30-second film is rendered from code |

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

In agent mode you additionally get the **live tool trail** (each op as it was called and whether it succeeded) and a `mode: agent` chip showing how many model turns and provider calls the run cost.

Export everything — including the upload briefing — as a formatted Word report.

## Methods & references

Every statistical claim above is a textbook method executed by deterministic code (`insight/analytics/`). The papers behind them:

**Classical inference** — *Welch's t-test* (Welch, 1947, *Biometrika*), *Mann–Whitney U* (Mann & Whitney, 1947), *chi-square independence* (Pearson, 1900), *Cramér's V* with bias correction (Cramér, 1946), *one-way ANOVA F-test* (Fisher, 1925), t-distribution p-values (Student, 1908). Effect sizes follow Cohen's conventions — Cohen's d (Cohen, 1988) and η² benchmarks (Cohen, 1973).

**Regression** — OLS via least squares (Legendre, 1805; Gauss), coefficient standard errors from σ²(XᵀX)⁻¹, standardized betas and adjusted R² per Montgomery, Peck & Vining, *Introduction to Linear Regression Analysis*.

**Distribution & outliers** — adaptive binning blends Sturges' rule (Sturges, 1926, *JASA*) with the Freedman–Diaconis rule (Freedman & Diaconis, 1981); outlier fences use Tukey's 1.5×IQR rule (*Exploratory Data Analysis*, 1977). Rank-based association via Spearman (1904).

**Probability calibration** — `isotonic_calibration` implements the pool-adjacent-violators algorithm (Ayer et al., 1955; Robertson, Wright & Dykstra, *Order Restricted Statistical Inference*, 1988), scored by the Brier score (Brier, 1950) and expected calibration error in the style of Guo et al. (2017), following the score-to-probability calibration literature (Zadrozny & Elkan, 2002; Niculescu-Mizil & Caruana, 2005).

**LLM reliability patterns** — the planner-emits-plan / engine-computes split follows program-aided language models (Gao et al., *PAL*, ICML 2023); bounded plan-repair mirrors iterative self-refinement (Madaan et al., *Self-Refine*, NeurIPS 2023); independent recomputation of headline aggregates is an application of chain-of-verification ideas (Dhuliawala et al., 2023).

## Security & privacy

- **No arbitrary code execution.** The LLM does not generate or execute arbitrary Python. All operations route through a fixed library of audited pandas/scipy functions or a strictly sandboxed, read-only embedded DuckDB engine — everything executes inside your own process. **Agent mode changes none of this:** a "tool call" can only *select* one of those registered operations and supply its parameters, which are schema-validated and column-checked before anything runs. The model still never writes code.
- **Context protection.** Full tabular datasets are never dumped into the LLM context window. The provider receives only schema definitions, statistical fingerprints (column types, cardinality, missingness, sampled summaries), and the small set of computed aggregate values a given answer actually cites.
- **Sanitized SQL.** DuckDB queries are accepted only as a single read-only `SELECT`/`WITH` statement. Destructive and administrative keywords (`ATTACH`, `COPY`, `DROP`, `PRAGMA`, …) are rejected, the connection runs in-memory with external access disabled and configuration locked, results are row-capped, and a hard 10-second interrupt aborts runaway queries.
- **Key handling.** Provider keys are read from `.env` or the environment and never logged. In BYOK mode, user keys travel per-request and are never persisted server-side.
- **Optional access control.** Set `INSIGHT_AUTH_TOKEN` to require Bearer auth (API) or the `/auth?token=...` cookie (browser) on every non-health route, or run behind an authenticating proxy with `INSIGHT_TRUST_PROXY_AUTH=on`. Queries and uploads are rate-limited per identity by default, sessions have an idle TTL and LRU cap, and generated artifacts are only served to the session that created them. Retention (`.artifacts/web/*`) is configurable and can be swept on startup or via `tools/retention.py`.

Still, aggregates and schemas leave your machine: whatever you ask, the numbers behind the answer are sent to your configured provider. Don't upload datasets whose derived statistics you wouldn't paste into a chat model.

## Troubleshooting

**`No provider available`** — no provider key found in `.env` or the environment. Set at least one (e.g. `GROQ_API_KEY`, `MISTRAL_API_KEY`, `OPENAI_API_KEY`).

**`unknown column 'x'` in the answer** — the model referenced a column that doesn't exist; the repair loop already retried. Rephrase using exact column names from the profile expander.

**Rate limit errors** — free tiers have minutely caps (e.g., Groq allows 30 requests/min on gpt-oss models). Wait a moment, switch to a smaller/faster model, or add a second provider key so fallback can kick in.

**Agentic runs feel slow on free tiers** — each turn re-sends the tool history, and a provider's free TPM cap can spend minutes in silent retry backoff before a call succeeds (the outcome reports `throttled=0`; the wall clock tells the truth). Space calls (`--delay` in the eval runners, `INSIGHT_AGENTIC_DELAY_S` in the app), keep budgets sane, or switch to a provider with burst-friendly throttling — e.g. NVIDIA NIM (`NVIDIA_API_KEY`, model `nvidia/openai/gpt-oss-20b`).

**Charts not appearing** — check the `.artifacts/web/<session>/` folder is writable; render warnings appear under answers.

**Slow first load** — provider model catalogs are scanned on startup (~10s, cached); the dataset profile and briefing compute once per upload.

**Streamlit vs webapp sessions** — each front-end keeps its own session state; uploads don't transfer between them.

## Observability

Set `LANGSMITH_API_KEY` (and optionally `LANGSMITH_PROJECT`) to enable full tracing via [LangSmith](https://smith.langchain.com). Every question produces a root `insight.analyze` run containing:

- the validated **plan** JSON (with repair attempts visible as failed→retried children),
- one **LLM span per provider call** (model, tokens, latency, retries),
- step-level execution results and validation outcomes.

Agent-mode runs add an `agentic.run_loop` span plus one span per model turn, with the tool calls it made.

Tracing is fully optional — with no key the engine runs identically and posts nothing.

Operational endpoints are always on and need no key: `GET /healthz` (liveness), `GET /readyz` (readiness — artifacts writable and a provider configured; 503 otherwise), and `GET /metrics` (Prometheus text: request counts/latency by normalized path, active sessions, uptime). Every response carries `X-Request-ID` (inbound IDs are preserved) and the server logs one JSON access line per request. The criteria these satisfy — and the gaps that remain (multi-worker session state above all) — live in [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).

## Roadmap

Completed: zero-prompt briefing ✅ · statistical engine (t-test / Mann-Whitney / chi-square / Cramer's V) ✅ · chained DAG plans ✅ · independent result verification ✅ · FastAPI + SSE front-end ✅ · token-by-token answer streaming & prefix caching ✅ · human-in-the-loop plan approval ✅ · LangSmith observability ✅ · statistics depth (ANOVA + eta-squared, OLS regression, confidence intervals) ✅ · isotonic calibration audits ✅ · conversation state machine ✅ · golden-question evaluation harness ✅ · Dockerfile + CI + query-result cache with eviction (API & SSE streaming paths) ✅ · embedded DuckDB relational analytics + dbt/alert artifact exports ✅ · agentic tool-use mode + alert-runner companion worker ✅ · agentic A/B eval harness + k-rep aggregator ✅ · MCP server + verification toolkit for external agents ✅ · production hardening (auth gate, rate limits, request IDs/metrics/readiness, session TTL + artifact retention, dependency/image scanning) ✅

**Up next** (detailed notes in [HANDOVER.md](HANDOVER.md) and [future_scope.md](future_scope.md)):

- External session storage (Redis / Valkey) for multi-worker deployments
- Live database connections (read-only SQLite/Postgres DSN connectors)
- MAR/MNAR-aware missingness diagnostics (currently the MCAR assumption is surfaced as a caveat)
- Repeated, pinned post-fix agentic-vs-single-shot A/B (k≥3), published as the canonical accuracy number (harness, transcript recording, and aggregator already landed)

*Explicitly deferred per architectural decision:* Python code-interpreter sandboxing (DuckDB covers relational analytics safely), auth / multi-user, PDF export, and persistent dashboards.

## Contributing

```bash
git checkout -b feature/your-feature
git commit -m 'add: your feature'
git push origin feature/your-feature
# open a pull request
```

Run `pytest` before submitting — CI-friendly, no keys required.

## Author

**Tanish Saroj** · [github.com/GitTanish](https://github.com/GitTanish) · [linkedin.com/in/tanishsaroj](https://linkedin.com/in/tanishsaroj)

---

*Built with [Pydantic](https://docs.pydantic.dev) · [FastAPI](https://fastapi.tiangolo.com) · pandas · scipy — and any LLM you bring.*
