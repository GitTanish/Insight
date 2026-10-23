# How to Run Insight

Quick local setup for testing. Full docs in [README.md](README.md).

---

## 1. Prerequisites

- **Python 3.10+** (3.11 recommended)
- **At least one LLM API key** (Groq recommended — free tier works)

---

## 2. Setup (first time only)

Open a terminal in the project folder:

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows (PowerShell / CMD):
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Add your API key(s)

Copy the template and fill in at least one key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Then edit `.env`:

```ini
GROQ_API_KEY=gsk_your_key_here        # <- minimum needed, free at console.groq.com

# optional extras (any subset works):
GROQ_API_KEY2=           # optional second Groq account (separate quota pool)
MISTRAL_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
NVIDIA_API_KEY=          # free NIM credits at build.nvidia.com
```

> Keys already present in your existing `.env` count too — nothing else to do.

## 4. Run the app

**Primary (FastAPI web server):**

```bash
uvicorn webapp.app:app --port 8000
```

Open `http://localhost:8000`. First launch scans provider model catalogs (~10s); later loads are instant.

**Legacy Streamlit UI (still maintained):**

```bash
streamlit run main.py
```

### Using it

1. **Upload a CSV or Excel file** in the left rail (multi-sheet workbooks register one table per sheet)
2. The **Zero-Prompt Briefing** appears immediately after upload — top findings with **Investigate** buttons, no question needed
3. Pick a model — defaults to Groq `gpt-oss-120b`
4. Ask questions; watch live pipeline stages (*Planning → Executing → Validating → Explaining*) stream in as they happen, and the answer itself type in token by token
5. Open **View calculation** / **View plan** under any answer to inspect it
6. **Export DOCX** includes the briefing findings plus every answer, chart, and caveat

### Optional: agent mode (tool-calling)

Tick **Agent mode** next to the composer (sidebar checkbox in Streamlit). The
model then stops writing one big plan and instead calls analysis tools one at a
time — seeing each real computed result before deciding what to do next. You'll
watch a **live tool trail** as it works, then get the same verified answer.

It is bounded (`INSIGHT_AGENTIC_MAX_ITERATIONS` / `INSIGHT_AGENTIC_MAX_LLM_CALLS`)
and stops with a visible message rather than a wrong answer if it runs out of
budget or the provider throttles it. To make agent mode the default for every
question, set `INSIGHT_AGENTIC=on` in `.env`.

### Optional: review every plan before it runs

Set `INSIGHT_PLAN_APPROVAL=on` in `.env` and restart. Each question then stops at
an editable plan card — the exact operations that *would* run — with an
**Approve & run** button. Edit the JSON first if you want different steps.

Approved plans are re-checked against your actual columns, so a typo'd column
name is rejected rather than executed.

### JSON API (headless)

```bash
# same cookie session as browser
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region has the highest revenue?"}'

# human-in-the-loop: get a plan back instead of an answer, then execute it
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region has the highest revenue?", "plan_only": true}'
# → {"plan": {...}}   then resubmit with the (optionally edited) plan:
#   -d '{"question": "...", "approved_plan": { ... }}'

# agentic mode for one request (true/false; omit to use the server default)
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region leads, and by how much?", "agentic": true}'
```

### Use the engine from another agent (MCP / toolkit, no LLM)

Any MCP-capable client (Claude Desktop, Cursor, opencode, a cron worker) can
drive the deterministic engine — load tables, run ops/DAG plans/sandboxed SQL,
and verify numbers in prose against computed evidence:

```bash
python -m insight.mcp_server        # stdio (default); SSE/streamable-http via INSIGHT_MCP_TRANSPORT
```

`insight/toolkit.py` is the same API as a plain Python import. Client config
and the tool table live in `README.md` → "MCP server". Validate the whole path
on real public data (downloads 4 CSVs on first run; needs `evaluation/data/`):

```bash
python evaluation/generate_datasets.py   # once, if evaluation/data/ is missing
python tools/smoke_toolkit.py            # 54 checks: 15 ops, SQL sandbox, DAG, verification, MCP
```

### Optional: run an exported alert monitor

A SQL answer in the webapp offers **Alert monitor**, which downloads a JSON
descriptor. Run it with the companion worker:

```bash
# one pass, print the payload, post nothing
python -m tools.alert_runner --data path/to/data --monitors path/to/monitors --once --dry-run

# loop every 5 minutes against a Slack incoming webhook
python -m tools.alert_runner --data data --monitors monitors --interval 300
#   set INSIGHT_ALERT_WEBHOOK_URL=... or pass --webhook https://hooks.slack.com/...
```

It only posts when the condition **changes state**, so a monitor that stays
triggered does not spam the channel; add `--notify-recovery` to also post when
it clears, or `--repeat` to post every cycle instead.

## 5. Run the tests (no API key required)

```bash
pytest
```

The full offline suite via scripted fake LLMs — no API key needed, and green in CI.
A separate live eval suite runs only
with `INSIGHT_RUN_EVAL=1 pytest -m eval` (needs a provider key):

```bash
python evaluation/generate_datasets.py
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b
python evaluation/run_eval_agentic.py --model groq/openai/gpt-oss-120b --limit 5
```

Pinned, fallback-free A/B — the recipe behind the published numbers (needs
`NVIDIA_API_KEY`; `--no-fallback` keeps the chain on one model so the run is
provable):

```bash
python evaluation/run_eval_agentic.py --model nvidia/openai/gpt-oss-20b --no-fallback --delay 2
python evaluation/run_eval.py        --model nvidia/openai/gpt-oss-20b --no-fallback
python evaluation/ab_stats.py --agentic DIR1,DIR2,DIR3 --baseline DIR4,DIR5,DIR6
```

Offline verification tooling for the agentic layer: `python tools/mutate.py`
(sampled mutation testing; mutmut is WSL-only) and a coverage run — both are
spelled out in TEST.md § "Core Regression Commands". A clean clone reaches
`pytest -q` exit 0 after `pip install -r requirements.txt -r requirements-dev.txt`
(missing `evaluation/data/` skips tests, it never fails them).

Assert that `pytest -q` exits 0 — not a test count (skips vary by platform and
by which optional corpora are present).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No LLM provider configured" | `.env` missing/empty — check key spelling and that you saved the file |
| First question is slow | model catalog scan + cold start; subsequent queries are faster |
| 429 / rate-limit errors | free Groq tier = 30 req/min; wait or switch model in sidebar |
| Agent mode says it ran out of budget | expected guardrail, not a crash — narrow the question, or raise `INSIGHT_AGENTIC_MAX_ITERATIONS` / `INSIGHT_AGENTIC_MAX_LLM_CALLS` |
| Agent mode says it was throttled | the provider rate-limited mid-run; retry shortly, add a fallback provider key, or set `INSIGHT_AGENTIC_DELAY_S=2` to space calls |
| Agent mode answer says nothing was invented | its tool calls all failed or the budget ran out — rephrase using exact column names, or turn agent mode off for that question |
| Agent mode toggle has no effect | `INSIGHT_PLAN_APPROVAL=on` forces the review-first planner for every question — that's by design |
| Port already in use | `streamlit run main.py --server.port 8502` |
| Wrong Python picked up | activate `.venv` first (step 2), or run `python -m streamlit run main.py` |

---

## Optional: run with Docker

```bash
docker build -t insight .
docker run -p 8000:8000 --env-file .env insight
```

Single worker by design (in-memory sessions) — see README deployment notes.

---

## Optional: production / shared deployment

Before exposing the webapp beyond localhost, set an auth boundary and TLS:

```bash
# .env
INSIGHT_AUTH_TOKEN=<long-random-token>
INSIGHT_COOKIE_SECURE=on
```

- **API clients** send `Authorization: Bearer <token>`.
- **Browsers** open `http://<host>:8000/auth?token=<token>` once; the server sets an
  `HttpOnly`, `SameSite=Lax` cookie. The token is then stripped from the URL.
- Behind an authenticating proxy, set `INSIGHT_TRUST_PROXY_AUTH=on` instead and have
  the proxy send `X-Forwarded-User`.

Other defaults in this mode: query/upload rate limits per identity
(`INSIGHT_RATE_LIMIT_PER_MINUTE=60`, `INSIGHT_UPLOAD_RATE_LIMIT_PER_MINUTE=10`,
429 + `Retry-After` when exceeded); sessions expire after
`INSIGHT_SESSION_TTL_MINUTES=240` idle and are capped at `INSIGHT_SESSION_MAX=64`;
uploaded tables are capped at `INSIGHT_MAX_UPLOAD_ROWS=1000000` rows, at most 6 files
per upload / 80 MB each / 160 MB total, and each session retains at most
`INSIGHT_SESSION_MAX_TURNS=25` turns while allowing one in-flight query or upload;
generated artifacts older than `INSIGHT_ARTIFACT_RETENTION_DAYS=14` are swept at
startup (run `python tools/retention.py --days 14 --dry-run` for a preview, or via
cron without `--dry-run`). Artifacts are only served to the session that created them.

Monitoring: `GET /readyz` (503 until a provider is configured), `GET /metrics`
(Prometheus text), and `X-Request-ID` on every response. Run a load smoke against
the upload path before opening traffic:

```bash
python tools/load_test.py --base-url http://127.0.0.1:8000 --count 20 --workers 8
```

The remaining structural gap is multi-worker deployment: sessions live in process
memory, so run exactly one uvicorn worker. Full criteria and gaps:
[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md).
