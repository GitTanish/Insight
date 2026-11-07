# How to run Insight

Installation and first run live in the [README Quickstart](../README.md#quickstart). That covers Python 3.10+, virtualenv setup, dependencies, provider keys, the primary FastAPI server, the legacy Streamlit UI, Docker, and the offline test suite.

This page keeps the operational details that are not part of the Quickstart.

## Legacy Streamlit UI

The Streamlit front end lives in `legacy_streamlit/` and is still maintained. Run it from the repository root:

```bash
streamlit run legacy_streamlit/main.py
```

It shares the same `insight/` engine as the FastAPI app. The two front ends keep separate session state, so uploads do not transfer between them.

## Use the UI

1. Upload a CSV or Excel file in the left rail. Multi-sheet workbooks register one table per sheet.
2. The zero-prompt briefing appears immediately after upload, with top findings and **Investigate** buttons.
3. Pick a model. The default is Groq `gpt-oss-120b` when that key is configured.
4. Ask a question. The pipeline stages stream live (Planning, Executing, Validating, Explaining), then the answer types in token by token.
5. Open **View calculation** or **View plan** under any answer to inspect the evidence.
6. **Export DOCX** includes the briefing findings plus every answer, chart, and caveat.

## Review a plan before it runs

Set `INSIGHT_PLAN_APPROVAL=on` in `.env` and restart. Each question then stops at an editable plan card showing the exact operations that would run, with an **Approve & run** button. Edit the JSON first if you want different steps.

Approved plans are re-checked against the actual columns, so a typo in a column name is rejected instead of executed.

## JSON API

The FastAPI app exposes the same pipeline without the browser UI:

```bash
# same cookie session as the browser
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region has the highest revenue?"}'

# human-in-the-loop: get a plan back instead of an answer, then execute it
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region has the highest revenue?", "plan_only": true}'
# then resubmit with the optionally edited plan:
#   -d '{"question": "...", "approved_plan": { ... }}'

# agentic mode for one request (true/false; omit to use the server default)
curl -X POST http://localhost:8000/api/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Which region leads, and by how much?", "agentic": true}'
```

## MCP server and toolkit

Any MCP-capable client can drive the deterministic engine without an LLM inside the server:

```bash
python -m insight.mcp_server        # stdio (default); SSE/streamable-http via INSIGHT_MCP_TRANSPORT
```

`insight/toolkit.py` is the same API as a plain Python import. The tool table and client configuration are in the README section [MCP server](../README.md#mcp-server-use-the-engine-from-any-agent).

Validate the whole path on real public data. The first run downloads four CSVs into `data/` and needs `evaluation/data/`:

```bash
python evaluation/generate_datasets.py   # once, if evaluation/data/ is missing
python tools/smoke_toolkit.py            # 54 checks: 15 ops, SQL sandbox, DAG, verification, MCP
```

## Standalone verifier (CLI and HTTP)

The same grounding/recomputation certificate is available without an MCP client.

```bash
# Evidence-only: every number in the answer must appear in evidence.json
python -m insight.verify_cli --answer "Revenue was 77." --evidence evidence.json

# Execute one deterministic operation, then verify the narrative
python -m insight.verify_cli --answer "North leads with 150." \
  --data sales.csv --operation group_aggregate --params-file params.json

# Verify prose against a sandboxed SQL result
python -m insight.verify_cli --answer-file draft.md --data sales.csv \
  --sql "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region"

# Gate CI on warnings as well as hard failures
python -m insight.verify_cli --answer-file draft.md --evidence evidence.json \
  --fail-on-warn --out cert.json
```

`--params-file` exists because PowerShell can mangle inline JSON. The CLI prints
the certificate to stdout (or `--out`), exits 1 on `fail`, and returns 2 for
input or operation errors. A runnable "lying agent" demo:

```bash
python tools/demo_epistemic_firewall.py
```

Local HTTP:

```bash
uvicorn insight.verify_service:app --host 127.0.0.1 --port 8787
curl -s http://127.0.0.1:8787/v1/verify \
  -H 'content-type: application/json' -d @request.json
```

`POST /v1/verify` accepts either `evidence` or `data` plus exactly one of
`operation` / `sql`. It returns HTTP 200 with the certificate even when the
verdict is `fail`; malformed requests are 422. Set
`INSIGHT_VERIFY_AUTH_TOKEN` for bearer auth and `INSIGHT_VERIFY_MAX_CSV_MB` to
cap inline CSV. This is a local/dev surface: no persistence, queues, or tenancy.
The full contract is in [CERTIFICATE.md](CERTIFICATE.md).

## Alert runner

A SQL answer in the webapp offers **Alert monitor**, which downloads a JSON descriptor. Run it with the companion worker:

```bash
# one pass, print the payload, post nothing
python -m tools.alert_runner --data path/to/data --monitors path/to/monitors --once --dry-run

# loop every 5 minutes against a Slack incoming webhook
python -m tools.alert_runner --data data --monitors monitors --interval 300
# set INSIGHT_ALERT_WEBHOOK_URL=... or pass --webhook https://hooks.slack.com/...
```

It posts only when the condition changes state, so a monitor that stays triggered does not spam the channel. Add `--notify-recovery` to also post when it clears, or `--repeat` to post every cycle.

## Troubleshooting

| Problem | Fix |
|---|---|
| "No LLM provider configured" | `.env` is missing or empty. Check key spelling and save the file. |
| First question is slow | Model catalog scan plus cold start. Later queries are faster. |
| 429 or rate-limit errors | Free Groq tier allows 30 requests per minute. Wait or switch model. |
| Agent mode says it ran out of budget | Expected guardrail, not a crash. Narrow the question or raise `INSIGHT_AGENTIC_MAX_ITERATIONS` / `INSIGHT_AGENTIC_MAX_LLM_CALLS`. |
| Agent mode says it was throttled | The provider rate-limited mid-run. Retry shortly, add a fallback key, or set `INSIGHT_AGENTIC_DELAY_S=2`. |
| Agent mode says nothing was invented | Its tool calls all failed or the budget ran out. Rephrase with exact column names, or turn agent mode off for that question. |
| Agent mode toggle has no effect | `INSIGHT_PLAN_APPROVAL=on` forces the review-first planner for every question. That is by design. |
| Port already in use | `streamlit run legacy_streamlit/main.py --server.port 8502` |
| Wrong Python picked up | Activate `.venv` first, or run `python -m streamlit run legacy_streamlit/main.py`. |

## Production deployment

Before exposing the webapp beyond localhost, set an auth boundary and TLS:

```bash
# .env
INSIGHT_AUTH_TOKEN=<long-random-token>
INSIGHT_COOKIE_SECURE=on
```

- **API clients** send `Authorization: Bearer <token>`.
- **Browsers** open `http://<host>:8000/auth?token=<token>` once. The server sets an `HttpOnly`, `SameSite=Lax` cookie and strips the token from the URL.
- Behind an authenticating proxy, set `INSIGHT_TRUST_PROXY_AUTH=on` instead and have the proxy send `X-Forwarded-User`.

The webapp is single-worker by design because sessions live in process memory. Run exactly one uvicorn worker until session state is externalized. Rate limits, session TTL, upload caps, artifact retention, and monitoring endpoints are documented in the README configuration section. The full release gate and remaining gaps are in [PRODUCTION_READINESS.md](../PRODUCTION_READINESS.md).
