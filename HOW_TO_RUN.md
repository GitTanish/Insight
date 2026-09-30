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
MISTRAL_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENROUTER_API_KEY=
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
```

## 5. Run the tests (no API key required)

```bash
pytest
```

215 offline tests via scripted fake LLMs. A separate live eval suite runs only
with `INSIGHT_RUN_EVAL=1 pytest -m eval` (needs a provider key):

```bash
python evaluation/generate_datasets.py
python evaluation/run_eval.py --model groq/openai/gpt-oss-120b
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "No LLM provider configured" | `.env` missing/empty — check key spelling and that you saved the file |
| First question is slow | model catalog scan + cold start; subsequent queries are faster |
| 429 / rate-limit errors | free Groq tier = 30 req/min; wait or switch model in sidebar |
| Port already in use | `streamlit run main.py --server.port 8502` |
| Wrong Python picked up | activate `.venv` first (step 2), or run `python -m streamlit run main.py` |

---

## Optional: run with Docker

```bash
docker build -t insight .
docker run -p 8000:8000 --env-file .env insight
```

Single worker by design (in-memory sessions) — see README deployment notes.
