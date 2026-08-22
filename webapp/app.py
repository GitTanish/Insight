from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import markdown
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from insight.briefing import findings_to_dicts, generate_briefing
from insight.domain.query import AnalysisRequest
from insight.llm import registry as llm_registry
from insight.orchestrator import analyze_stream
from insight.profiling.profiler import profile_dataframe
from insight.settings import get_settings
from webapp.sessions import STORE, WebSession
import webapp.services as svc

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["markdown"] = lambda text: markdown.markdown(
    text or "", extensions=["fenced_code"]
)

TIER_ORDER = {"fast": 0, "balanced": 1, "accurate": 2, "discovered": 3}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    try:
        task = asyncio.create_task(_lazy_discovery())
    except Exception:
        pass
    yield
    if task:
        task.cancel()


async def _lazy_discovery():
    try:
        await asyncio.wait_for(
            llm_registry.discover_models(), timeout=get_settings().discovery_timeout_s + 10
        )
    except Exception:
        pass


app = FastAPI(title="Insight", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

LOGO_PATH = BASE_DIR.parent / "assets" / "logo.png"
HAS_LOGO = LOGO_PATH.exists()
if HAS_LOGO:
    (BASE_DIR / "static").mkdir(exist_ok=True)
    import shutil as _shutil

    target = BASE_DIR / "static" / "logo.png"
    if not target.exists():
        _shutil.copy(LOGO_PATH, target)


def _sorted_models():
    return sorted(
        llm_registry.available_models(),
        key=lambda m: (m.provider, TIER_ORDER.get(m.tier, 9), m.model),
    )


def _provider_status() -> list[dict]:
    settings = get_settings()
    rows = []
    for spec in llm_registry.PROVIDERS.values():
        enabled = llm_registry.provider_enabled(spec, settings)
        count = sum(1 for m in {**llm_registry.MODEL_CATALOG, **llm_registry.DISCOVERED}.values() if m.provider == spec.name)
        hint = spec.api_key_env or "OLLAMA_BASE_URL"
        rows.append(
            {
                "name": spec.name.title(),
                "enabled": enabled,
                "dot": "\U0001F7E2" if enabled else "\u26AA",
                "state": f"{count} models" if enabled else f"set {hint}",
            }
        )
    return rows


def _base_ctx(request: Request, session: WebSession) -> dict:
    settings = get_settings()
    selected_model = session.model_id or next(
        (m.model_id for m in _sorted_models() if m.model_id == settings.default_model_id),
        (_sorted_models()[0].model_id if _sorted_models() else ""),
    )
    return {
        "session": session,
        "models": _sorted_models(),
        "selected_model": selected_model,
        "temperature": session.temperature if session.temperature is not None else settings.temperature,
        "providers": _provider_status(),
        "has_logo": HAS_LOGO,
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def index(request: Request):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)
    ctx = _base_ctx(request, session)
    response = templates.TemplateResponse(request, "index.html", ctx)
    if request.cookies.get("insight_sid") != sid:
        response.set_cookie("insight_sid", sid, httponly=True, samesite="lax")
    return response


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] = File(default=[]),
):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)

    incoming: list[UploadFile] = list(files or [])
    if file is not None and file.filename:
        incoming.insert(0, file)

    flash = None
    contents: dict[str, bytes] = {}
    primary_df = None
    if not incoming:
        flash = "No file received."
    else:
        for upload_file in incoming[:6]:
            data = await upload_file.read()
            if len(data) > MAX_UPLOAD_BYTES:
                flash = f"{upload_file.filename} exceeds the 80 MB limit."
                continue
            parsed, err = svc.parse_csv_content(data)
            if parsed is None:
                flash = f"{upload_file.filename}: {err or 'could not parse CSV.'}"
                continue
            contents[upload_file.filename] = data
            if primary_df is None:
                primary_df = parsed

        if primary_df is None:
            flash = flash or "Could not parse CSV."
        else:
            old_dir = get_settings().artifacts_dir / "web" / session.data_id
            session.reset_data()
            if old_dir.exists():
                shutil.rmtree(old_dir, ignore_errors=True)
            primary_name = next(iter(contents))
            profile = profile_dataframe(primary_df, primary_name, content_hash=None)
            findings = findings_to_dicts(generate_briefing(primary_df, profile))
            if get_settings().briefing_llm_polish:
                try:
                    from insight.briefing import polish_findings_with_llm
                    from insight.orchestrator import _build_chain_for

                    chain = _build_chain_for(
                        get_settings().fast_model_id, get_settings().temperature
                    )
                    findings = await polish_findings_with_llm(
                        findings,
                        chain.generate,
                        model_name=chain.primary_model,
                    )
                except Exception:
                    pass
            session.content = contents[primary_name]
            session.uploaded_name = primary_name
            session.profile = profile
            session.findings = findings
            session.multi_content = contents

    response = RedirectResponse(f"/?flash={flash}" if flash else "/", status_code=303)
    if request.cookies.get("insight_sid") != sid:
        response.set_cookie("insight_sid", sid, httponly=True, samesite="lax")
    return response


@app.post("/models/refresh")
async def refresh_models():
    try:
        await asyncio.wait_for(
            llm_registry.discover_models(),
            timeout=get_settings().discovery_timeout_s + 10,
        )
    except Exception:
        pass
    return RedirectResponse("/", status_code=303)


@app.post("/clear")
async def clear(request: Request):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)
    svc.clear_session_artifacts(session.data_id)
    session.reset_data()
    response = RedirectResponse("/", status_code=303)
    if request.cookies.get("insight_sid") != sid:
        response.set_cookie("insight_sid", sid, httponly=True, samesite="lax")
    return response


@app.post("/query")
async def query(
    request: Request,
    question: str = Form(...),
    model_id: str = Form(default=""),
    temperature: float = Form(default=-1),
):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)

    if not session.has_dataset:
        return JSONResponse({"error": "no dataset uploaded"}, status_code=400)

    analysis_request = AnalysisRequest(
        question=question,
        model_id=model_id or session.model_id or None,
        temperature=(temperature if temperature is not None and temperature >= 0 else None),
    )
    df, err = svc.parse_csv_content(session.content)
    if df is None:
        return JSONResponse({"error": err or "dataset unreadable"}, status_code=400)

    datasets: dict = {}
    for name, blob in session.multi_content.items():
        if name == session.uploaded_name:
            datasets[name] = df
            continue
        parsed, parse_err = svc.parse_csv_content(blob)
        if parsed is not None:
            datasets[name] = parsed

    artifacts_dir = svc.session_artifacts_dir(session.data_id)
    history: list[tuple[str, str]] = []
    for turn in session.turns[-3:]:
        history.append(("user", turn["question"]))
        history.append(("assistant", turn["response"].answer[:400]))

    session_state_model = None
    if session.analysis_state:
        from insight.conversation.state import AnalysisSessionState

        try:
            session_state_model = AnalysisSessionState.model_validate(
                session.analysis_state
            )
        except Exception:
            session_state_model = None

    def sse_frame(event_name: str, data_obj) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data_obj)}\n\n"

    async def event_generator():
        final_response = None
        async for event in analyze_stream(
            df, session.profile, analysis_request, artifacts_dir=artifacts_dir,
            history=history, state=session_state_model, datasets=datasets,
        ):
            etype = event["type"]
            if etype == "result":
                final_response = event["response"]
                html = templates.get_template("partials/_assistant.html").render(
                    response=final_response, data_id=session.data_id
                )
                session.turns.append({"question": question, "response": final_response})
                if final_response.meta.get("analysis_state"):
                    session.analysis_state = final_response.meta["analysis_state"]
                yield sse_frame("result", {"html": html})
            elif etype == "stage":
                yield sse_frame("stage", event)
            else:
                yield sse_frame(etype, event)
        yield sse_frame("end", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.post("/api/query")
async def api_query(request: Request):
    body = await request.json()
    question = (body or {}).get("question", "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)
    if not session.has_dataset:
        return JSONResponse({"error": "no dataset uploaded"}, status_code=400)

    df, err = svc.parse_csv_content(session.content)
    if df is None:
        return JSONResponse({"error": err}, status_code=400)

    analysis_request = AnalysisRequest(
        question=question,
        model_id=body.get("model_id"),
        temperature=body.get("temperature"),
    )
    from insight.orchestrator import analyze

    datasets: dict = {}
    for name, blob in session.multi_content.items():
        parsed, _ = svc.parse_csv_content(blob)
        if parsed is not None:
            datasets[name] = parsed

    response = await analyze(
        df,
        session.profile,
        analysis_request,
        artifacts_dir=svc.session_artifacts_dir(session.data_id),
        datasets=datasets or None,
    )
    return JSONResponse(response.model_dump(mode="json"))


@app.get("/export/docx")
async def export_docx(request: Request):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid)
    if session is None or not session.turns:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    import datetime

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset_name = session.uploaded_name or "Dataset"
    payload = svc.build_report_docx_bytes(
        session.turns, dataset_name, date_str, findings=session.findings
    )
    filename = f"Insight_Report_{dataset_name.replace('.csv', '')}.docx"
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _locate_sql_step(session: WebSession, step_id: int):
    for turn in reversed(session.turns):
        for step in turn["response"].meta.get("steps", []):
            if (
                step.get("operation") == "sql_query"
                and step.get("success")
                and step.get("step_id") == step_id
                and isinstance(step.get("params", {}).get("query"), str)
            ):
                return step["params"]["query"], turn["question"]
    return None, None


@app.post("/export/dbt")
async def export_dbt(request: Request, step_id: int = Form(...)):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid)
    if session is None or not session.turns:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    query, question = _locate_sql_step(session, step_id)
    if query is None:
        return JSONResponse({"error": "sql step not found"}, status_code=404)

    columns: list[str] = []
    for turn in reversed(session.turns):
        for table in turn["response"].tables:
            if table.step_id == step_id and table.columns:
                columns = [str(c) for c in table.columns]
                break
        if columns:
            break

    import io
    import zipfile

    from insight.reports.artifacts import build_dbt_model, slugify

    model_name = f"insight_{slugify(question)[:40]}" or "insight_model"
    files = build_dbt_model(model_name, query, description=question, columns=columns)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    slug = slugify(model_name)
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug}_dbt.zip"'},
    )


@app.get("/export/alert")
async def export_alert(request: Request, step_id: int):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid)
    if session is None or not session.turns:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    query, question = _locate_sql_step(session, step_id)
    if query is None:
        return JSONResponse({"error": "sql step not found"}, status_code=404)

    import json

    from insight.reports.artifacts import build_alert_monitor, slugify

    monitor = build_alert_monitor(f"insight_{slugify(question)[:40]}", query, question)
    return Response(
        content=json.dumps(monitor, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": 'attachment; filename="insight_monitor.json"'
        },
    )


@app.get("/artifacts/{data_id}/{filename}")
async def artifact(data_id: str, filename: str):
    root = (get_settings().artifacts_dir / "web" / data_id).resolve()
    candidate = (root / Path(filename).name).resolve()
    if not str(candidate).startswith(str(root)) or not candidate.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(candidate, media_type="image/png")
