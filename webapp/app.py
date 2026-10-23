from __future__ import annotations

import asyncio
import hmac
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import markdown
from fastapi import BackgroundTasks, FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
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
from insight.retention import sweep_artifacts
from insight.settings import get_settings
from webapp.sessions import STORE, SessionStore, WebSession
from webapp.readiness import readiness_report
from webapp.ratelimit import RateLimitMiddleware
from webapp.security import (
    AUTH_COOKIE,
    AuthMiddleware,
    SecurityHeadersMiddleware,
    session_cookie_kwargs,
)
from webapp.telemetry import REGISTRY, TelemetryMiddleware
import webapp.services as svc

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.filters["markdown"] = lambda text: markdown.markdown(
    text or "", extensions=["fenced_code"]
)

TIER_ORDER = {"fast": 0, "balanced": 1, "accurate": 2, "discovered": 3}
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_TOTAL_UPLOAD_BYTES = 160 * 1024 * 1024
MAX_TOTAL_NORMALIZED_BYTES = 256 * 1024 * 1024
MAX_UPLOAD_FILES = 6
UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = None
    try:
        await asyncio.to_thread(
            sweep_artifacts,
            get_settings().artifacts_dir,
            get_settings().artifact_retention_days,
        )
    except Exception:
        pass
    try:
        task = asyncio.create_task(_lazy_discovery())
    except Exception:
        pass
    yield
    if task:
        task.cancel()
        try:
            await task
        except BaseException:
            pass
    try:
        from insight.llm.http_pool import close_http_clients

        await close_http_clients()
    except Exception:
        pass
    try:
        from insight.observability import close_langsmith_client

        close_langsmith_client()
    except Exception:
        pass


async def _lazy_discovery():
    try:
        await asyncio.wait_for(
            llm_registry.discover_models(), timeout=get_settings().discovery_timeout_s + 10
        )
    except Exception:
        pass


app = FastAPI(title="Insight", lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

_BOOT = get_settings()
if _BOOT.rate_limit_enabled:
    app.add_middleware(
        RateLimitMiddleware,
        query_limit=_BOOT.rate_limit_per_minute,
        upload_limit=_BOOT.upload_rate_limit_per_minute,
    )
if _BOOT.auth_token or _BOOT.trust_proxy_auth:
    app.add_middleware(
        AuthMiddleware,
        token=_BOOT.auth_token or "",
        trust_proxy=_BOOT.trust_proxy_auth,
    )
app.add_middleware(SecurityHeadersMiddleware, hsts=_BOOT.cookie_secure)
app.add_middleware(TelemetryMiddleware)

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
        "agentic_default": settings.agentic,
    }


def _resolve_agentic_flag(raw) -> Optional[bool]:
    """None = follow the server default (INSIGHT_AGENTIC); else force a mode."""
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if value in {"", "default"}:
        return None
    return value in {"1", "true", "on", "yes"}


def _with_session_cookie(response, request: Request, sid: str):
    if request.cookies.get("insight_sid") != sid:
        response.set_cookie(
            "insight_sid", sid, **session_cookie_kwargs(secure=get_settings().cookie_secure)
        )
    return response


def _flash_redirect(request: Request, sid: str, message: str):
    return _with_session_cookie(
        RedirectResponse(f"/?flash={quote(message)}", status_code=303), request, sid
    )


async def _read_upload_limited(upload_file: UploadFile, limit: int):
    """Read an upload in chunks, stopping one chunk past ``limit``."""
    buffer = bytearray()
    while True:
        chunk = await upload_file.read(UPLOAD_CHUNK_BYTES)
        if not chunk:
            return bytes(buffer), len(buffer)
        buffer.extend(chunk)
        if len(buffer) > limit:
            return None, len(buffer)


async def _close_uploads(uploads: list[UploadFile]) -> None:
    for upload_file in uploads:
        try:
            await upload_file.close()
        except Exception:
            pass


@dataclass
class _PreparedUpload:
    content: bytes
    frames: dict
    primary_table: str
    uploaded_name: str
    profile: object
    findings: list[dict]
    multi_content: dict[str, bytes]


def _prepare_upload(
    raw: list[tuple[str, bytes]],
) -> tuple[_PreparedUpload | None, str | None]:
    """Parse uploaded bytes off the event loop; no session state is touched."""
    contents: dict[str, bytes] = {}
    frames: dict = {}
    primary_df = None
    primary_display = None
    normalized_total = 0

    for filename, data in raw:
        tables, err = svc.parse_tables(filename, data)
        if tables is None:
            return None, f"{filename}: {err or 'could not parse file.'}"
        for table_name, table_df in tables.items():
            normalized = table_df.to_csv(index=False).encode("utf-8")
            normalized_total += len(normalized)
            if normalized_total > MAX_TOTAL_NORMALIZED_BYTES:
                return None, "Upload expands to more than 256 MB of normalized CSV."
            contents[table_name] = normalized
            frames[table_name] = table_df
        if primary_df is None:
            primary_df = next(iter(tables.values()))
            primary_display = filename

    if primary_df is None or not contents:
        return None, "Could not parse CSV."

    primary_name = next(iter(contents))
    primary_frame = frames.get(primary_name, primary_df)
    profile = profile_dataframe(primary_frame, primary_name, content_hash=None)
    findings = findings_to_dicts(generate_briefing(primary_frame, profile))
    return (
        _PreparedUpload(
            content=contents[primary_name],
            frames=frames,
            primary_table=primary_name,
            uploaded_name=primary_display or primary_name,
            profile=profile,
            findings=findings,
            multi_content=contents,
        ),
        None,
    )


@dataclass
class _QueryContext:
    df: object
    profile: object
    datasets: dict
    history: list[tuple[str, str]]
    state: object
    artifacts_dir: Path
    data_id: str
    generation: int


def _build_query_context(
    snapshot: dict,
) -> tuple[_QueryContext | None, str | None]:
    primary_table = snapshot["primary_table"]
    frames = snapshot["frames"]
    df = frames.get(primary_table) if primary_table else None
    if df is None:
        if snapshot["content"] is None:
            return None, "dataset unreadable"
        df, err = svc.parse_csv_content(snapshot["content"])
        if df is None:
            return None, err or "dataset unreadable"

    datasets: dict = {}
    for name, content in snapshot["multi_content"].items():
        parsed = frames.get(name)
        if parsed is None:
            parsed, _parse_err = svc.parse_csv_content(content)
        if parsed is not None:
            datasets[name] = parsed

    history: list[tuple[str, str]] = []
    for turn in snapshot["turns"][-3:]:
        history.append(("user", turn["question"]))
        history.append(("assistant", turn["response"].answer[:400]))

    session_state_model = None
    if snapshot["analysis_state"]:
        from insight.conversation.state import AnalysisSessionState

        try:
            session_state_model = AnalysisSessionState.model_validate(
                snapshot["analysis_state"]
            )
        except Exception:
            session_state_model = None

    return (
        _QueryContext(
            df=df,
            profile=snapshot["profile"],
            datasets=datasets,
            history=history,
            state=session_state_model,
            artifacts_dir=svc.session_artifacts_dir(snapshot["data_id"]),
            data_id=snapshot["data_id"],
            generation=snapshot["generation"],
        ),
        None,
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    ready, report = readiness_report()
    return JSONResponse(report, status_code=200 if ready else 503)


@app.get("/metrics")
async def metrics():
    return Response(
        REGISTRY.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@app.get("/auth")
async def auth(request: Request):
    settings = get_settings()
    supplied = request.query_params.get("token", "")
    if settings.auth_token and not hmac.compare_digest(
        supplied.encode("utf-8"), settings.auth_token.encode("utf-8")
    ):
        return JSONResponse({"error": "invalid token"}, status_code=401)
    response = RedirectResponse("/", status_code=303)
    if settings.auth_token:
        response.set_cookie(
            AUTH_COOKIE,
            settings.auth_token,
            max_age=7 * 24 * 3600,
            **session_cookie_kwargs(secure=settings.cookie_secure),
        )
    return response


@app.get("/")
async def index(request: Request):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)
    ctx = _base_ctx(request, session)
    response = templates.TemplateResponse(request, "index.html", ctx)
    if request.cookies.get("insight_sid") != sid:
        response.set_cookie(
            "insight_sid", sid, **session_cookie_kwargs(secure=get_settings().cookie_secure)
        )
    return response


async def _polish_session_findings(
    store: SessionStore,
    sid: str,
    data_id: str,
) -> None:
    """LLM-polish a session's briefing after the upload response was sent."""
    session = store.get(sid)
    if session is None:
        return
    generation, current_data_id, findings = session.polish_context()
    if current_data_id != data_id:
        return
    try:
        from insight.briefing import polish_findings_with_llm
        from insight.orchestrator import _build_chain_for

        settings = get_settings()
        chain = _build_chain_for(settings.fast_model_id, settings.temperature)
        polished = await polish_findings_with_llm(
            findings,
            chain.generate,
            model_name=chain.primary_model,
        )
        if polished and session.set_findings(
            polished, generation=generation, data_id=data_id
        ):
            return
    except Exception:
        pass
    session.set_findings_status("ready", generation=generation, data_id=data_id)


@app.post("/upload")
async def upload(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    files: list[UploadFile] = File(default=[]),
):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)

    incoming: list[UploadFile] = list(files or [])
    if file is not None and file.filename:
        incoming.insert(0, file)

    if not session.begin_upload():
        await _close_uploads(incoming)
        return _flash_redirect(
            request,
            sid,
            "An analysis is already running for this session. Wait for it to finish "
            "before uploading a new dataset.",
        )

    flash = None
    try:
        if not incoming:
            flash = "No file received."
        elif len(incoming) > MAX_UPLOAD_FILES:
            flash = f"Too many files. Upload at most {MAX_UPLOAD_FILES} at once."
        else:
            raw: list[tuple[str, bytes]] = []
            total = 0
            for upload_file in incoming[:MAX_UPLOAD_FILES]:
                filename = upload_file.filename or "upload"
                data, size = await _read_upload_limited(upload_file, MAX_UPLOAD_BYTES)
                if data is None:
                    flash = f"{filename} exceeds the 80 MB limit."
                    break
                total += size
                if total > MAX_TOTAL_UPLOAD_BYTES:
                    flash = (
                        "Uploads exceed the 160 MB total limit. Upload fewer or "
                        "smaller files."
                    )
                    break
                raw.append((filename, data))

            if flash is None:
                try:
                    prepared, parse_error = await asyncio.to_thread(
                        _prepare_upload, raw
                    )
                except Exception:
                    prepared, parse_error = None, "Could not process the upload."
                if prepared is None:
                    flash = parse_error or "Could not parse CSV."
                else:
                    old_data_id, generation = session.replace_dataset(
                        content=prepared.content,
                        frames=prepared.frames,
                        primary_table=prepared.primary_table,
                        uploaded_name=prepared.uploaded_name,
                        profile=prepared.profile,
                        findings=prepared.findings,
                        multi_content=prepared.multi_content,
                    )
                    svc.clear_session_artifacts(old_data_id)

                    if (
                        get_settings().briefing_llm_polish
                        and not get_settings().require_user_key
                    ):
                        # The deterministic briefing is already available, so the
                        # LLM polish is a progressive enhancement. Run it after
                        # the redirect instead of making the user wait.
                        session.set_findings_status(
                            "polishing",
                            generation=generation,
                            data_id=session.data_id,
                        )
                        background_tasks.add_task(
                            _polish_session_findings,
                            STORE,
                            sid,
                            session.data_id,
                        )
    finally:
        await _close_uploads(incoming)
        session.end_upload()

    response = RedirectResponse(f"/?flash={quote(flash)}" if flash else "/", status_code=303)
    return _with_session_cookie(response, request, sid)


@app.post("/export/docx/custom")
async def export_docx_custom(request: Request, turn: list[int] = Form(default=[])):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid)
    turns = session.turns_snapshot() if session is not None else []
    if session is None or not turns:
        return RedirectResponse(
            "/?flash=Ask%20a%20question%20first%20%E2%80%94%20nothing%20to%20export%20yet.",
            status_code=303,
        )
    wanted = sorted({int(i) for i in turn if 0 <= int(i) < len(turns)})
    if not wanted:
        return RedirectResponse(
            "/?flash=Select%20at%20least%20one%20question%20for%20a%20custom%20report.",
            status_code=303,
        )
    import datetime

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset_name = session.uploaded_name or "Dataset"
    selected_turns = [turns[i] for i in wanted]
    payload = svc.build_report_docx_bytes(
        selected_turns, dataset_name, date_str, findings=session.findings_snapshot()
    )
    filename = f"Insight_Report_{dataset_name.replace('.csv', '')}_custom.docx"
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    if session.busy:
        return _flash_redirect(
            request,
            sid,
            "Wait for the current analysis to finish before clearing the session.",
        )
    svc.clear_session_artifacts(session.data_id)
    session.reset_data()
    return _with_session_cookie(
        RedirectResponse("/", status_code=303), request, sid
    )


@app.post("/query")
async def query(
    request: Request,
    question: str = Form(...),
    model_id: str = Form(default=""),
    temperature: float = Form(default=-1),
    approved_plan: str = Form(default=""),
    agentic: str = Form(default=""),
):
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)

    if not session.has_dataset:
        return JSONResponse({"error": "no dataset uploaded"}, status_code=400)

    user_key = request.headers.get("x-insight-key") or None
    if get_settings().require_user_key and not user_key:
        return JSONResponse(
            {"error": "This deployment is bring-your-own-key. Paste an API key in the left rail."},
            status_code=401,
        )

    plan_payload: dict | None = None
    if approved_plan.strip():
        try:
            plan_payload = json.loads(approved_plan)
        except json.JSONDecodeError:
            return JSONResponse({"error": "approved_plan is not valid JSON"}, status_code=400)

    analysis_request = AnalysisRequest(
        question=question,
        model_id=model_id or session.model_id or None,
        temperature=(temperature if temperature is not None and temperature >= 0 else None),
        api_key=user_key,
        approved_plan=plan_payload,
        agentic=_resolve_agentic_flag(agentic),
    )

    if not session.begin_query():
        return JSONResponse(
            {"error": "another analysis is already running for this session"},
            status_code=409,
        )
    try:
        snapshot = session.dataset_snapshot()
        context, context_error = await asyncio.to_thread(
            _build_query_context, snapshot
        )
        if context is None:
            session.end_query()
            return JSONResponse(
                {"error": context_error or "dataset unreadable"}, status_code=400
            )
    except BaseException:
        session.end_query()
        raise

    def sse_frame(event_name: str, data_obj) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data_obj)}\n\n"

    async def event_generator():
        try:
            async for event in analyze_stream(
                context.df,
                context.profile,
                analysis_request,
                artifacts_dir=context.artifacts_dir,
                history=context.history,
                state=context.state,
                datasets=context.datasets,
            ):
                etype = event["type"]
                if etype == "result":
                    final_response = event["response"]
                    html = templates.get_template("partials/_assistant.html").render(
                        response=final_response, data_id=context.data_id
                    )
                    session.add_turn(
                        generation=context.generation,
                        data_id=context.data_id,
                        question=question,
                        response=final_response,
                        analysis_state=final_response.meta.get("analysis_state"),
                        max_turns=get_settings().session_max_turns,
                    )
                    yield sse_frame("result", {"html": html})
                elif etype == "plan_review":
                    # Stops here by design: the user reviews/edits before execution.
                    yield sse_frame("plan_review", event)
                elif etype == "stage":
                    yield sse_frame("stage", event)
                else:
                    yield sse_frame(etype, event)
            yield sse_frame("end", {})
        finally:
            session.end_query()

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
    if not isinstance(body, dict):
        return JSONResponse({"error": "request body must be a JSON object"}, status_code=400)
    question = str(body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required"}, status_code=400)
    sid = request.cookies.get("insight_sid") or svc.new_session_id()
    session = STORE.get_or_create(sid)
    if not session.has_dataset:
        return JSONResponse({"error": "no dataset uploaded"}, status_code=400)

    user_key = request.headers.get("x-insight-key") or (body or {}).get("api_key") or None
    if get_settings().require_user_key and not user_key:
        return JSONResponse(
            {"error": "This deployment is bring-your-own-key. Provide api_key or X-Insight-Key."},
            status_code=401,
        )

    plan_only = bool(body.get("plan_only")) or get_settings().plan_approval
    approved_plan = body.get("approved_plan")
    if plan_only and approved_plan is not None:
        return JSONResponse(
            {"error": "plan_only and approved_plan are mutually exclusive"},
            status_code=400,
        )

    analysis_request = AnalysisRequest(
        question=question,
        model_id=body.get("model_id"),
        temperature=body.get("temperature"),
        api_key=user_key,
        plan_only=plan_only,
        approved_plan=approved_plan,
        agentic=_resolve_agentic_flag(body.get("agentic")),
    )

    if not session.begin_query():
        return JSONResponse(
            {"error": "another analysis is already running for this session"},
            status_code=409,
        )
    try:
        snapshot = session.dataset_snapshot()
        context, context_error = await asyncio.to_thread(
            _build_query_context, snapshot
        )
        if context is None:
            return JSONResponse(
                {"error": context_error or "dataset unreadable"}, status_code=400
            )

        if plan_only:
            # Human-in-the-loop: return the reviewable plan and stop before execution.
            async for event in analyze_stream(
                context.df,
                context.profile,
                analysis_request,
                artifacts_dir=context.artifacts_dir,
                datasets=context.datasets or None,
            ):
                if event.get("type") == "plan_review":
                    return JSONResponse({"plan": event["plan"]})
            return JSONResponse(
                {"error": "planner did not return a reviewable plan"}, status_code=502
            )

        response = None
        async for event in analyze_stream(
            context.df,
            context.profile,
            analysis_request,
            artifacts_dir=context.artifacts_dir,
            datasets=context.datasets or None,
        ):
            if event.get("type") == "result":
                response = event["response"]
        if response is None:
            return JSONResponse(
                {"error": "analysis produced no result"}, status_code=500
            )
        return JSONResponse(response.model_dump(mode="json"))
    finally:
        session.end_query()


@app.get("/export/docx")
async def export_docx(request: Request):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid)
    turns = session.turns_snapshot() if session is not None else []
    if session is None or not turns:
        return RedirectResponse(
            "/?flash=Ask%20a%20question%20first%20%E2%80%94%20nothing%20to%20export%20yet.",
            status_code=303,
        )
    import datetime

    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset_name = session.uploaded_name or "Dataset"
    payload = svc.build_report_docx_bytes(
        turns, dataset_name, date_str, findings=session.findings_snapshot()
    )
    filename = f"Insight_Report_{dataset_name.replace('.csv', '')}.docx"
    return Response(
        content=payload,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _locate_sql_step(turns: list[dict], step_id: int):
    for turn in reversed(turns):
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
    turns = session.turns_snapshot() if session is not None else []
    if session is None or not turns:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    query, question = _locate_sql_step(turns, step_id)
    if query is None:
        return JSONResponse({"error": "sql step not found"}, status_code=404)

    columns: list[str] = []
    for turn in reversed(turns):
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
    turns = session.turns_snapshot() if session is not None else []
    if session is None or not turns:
        return JSONResponse({"error": "nothing to export"}, status_code=400)
    query, question = _locate_sql_step(turns, step_id)
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
async def artifact(data_id: str, filename: str, request: Request):
    sid = request.cookies.get("insight_sid")
    session = STORE.get(sid) if sid else None
    if session is None or session.data_id != data_id:
        return JSONResponse({"error": "not found"}, status_code=404)
    root = (get_settings().artifacts_dir / "web" / data_id).resolve()
    candidate = (root / Path(filename).name).resolve()
    if not str(candidate).startswith(str(root)) or not candidate.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(candidate, media_type="image/png")
