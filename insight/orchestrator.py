from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, AsyncIterator, Optional

from insight.domain.dataset import DatasetProfile
from insight.domain.errors import (
    AllProvidersFailedError,
    InsightError,
    PlanValidationError,
)
from insight.domain.query import AnalysisRequest
from insight.domain.visualization import AnalysisResponse, Evidence
from insight.conversation.state import (
    AnalysisSessionState,
    extract_state_from_plan,
    serialize_state_for_planner,
)
from insight.execution.executor import execute_plan
from insight.llm.base import LLMRequest
from insight.llm.registry import build_llm_chain
from insight.observability import attach_metadata, configure_langsmith, trace_span
from insight.planning.planner import Planner
from insight.planning.prompts import build_explainer_messages
from insight.settings import get_settings
from insight.validation.result_validator import validate_execution
from insight.visualization.renderer import render_charts
from insight.visualization.themes import EDITORIAL_THEME

if TYPE_CHECKING:
    from insight.llm.base import LLMProvider
    from insight.llm.registry import ModelInfo

_ChainEntry = tuple["LLMProvider", str, "ModelInfo"]


class _Chain:
    def __init__(
        self,
        entries: list[_ChainEntry],
        temperature: float,
        reasoning_effort: str,
    ):
        self.entries = entries
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.last_used: Optional[str] = None
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.llm_calls: int = 0
        self.llm_latency_ms: int = 0

    @property
    def primary_model(self) -> str:
        return self.entries[0][1]

    async def generate(self, request: LLMRequest):
        errors: list[Exception] = []
        for provider, model, info in self.entries:
            updates: dict = {
                "model": model,
                "temperature": (
                    self.temperature
                    if request.temperature is None
                    else request.temperature
                ),
            }
            if info.supports_reasoning_effort is False:
                updates["reasoning_effort"] = None
            if info.supports_json_mode is False:
                updates["allow_server_json"] = False
            effective = request.model_copy(update=updates)
            started = time.monotonic()
            try:
                response = await provider.generate(effective)
                self.last_used = f"{provider.name}/{model}"
                self.llm_calls += 1
                self.llm_latency_ms += response.latency_ms
                self.tokens_in += response.prompt_tokens or 0
                self.tokens_out += response.completion_tokens or 0
                attach_metadata(
                    llm_provider=provider.name,
                    llm_model=model,
                    llm_latency_ms=response.latency_ms,
                    llm_prompt_tokens=response.prompt_tokens,
                    llm_completion_tokens=response.completion_tokens,
                    wall_ms=int((time.monotonic() - started) * 1000),
                )
                return response
            except InsightError as exc:
                attach_metadata(
                    llm_provider=provider.name,
                    llm_model=model,
                    llm_failed=True,
                    llm_error=str(getattr(exc, "detail", exc.message))[:300],
                )
                errors.append(exc)
            except Exception as exc:
                errors.append(InsightError(str(exc)))
        raise AllProvidersFailedError(errors)


def _build_chain_for(model_id: Optional[str], temperature: float) -> _Chain:
    _, entries = build_llm_chain(model_id or get_settings().default_model_id)
    return _Chain(entries=entries, temperature=temperature, reasoning_effort=get_settings().reasoning_effort)


def _cell(value) -> str:
    return "" if value is None else str(value)


def _tables_markdown(execution_result) -> str:
    sections: list[str] = []
    for step_id, table in execution_result.tables_by_step().items():
        header = f"| {' | '.join(table.columns)} |"
        sep = "|" + "|".join("---" for _ in table.columns) + "|"
        rows = [
            "| " + " | ".join(_cell(v) for v in row[: len(table.columns)]) + " |"
            for row in table.rows[:10]
        ]
        truncation = (
            f"\n(showing 10 of {table.total_rows:,} rows)" if table.total_rows > 10 else ""
        )
        sections.append(
            f"### step {step_id}: {table.name}\n{header}\n{sep}\n" + "\n".join(rows) + truncation
        )
    return "\n\n".join(sections) if sections else "(no tabular results)"


def _calculations_text(execution_result) -> str:
    lines = [
        f"- {c.label}: {_cell(c.value)}" + (f" ({c.detail})" if c.detail else "")
        for c in execution_result.all_calculations()
    ]
    return "\n".join(lines) if lines else "(none)"


def _fallback_answer(validation, execution_result) -> str:
    lines: list[str] = ["**Analysis complete.** (deterministic summary; narrative generation failed)"]
    calcs = execution_result.all_calculations()
    if calcs:
        lines.append("")
        for calc in calcs[:8]:
            detail = f" â€” {calc.detail}" if calc.detail else ""
            lines.append(f"- **{calc.label}**: {_cell(calc.value)}{detail}")
    tables = execution_result.tables_by_step()
    if tables:
        lines.append(f"\n{len(tables)} result table(s) produced; see evidence below.")
    if validation.warning_messages():
        lines.append("\n*Caveats:* " + "; ".join(validation.warning_messages()[:3]))
    return "\n".join(lines)


async def analyze_stream(
    df,
    profile: DatasetProfile,
    request: AnalysisRequest,
    artifacts_dir=None,
    history: Optional[list[tuple[str, str]]] = None,
    state: Optional[AnalysisSessionState] = None,
    datasets: Optional[dict] = None,
) -> AsyncIterator[dict]:
    configure_langsmith()
    settings = get_settings()
    started = time.monotonic()
    if artifacts_dir is not None:
        from pathlib import Path as _PathDir

        artifacts_dir = _PathDir(artifacts_dir)

    sql_schema_text: Optional[str] = None
    table_datasets = dict(datasets) if datasets else {profile.name: df}
    if settings.sql_enabled:
        from insight.analytics.sql_engine import DuckSession

        with DuckSession(table_datasets) as schema_session:
            sql_schema_text = schema_session.describe()

    async def _execute_with_sql(plan_to_run):
        if not settings.sql_enabled:
            return await execute_plan(df, plan_to_run)
        from insight.analytics.sql_engine import DuckSession

        session = DuckSession(table_datasets)
        try:
            return await execute_plan(df, plan_to_run, duck_ctx=session)
        finally:
            session.close()

    session_context = serialize_state_for_planner(state)
    if sql_schema_text:
        session_context = (
            (session_context + "\n\n" if session_context
             and not session_context.startswith("(") else "")
            + "Available SQL tables (embedded DuckDB, read-only):\n" + sql_schema_text
        )
    chain = _build_chain_for(
        request.model_id,
        settings.temperature if request.temperature is None else request.temperature,
    )
    attach_metadata(
        dataset_id=profile.fingerprint.dataset_id,
        requested_model=request.model_id,
        row_count=profile.row_count,
        column_count=profile.column_count,
    )

    planner = Planner(
        generate=chain.generate,
        model_name=chain.primary_model,
        max_repair_attempts=settings.max_repair_attempts,
        max_tokens=settings.planner_max_tokens,
        reasoning_effort=chain.reasoning_effort,
    )

    repair_attempted = False

    yield {"type": "stage", "key": "planning", "label": "Planning investigation..."}

    try:
        t_plan = time.monotonic()
        plan = await planner.plan(
            request.question,
            profile,
            history=history,
            session_context=session_context,
            sql_schema=sql_schema_text,
        )
        planning_ms = int((time.monotonic() - t_plan) * 1000)
    except PlanValidationError as exc:
        yield {
            "type": "stage",
            "key": "planning_failed",
            "label": "Planning failed",
            "level": "error",
        }
        yield {
            "type": "result",
            "response": AnalysisResponse(
                question=request.question,
                answer=(
                    "I could not construct a reliable analysis plan for this question.\n\n"
                    "**What went wrong:**\n"
                    + "\n".join(f"- {i}" for i in exc.issues[:6])
                    + "\n\nTry rephrasing with specific column names."
                ),
                meta={"model": chain.last_used},
                error="plan_validation_failed",
            ),
        }
        return

    yield {
        "type": "plan_preview",
        "objective": plan.objective,
        "steps": [{"id": s.step_id, "operation": s.operation} for s in plan.steps],
    }

    yield {
        "type": "stage",
        "key": "executing",
        "label": f"Executing {len(plan.steps)} deterministic step(s)...",
    }
    execution = await _execute_with_sql(plan)
    validation = validate_execution(plan, execution, df=df, profile=profile)
    yield {
        "type": "validation",
        "valid": validation.valid,
        "warning_count": len(validation.warnings),
    }

    if not validation.valid and settings.max_repair_attempts > 0:
        repair_attempted = True
        yield {
            "type": "stage",
            "key": "repairing",
            "label": "Validator rejected results â€” repairing plan once...",
            "level": "warn",
        }
        repair_issues = "; ".join(validation.error_messages()[:8])
        repair_question = (
            f"{request.question}\n\n(A previous attempt failed: {repair_issues}. "
            "Produce a corrected plan avoiding these problems.)"
        )
        try:
            repaired_plan = await planner.plan(
                repair_question,
                profile,
                history=history,
                session_context=session_context,
                sql_schema=sql_schema_text,
            )
            second_execution = await _execute_with_sql(repaired_plan)
            second_validation = validate_execution(
                repaired_plan, second_execution, df=df, profile=profile
            )
            plan, execution, validation = (
                repaired_plan,
                second_execution,
                second_validation,
            )
        except PlanValidationError:
            pass

    analysis_state = extract_state_from_plan(plan, base=state)
    if profile.fingerprint.dataset_id:
        analysis_state.dataset_id = profile.fingerprint.dataset_id
    yield {"type": "state", "state": analysis_state.model_dump(mode="json")}

    yield {"type": "stage", "key": "rendering", "label": "Rendering editorial charts..."}
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots, chart_warnings = render_charts(
        plan.charts, execution.tables_by_step(), artifacts_dir, EDITORIAL_THEME
    )

    yield {"type": "stage", "key": "explaining", "label": "Composing the dispatch..."}
    explainer_request = LLMRequest(
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in build_explainer_messages(
                question=request.question,
                objective=plan.objective,
                tables_markdown=_tables_markdown(execution),
                calculations_text=_calculations_text(execution),
                warnings_text="\n".join(validation.warning_messages() + chart_warnings),
            )
        ],
        model=chain.primary_model,
        temperature=None,
        max_tokens=settings.explainer_max_tokens,
        reasoning_effort=chain.reasoning_effort,
    )

    t_explain = time.monotonic()
    try:
        explanation_response = await chain.generate(explainer_request)
        answer = explanation_response.content.strip()
    except InsightError:
        answer = _fallback_answer(validation, execution)
    explaining_ms = int((time.monotonic() - t_explain) * 1000)

    evidence: list[Evidence] = [
        Evidence(kind="calculation", label=c.label, value=_cell(c.value), detail=c.detail)
        for c in execution.all_calculations()[:12]
    ]
    for message in validation.warning_messages() + chart_warnings:
        evidence.append(Evidence(kind="warning", label=message))

    meta: dict = {
        "model": chain.last_used,
        "objective": plan.objective,
        "repair_attempted": repair_attempted,
        "planning_ms": planning_ms,
        "explaining_ms": explaining_ms,
        "execution_ms": execution.duration_ms,
        "total_ms": int((time.monotonic() - started) * 1000),
        "tokens_in": chain.tokens_in,
        "tokens_out": chain.tokens_out,
        "llm_calls": chain.llm_calls,
        "llm_latency_ms": chain.llm_latency_ms,
        "dataset_id": profile.fingerprint.dataset_id,
        "analysis_state": analysis_state.model_dump(mode="json"),
        "steps": [
            {
                "step_id": s.step_id,
                "operation": s.operation,
                "success": s.success,
                "error": s.error,
                "params": next(
                    (
                        {"reason": p.reason, **p.params}
                        for p in plan.steps
                        if p.step_id == s.step_id
                    ),
                    {},
                ),
            }
            for s in execution.steps
        ],
    }

    yield {
        "type": "result",
        "response": AnalysisResponse(
        question=request.question,
        answer=answer,
        evidence=evidence,
        tables=list(execution.tables_by_step().values()),
        plots=plots,
        validation=validation,
        meta=meta,
        ),
    }


async def analyze(
    df,
    profile: DatasetProfile,
    request: AnalysisRequest,
    artifacts_dir=None,
    history: Optional[list[tuple[str, str]]] = None,
    state: Optional[AnalysisSessionState] = None,
    use_cache: bool = True,
    datasets: Optional[dict] = None,
) -> AnalysisResponse:
    import datetime as _dt
    from pathlib import Path as _Path

    from insight import cache as result_cache
    from insight.observability import current_trace_url, is_tracing_enabled

    configure_langsmith()

    cache_key = None
    if use_cache and artifacts_dir is not None:
        cache_key = result_cache.cache_key_for(profile, request, state)
        if cache_key:
            cached = result_cache.load_cached_response(cache_key, _Path(artifacts_dir))
            if cached is not None:
                trace_url = current_trace_url()
                if trace_url:
                    cached.meta.setdefault("trace_url", trace_url)
                cached.meta["cache_hit"] = True
                return cached

    client = None
    root_run = None
    if is_tracing_enabled():
        try:
            from langsmith import Client

            client = Client()
            root_run = client.create_run(
                name="insight.analyze",
                run_type="chain",
                inputs={
                    "question": request.question,
                    "dataset_id": profile.fingerprint.dataset_id,
                    "row_count": profile.row_count,
                    "column_count": profile.column_count,
                },
                start_time=_dt.datetime.now(_dt.timezone.utc),
            )
        except Exception:
            client = None
            root_run = None

    response: AnalysisResponse | None = None
    pipeline_error: Exception | None = None
    try:
        async for event in analyze_stream(
            df, profile, request, artifacts_dir, history, state, datasets
        ):
            if event["type"] == "result":
                response = event["response"]
    except Exception as exc:
        pipeline_error = exc
    finally:
        if client is not None and root_run is not None:
            try:
                outputs = None
                if response is not None:
                    outputs = {
                        "answer_chars": len(response.answer),
                        "model": response.meta.get("model"),
                        "validation_valid": (
                            response.validation.valid if response.validation else None
                        ),
                        "steps": [
                            {"id": s["step_id"], "op": s["operation"], "ok": s["success"]}
                            for s in response.meta.get("steps", [])
                        ],
                        "total_ms": response.meta.get("total_ms"),
                        "tokens_in": response.meta.get("tokens_in"),
                        "tokens_out": response.meta.get("tokens_out"),
                        "plots": len(response.plots),
                    }
                client.update_run(
                    root_run.id,
                    outputs={"output": outputs} if outputs else None,
                    error=(
                        f"{type(pipeline_error).__name__}: {pipeline_error}"
                        if pipeline_error
                        else None
                    ),
                    end_time=_dt.datetime.now(_dt.timezone.utc),
                )
            except Exception:
                pass

    if pipeline_error is not None:
        raise pipeline_error
    if response is None:
        raise InsightError("analysis pipeline produced no result")

    if cache_key and response.error is None and response.validation and response.validation.valid:
        result_cache.store_cached_response(cache_key, response)
        response.meta["cache_hit"] = False

    trace_url = current_trace_url()
    if trace_url:
        response.meta.setdefault("trace_url", trace_url)
    return response


def run_analysis_sync(
    df,
    profile: DatasetProfile,
    request: AnalysisRequest,
    artifacts_dir=None,
    history: Optional[list[tuple[str, str]]] = None,
    state: Optional[AnalysisSessionState] = None,
    use_cache: bool = True,
    datasets: Optional[dict] = None,
) -> AnalysisResponse:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            analyze(df, profile, request, artifacts_dir, history, state, use_cache, datasets)
        )
    finally:
        loop.close()
