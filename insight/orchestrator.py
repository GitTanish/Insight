from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

from insight.domain.dataset import DatasetProfile
from insight.domain.errors import (
    AllProvidersFailedError,
    InsightError,
    PlanValidationError,
)
from insight.analytics.operations import validate_plan_columns
from insight.domain.analysis import AnalysisPlan
from insight.domain.query import AnalysisRequest
from insight.domain.visualization import AnalysisResponse, Evidence
from insight.conversation.state import (
    AnalysisSessionState,
    extract_state_from_plan,
    record_result,
    serialize_state_for_planner,
)
from insight.execution.executor import execute_plan
from insight.llm.base import LLMRequest
from insight.llm.registry import build_llm_chain
from insight.observability import (
    attach_metadata,
    configure_langsmith,
    current_trace_url,
    trace_span,
)
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
        self.cached_tokens_in: int = 0
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
                # A tool-calling turn legitimately returns empty content with
                # the invocation in `tool_calls` — that is not an error.
                if not (response.content or "").strip() and not (
                    response.tool_calls or []
                ):
                    hint = ""
                    if (
                        effective.max_tokens
                        and response.completion_tokens == effective.max_tokens
                    ):
                        hint = (
                            f" (hit max_tokens={effective.max_tokens}; the model "
                            "likely spent the whole budget on internal reasoning)"
                        )
                    raise InsightError(
                        f"{provider.name}/{model} returned empty content{hint}"
                    )
                self.last_used = f"{provider.name}/{model}"
                self.llm_calls += 1
                self.llm_latency_ms += response.latency_ms
                self.tokens_in += response.prompt_tokens or 0
                self.tokens_out += response.completion_tokens or 0
                self.cached_tokens_in += response.cached_prompt_tokens or 0
                attach_metadata(
                    llm_provider=provider.name,
                    llm_model=model,
                    llm_latency_ms=response.latency_ms,
                    llm_prompt_tokens=response.prompt_tokens,
                    llm_completion_tokens=response.completion_tokens,
                    llm_cached_prompt_tokens=response.cached_prompt_tokens,
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

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Yield answer text as it arrives, failing over between providers.

        Failover only happens while nothing has been emitted yet: once text
        reaches the user we cannot un-say it, so a mid-stream failure aborts
        instead of splicing two providers together.
        """
        errors: list[Exception] = []
        emitted = False
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
            try:
                chunks: list[str] = []
                async for delta in provider.stream(effective):
                    if delta:
                        chunks.append(delta)
                        emitted = True
                        yield delta
                if chunks:
                    self.last_used = f"{provider.name}/{model}"
                    self.llm_calls += 1
                    return
            except InsightError as exc:
                errors.append(exc)
                if emitted:
                    raise
            except Exception as exc:
                errors.append(InsightError(str(exc)))
                if emitted:
                    raise
        if errors:
            raise AllProvidersFailedError(errors)


def _build_chain_for(
    model_id: Optional[str],
    temperature: float,
    api_key: Optional[str] = None,
    no_fallback: bool = False,
) -> _Chain:
    _, entries = build_llm_chain(
        model_id or get_settings().default_model_id,
        api_key=api_key,
        no_fallback=no_fallback,
    )
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


@dataclass
class _TailCtx:
    """Everything the shared validate→render→explain→verify tail needs."""

    df: Any
    profile: DatasetProfile
    request: AnalysisRequest
    plan: AnalysisPlan
    execution: Any
    validation: Any
    chain: Any
    history: Optional[list[tuple[str, str]]]
    state: Optional[AnalysisSessionState]
    table_datasets: dict
    artifacts_dir: Any
    started: float
    planning_ms: int
    root_client: Any = None
    root_run: Any = None
    cache_key: Optional[str] = None
    cacheable: bool = False
    repair_attempted: bool = False
    repair_ms: int = 0
    repair_attempts: int = 0
    repair_issues: list[str] = field(default_factory=list)
    # Merged into response.meta / appended as evidence warnings.
    extra_meta: dict = field(default_factory=dict)
    extra_warnings: list[str] = field(default_factory=list)


async def _tail_events(ctx: _TailCtx) -> AsyncIterator[dict]:
    """Shared pipeline tail: state, charts, explainer, grounding, response.

    Identical for single-shot and agentic mode — the point is that an agentic
    answer is validated, recomputed, rendered, narrated and grounded by the
    exact same code path as a planned one.
    """
    from pathlib import Path as _PathDir

    from insight.observability import finish_root_run, trace_phase

    plan = ctx.plan
    execution = ctx.execution
    validation = ctx.validation
    chain = ctx.chain
    profile = ctx.profile
    request = ctx.request
    settings = get_settings()
    table_datasets = ctx.table_datasets

    if ctx.artifacts_dir is not None:
        ctx.artifacts_dir = _PathDir(ctx.artifacts_dir)
    else:
        # Library callers may omit artifacts_dir; charts still need a directory.
        import tempfile

        ctx.artifacts_dir = _PathDir(tempfile.mkdtemp(prefix="insight_artifacts_"))

    analysis_state = extract_state_from_plan(plan, base=ctx.state)
    if profile.fingerprint.dataset_id:
        analysis_state.dataset_id = profile.fingerprint.dataset_id

    key_facts = [
        f"{c.label}: {_cell(c.value)}"
        for c in execution.all_calculations()[:12]
        if c.value not in (None, "")
    ][:3]
    record_result(
        analysis_state,
        question=request.question,
        operations=[s.operation for s in execution.steps if s.success],
        key_facts=key_facts,
    )
    yield {"type": "state", "state": analysis_state.model_dump(mode="json")}

    try:
        from insight.agentic import suggest_followups

        followups = suggest_followups(
            profile, state=analysis_state, tables=table_datasets
        )
    except Exception:
        followups = []
    yield {"type": "suggestions", "followups": followups}

    warnings_text = "\n".join(
        validation.warning_messages() + ctx.extra_warnings
    )

    yield {"type": "stage", "key": "rendering", "label": "Rendering editorial charts..."}
    t_charts = time.monotonic()
    ctx.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with trace_phase("visualization.render_charts", charts=len(plan.charts)):
        plots, chart_warnings = render_charts(
            plan.charts, execution.tables_by_step(), ctx.artifacts_dir, EDITORIAL_THEME
        )
    charts_ms = int((time.monotonic() - t_charts) * 1000)
    figures_text = "\n".join(
        f"{i}. {plot.chart_type} — \"{plot.title}\""
        for i, plot in enumerate(plots, start=1)
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
                warnings_text="\n".join(
                    filter(None, [warnings_text, "\n".join(chart_warnings)])
                ),
                figures_text=figures_text,
            )
        ],
        model=chain.primary_model,
        temperature=None,
        max_tokens=settings.explainer_max_tokens,
        reasoning_effort=chain.reasoning_effort,
    )

    t_explain = time.monotonic()
    answer_chunks: list[str] = []
    first_token_ms: Optional[int] = None
    with trace_phase("explainer.compose"):
        try:
            async for delta in chain.stream(explainer_request):
                if not delta:
                    continue
                if first_token_ms is None:
                    first_token_ms = int((time.monotonic() - t_explain) * 1000)
                answer_chunks.append(delta)
                yield {"type": "delta", "text": delta}
            answer = "".join(answer_chunks).strip()
            if not answer:
                answer = _fallback_answer(validation, execution)
        except InsightError:
            answer = _fallback_answer(validation, execution)
    explaining_ms = int((time.monotonic() - t_explain) * 1000)

    unverified_figures: list[str] = []
    grounding_retry_ms = 0
    if settings.answer_grounded_check and answer and answer != _fallback_answer(validation, execution):
        from insight.validation.grounding import (
            build_grounding_retry_message,
            find_ungrounded_numbers,
        )

        grounding_notes = "; ".join(
            filter(None, [getattr(s.output, "notes", None) for s in execution.steps if s.success])
        )
        unverified_figures = find_ungrounded_numbers(
            answer,
            execution.all_calculations(),
            list(execution.tables_by_step().values()),
            notes=grounding_notes,
            warnings=validation.warning_messages() + chart_warnings,
        )
        if unverified_figures:
            yield {
                "type": "stage",
                "key": "grounding",
                "label": "Verifying cited numbers...",
                "level": "warn",
            }
            retry_messages = [
                {"role": m["role"], "content": m["content"]}
                for m in build_explainer_messages(
                    question=request.question,
                    objective=plan.objective,
                    tables_markdown=_tables_markdown(execution),
                    calculations_text=_calculations_text(execution),
                    warnings_text="\n".join(
                        filter(None, [warnings_text, "\n".join(chart_warnings)])
                    ),
                    figures_text=figures_text,
                )
            ]
            retry_messages.append({"role": "assistant", "content": answer})
            retry_messages.append({
                "role": "user",
                "content": build_grounding_retry_message(unverified_figures),
            })
            try:
                t_ground = time.monotonic()
                yield {"type": "delta_reset"}
                rewritten: list[str] = []
                async for delta in chain.stream(LLMRequest(
                    messages=retry_messages,
                    model=chain.primary_model,
                    temperature=None,
                    max_tokens=settings.explainer_max_tokens,
                    reasoning_effort=chain.reasoning_effort,
                )):
                    if delta:
                        rewritten.append(delta)
                        yield {"type": "delta", "text": delta}
                retried_answer = "".join(rewritten).strip()
                if not retried_answer:
                    raise InsightError("empty rewrite")
                still_bad = find_ungrounded_numbers(
                    retried_answer,
                    execution.all_calculations(),
                    list(execution.tables_by_step().values()),
                    notes=grounding_notes,
                    warnings=validation.warning_messages() + chart_warnings,
                )
                answer = retried_answer
                unverified_figures = still_bad
            except InsightError:
                pass
            finally:
                grounding_retry_ms = int((time.monotonic() - t_ground) * 1000)

    evidence: list[Evidence] = [
        Evidence(kind="calculation", label=c.label, value=_cell(c.value), detail=c.detail)
        for c in execution.all_calculations()[:12]
    ]
    for message in validation.warning_messages() + ctx.extra_warnings:
        evidence.append(Evidence(kind="warning", label=message))
    for message in chart_warnings:
        evidence.append(Evidence(kind="warning", label=message))
    if unverified_figures:
        evidence.append(Evidence(
            kind="warning",
            label=(
                "answer contained figures not found in computed results: "
                + ", ".join(unverified_figures[:6])
            ),
        ))

    meta: dict = {
        "model": chain.last_used,
        "objective": plan.objective,
        "repair_attempted": ctx.repair_attempted,
        "planning_ms": ctx.planning_ms,
        "repair_ms": ctx.repair_ms,
        "repair_attempts": ctx.repair_attempts,
        "repair_issues": list(ctx.repair_issues)[:8],
        "charts_ms": charts_ms,
        "cached_prompt_tokens": chain.cached_tokens_in,
        "prompt_tokens": chain.tokens_in,
        "completion_tokens": chain.tokens_out,
        "grounding_retry_ms": grounding_retry_ms,
        "first_token_ms": first_token_ms,
        "explaining_ms": explaining_ms,
        "execution_ms": execution.duration_ms,
        "total_ms": int((time.monotonic() - ctx.started) * 1000),
        "tokens_in": chain.tokens_in,
        "tokens_out": chain.tokens_out,
        "llm_calls": chain.llm_calls,
        "llm_latency_ms": chain.llm_latency_ms,
        "dataset_id": profile.fingerprint.dataset_id,
        "analysis_state": analysis_state.model_dump(mode="json"),
        "followups": followups,
        "unverified_figures": unverified_figures,
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
    meta.update(ctx.extra_meta)

    final_response = AnalysisResponse(
        question=request.question,
        answer=answer,
        evidence=evidence,
        tables=list(execution.tables_by_step().values()),
        plots=plots,
        validation=validation,
        meta=meta,
    )

    if ctx.cacheable and final_response.error is None and validation.valid:
        from insight import cache as result_cache

        result_cache.store_cached_response(
            ctx.cache_key or result_cache.cache_key_for(
                profile, request, ctx.state
            ), final_response
        )

    trace_url = current_trace_url()
    if trace_url:
        final_response.meta.setdefault("trace_url", trace_url)
    finish_root_run(
        ctx.root_client,
        ctx.root_run,
        outputs={
            "answer_chars": len(final_response.answer),
            "model": final_response.meta.get("model"),
            "validation_valid": validation.valid,
            "cache_hit": bool(final_response.meta.get("cache_hit")),
            "agentic": bool(final_response.meta.get("agentic")),
            "repair_attempts": ctx.repair_attempts,
            "repair_issues": list(ctx.repair_issues)[:8],
            "unverified_figures": unverified_figures,
            "followups": followups,
            "latency_ms": {
                "planning": ctx.planning_ms,
                "repair": ctx.repair_ms,
                "execution": execution.duration_ms,
                "charts": charts_ms,
                "explaining": explaining_ms,
                "grounding_retry": grounding_retry_ms,
                "total": int((time.monotonic() - ctx.started) * 1000),
            },
            "tokens_in": final_response.meta.get("tokens_in"),
            "tokens_out": final_response.meta.get("tokens_out"),
            "plots": len(plots),
        },
    )

    yield {
        "type": "result",
        "response": final_response,
    }


def _agentic_error_answer(outcome) -> str:
    """Explain an aborted agentic run without inventing anything."""
    if outcome.throttled:
        return (
            "The provider rate-limited the agent mid-analysis, so it stopped "
            "early — no answer was invented. Wait a moment and retry, switch "
            "models, or add a fallback provider key."
        )
    if outcome.error == "llm_call_budget_exhausted":
        return (
            "The agent reached its per-question tool-call budget before it had "
            "enough evidence. Try a more specific question, or raise "
            "`INSIGHT_AGENTIC_MAX_LLM_CALLS`."
        )
    if outcome.error == "iteration_cap_reached":
        return (
            "The agent ran out of allowed turns while gathering evidence — no "
            "answer was invented. Rephrase the question, or raise "
            "`INSIGHT_AGENTIC_MAX_ITERATIONS`."
        )
    if outcome.error == "repeated_failing_call":
        return (
            "The agent kept making the same failing tool call and was stopped. "
            "Rephrase using the exact column names from the dataset profile."
        )
    if outcome.error and not outcome.tool_results:
        return (
            f"The agent stopped before answering: {outcome.error}. "
            "No answer was invented — check your provider key/quota and retry."
        )
    if not outcome.tool_results:
        return (
            "The agent finished without calling any analysis tools, so there "
            "is no verified evidence to answer from."
        )
    return (
        f"The agent's tool calls all failed ({outcome.error or 'unknown'}). "
        "No answer was invented — check the column names in the profile and retry."
    )


async def analyze_stream(
    df,
    profile: DatasetProfile,
    request: AnalysisRequest,
    artifacts_dir=None,
    history: Optional[list[tuple[str, str]]] = None,
    state: Optional[AnalysisSessionState] = None,
    datasets: Optional[dict] = None,
    use_cache: bool = True,
) -> AsyncIterator[dict]:
    configure_langsmith()
    settings = get_settings()
    started = time.monotonic()
    if artifacts_dir is not None:
        from pathlib import Path as _PathDir

        artifacts_dir = _PathDir(artifacts_dir)

    table_datasets = dict(datasets) if datasets else {profile.name: df}
    cache_key = None

    from insight.observability import (
        finish_root_run,
        start_root_run,
        trace_phase,
    )

    root_client, root_run = start_root_run(
        "insight.analyze",
        {
            "question": request.question,
            "dataset_id": profile.fingerprint.dataset_id,
            "row_count": profile.row_count,
            "column_count": profile.column_count,
            "streaming": True,
            "model": request.model_id,
        },
    )

    # A user-edited plan is request-specific and must not be served from, or
    # written to, the shared cache.
    cacheable = (
        use_cache
        and artifacts_dir is not None
        and settings.query_cache_enabled
        and not request.plan_only
        and not settings.plan_approval
        and request.approved_plan is None
    )

    if cacheable:
        from insight import cache as result_cache

        cache_key = result_cache.cache_key_for(profile, request, state)
        if cache_key:
            cached = result_cache.load_cached_response(cache_key, artifacts_dir)
            if cached is not None:
                cached.meta["cache_hit"] = True
                cached.meta["total_ms"] = int((time.monotonic() - started) * 1000)
                yield {"type": "stage", "key": "planning", "label": "Restoring cached answer..."}
                yield {"type": "state", "state": (state.model_dump(mode="json") if state else {})}
                yield {"type": "suggestions", "followups": cached.meta.get("followups", [])}
                yield {"type": "result", "response": cached}
                finish_root_run(
                    root_client, root_run,
                    outputs={"cache_hit": True, "question": request.question,
                             "total_ms": cached.meta.get("total_ms")},
                )
                return

    sql_schema_text: Optional[str] = None
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
        api_key=request.api_key,
        no_fallback=request.no_fallback,
    )
    attach_metadata(
        dataset_id=profile.fingerprint.dataset_id,
        requested_model=request.model_id,
        row_count=profile.row_count,
        column_count=profile.column_count,
    )

    agentic = request.agentic if request.agentic is not None else settings.agentic
    # Human-in-the-loop wins: the approval gate needs a reviewable plan *before*
    # anything executes, which only the single-shot planner can produce.
    if request.plan_only or settings.plan_approval:
        agentic = False
    if agentic:
        # Agentic mode: the LLM calls engine ops as tools and iterates, then
        # the SAME tail (validate → recompute → charts → explainer → grounding)
        # runs on the rebuilt plan/execution.
        from insight.analytics.operations import operation_catalog
        from insight.analytics.sql_engine import DuckSession
        from insight.planning.prompts import build_agentic_messages
        from insight.tools.loop import LoopBudget, run_loop
        from insight.tools.pipeline import (
            agentic_meta,
            execution_from_outcome,
            plan_from_outcome,
        )

        yield {
            "type": "stage",
            "key": "planning",
            "label": "Agent is investigating (tool-calling mode)...",
        }

        tool_shape = "anthropic" if (
            getattr(chain, "entries", None)
            and chain.entries[0][0].name == "anthropic"
        ) else "openai"
        budget = LoopBudget(
            max_iterations=settings.agentic_max_iterations,
            max_llm_calls=settings.agentic_max_llm_calls,
            call_delay_s=settings.agentic_call_delay_s,
        )
        initial_messages = build_agentic_messages(
            request.question,
            profile,
            operation_catalog(),
            sql_schema=sql_schema_text,
            history=history,
            session_context=session_context,
        )

        queue: asyncio.Queue = asyncio.Queue()

        async def _emit(event: dict) -> None:
            await queue.put(event)

        session = (
            DuckSession(table_datasets) if settings.sql_enabled else None
        )
        t_agentic = time.monotonic()
        with trace_phase("agentic.run_loop", question=request.question):
            task = asyncio.create_task(run_loop(
                request.question,
                df,
                profile,
                chain.generate,
                chain.primary_model,
                duck_ctx=session,
                budget=budget,
                tool_shape=tool_shape,
                on_event=_emit,
                initial_messages=initial_messages,
            ))
            try:
                while not task.done() or not queue.empty():
                    if queue.empty():
                        await asyncio.sleep(0.02)
                        continue
                    yield await queue.get()
                outcome = await task
            finally:
                if not task.done():
                    task.cancel()
                if session is not None:
                    session.close()
        planning_ms = int((time.monotonic() - t_agentic) * 1000)

        ok_results = [r for r in outcome.tool_results if r.ok]
        if outcome.error is not None or not ok_results:
            yield {
                "type": "stage",
                "key": "planning_failed",
                "label": "Agent could not complete the analysis",
                "level": "error",
            }
            finish_root_run(
                root_client, root_run,
                error=outcome.error or "no_successful_tool_calls",
            )
            yield {
                "type": "result",
                "response": AnalysisResponse(
                    question=request.question,
                    answer=_agentic_error_answer(outcome),
                    meta={
                        "model": chain.last_used,
                        "planning_ms": planning_ms,
                        **agentic_meta(outcome),
                    },
                    error=outcome.error or "no_tools_called",
                ),
            }
            return

        try:
            agentic_plan = plan_from_outcome(outcome, request.question)
            agentic_execution = execution_from_outcome(outcome, df)
        except ValueError as exc:
            yield {
                "type": "result",
                "response": AnalysisResponse(
                    question=request.question,
                    answer=(
                        "The agent's tool calls could not be turned into an "
                        f"analysis: {exc}"
                    ),
                    meta={"model": chain.last_used, **agentic_meta(outcome)},
                    error="agentic_plan_rebuild_failed",
                ),
            }
            return

        yield {
            "type": "plan_preview",
            "objective": agentic_plan.objective,
            "steps": [
                {"id": s.step_id, "operation": s.operation}
                for s in agentic_plan.steps
            ],
        }

        extra_warnings = [
            f"agent tool call '{r.name}' failed (recovered or abandoned): "
            f"{r.content[:160]}"
            for r in outcome.tool_results
            if not r.ok
        ][:3]

        with trace_phase("validation.validate_execution"):
            validation = validate_execution(
                agentic_plan, agentic_execution, df=df, profile=profile
            )
        yield {
            "type": "validation",
            "valid": validation.valid,
            "warning_count": len(validation.warnings),
        }

        async for event in _tail_events(_TailCtx(
            df=df,
            profile=profile,
            request=request,
            plan=agentic_plan,
            execution=agentic_execution,
            validation=validation,
            chain=chain,
            history=history,
            state=state,
            table_datasets=table_datasets,
            artifacts_dir=artifacts_dir,
            started=started,
            planning_ms=planning_ms,
            root_client=root_client,
            root_run=root_run,
            cache_key=cache_key,
            cacheable=cacheable,
            extra_meta=agentic_meta(outcome),
            extra_warnings=extra_warnings,
        )):
            yield event
        return

    planner = Planner(
        generate=chain.generate,
        model_name=chain.primary_model,
        max_repair_attempts=settings.max_repair_attempts,
        max_tokens=settings.planner_max_tokens,
        reasoning_effort=chain.reasoning_effort,
    )

    repair_attempted = False
    repair_ms = 0
    repair_attempts = 0
    repair_issues: list[str] = []

    yield {"type": "stage", "key": "planning", "label": "Planning investigation..."}

    try:
        t_plan = time.monotonic()
        if request.approved_plan is not None:
            # The user reviewed (and possibly edited) this plan. Re-validate it
            # against the current schema instead of trusting it blindly.
            with trace_phase("planning.plan.approved", question=request.question):
                plan, approved_issues = validate_plan_columns(
                    AnalysisPlan.model_validate(request.approved_plan), profile
                )
            if approved_issues:
                raise PlanValidationError(approved_issues)
            planning_ms = int((time.monotonic() - t_plan) * 1000)
        else:
            with trace_phase("planning.plan.stream", question=request.question):
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
        finish_root_run(
            root_client, root_run,
            error=f"plan_validation_failed: {'; '.join(exc.issues[:4])[:200]}",
        )
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

    if request.plan_only or settings.plan_approval:
        # Human-in-the-loop: hand the plan to the user for review/edit and stop
        # before any data is executed.
        plan_payload = plan.model_dump(mode="json")
        yield {
            "type": "plan_review",
            "plan": plan_payload,
            "question": request.question,
        }
        finish_root_run(
            root_client, root_run,
            outputs={"plan_awaiting_approval": True, "step_count": len(plan.steps)},
        )
        return

    yield {
        "type": "stage",
        "key": "executing",
        "label": f"Executing {len(plan.steps)} deterministic step(s)...",
    }
    with trace_phase(
        "execution.execute_plan",
        steps=[s.operation for s in plan.steps],
        step_count=len(plan.steps),
    ):
        execution = await _execute_with_sql(plan)
    with trace_phase("validation.validate_execution"):
        validation = validate_execution(plan, execution, df=df, profile=profile)
    yield {
        "type": "validation",
        "valid": validation.valid,
        "warning_count": len(validation.warnings),
    }

    if not validation.valid and settings.max_repair_attempts > 0:
        repair_attempted = True
        repair_issues: list[str] = list(validation.error_messages())[:8]
        yield {
            "type": "stage",
            "key": "repairing",
            "label": "Validator rejected results â€” repairing plan once...",
            "level": "warn",
        }
        repair_summary = "; ".join(validation.error_messages()[:8])
        repair_question = (
            f"{request.question}\n\n(A previous attempt failed: {repair_summary}. "
            "Produce a corrected plan avoiding these problems.)"
        )
        try:
            t_repair = time.monotonic()
            repaired_plan = await planner.plan(
                repair_question,
                profile,
                history=history,
                session_context=session_context,
                sql_schema=sql_schema_text,
            )
            repair_attempts = 1
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
            repair_attempts = 1
        finally:
            repair_ms += int((time.monotonic() - t_repair) * 1000)

    async for event in _tail_events(_TailCtx(
        df=df,
        profile=profile,
        request=request,
        plan=plan,
        execution=execution,
        validation=validation,
        chain=chain,
        history=history,
        state=state,
        table_datasets=table_datasets,
        artifacts_dir=artifacts_dir,
        started=started,
        planning_ms=planning_ms,
        root_client=root_client,
        root_run=root_run,
        cache_key=cache_key,
        cacheable=cacheable,
        repair_attempted=repair_attempted,
        repair_ms=repair_ms,
        repair_attempts=repair_attempts,
        repair_issues=repair_issues,
    )):
        yield event
    return


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

    if request.plan_only or get_settings().plan_approval:
        raise InsightError(
            "plan review requires the streaming endpoint; "
            "call analyze_stream() and read the 'plan_review' event"
        )
    if request.approved_plan is not None and request.plan_only:
        raise InsightError("plan_only and approved_plan are mutually exclusive")

    cache_key = None
    cacheable = (
        use_cache
        and artifacts_dir is not None
        and request.approved_plan is None
    )
    if cacheable:
        cache_key = result_cache.cache_key_for(profile, request, state)
        if cache_key:
            cached = result_cache.load_cached_response(cache_key, _Path(artifacts_dir))
            if cached is not None:
                trace_url = current_trace_url()
                if trace_url:
                    cached.meta.setdefault("trace_url", trace_url)
                cached.meta["cache_hit"] = True
                return cached
    response: AnalysisResponse | None = None
    pipeline_error: Exception | None = None
    try:
        async for event in analyze_stream(
            df, profile, request, artifacts_dir, history, state, datasets, use_cache
        ):
            if event["type"] == "result":
                response = event["response"]
    except Exception as exc:
        pipeline_error = exc

    if pipeline_error is not None:
        raise pipeline_error
    if response is None:
        raise InsightError("analysis pipeline produced no result")


    if (
        cacheable
        and cache_key
        and response.error is None
        and response.validation
        and response.validation.valid
    ):
        # The streaming path may already have persisted this response under
        # the same key; only write when it did not (e.g. it ran with cache
        # writes disabled) so a miss is stored exactly once.
        if result_cache.load_cached_response(cache_key, _Path(artifacts_dir)) is None:
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
