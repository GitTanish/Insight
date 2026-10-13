"""Bounded observe/plan/act/refine agentic loop.

The default single-shot planner asks the model for one complete ``AnalysisPlan``
and executes it. This loop instead lets the model call ops *as tools*, observe
the real computed result, and decide what to do next — the genuinely "agentic"
behaviour — while keeping the same guarantees:

- **Bounded.** ``max_iterations`` caps the number of model turns, and
  ``max_llm_calls`` caps total provider calls per question. A runaway loop
  cannot multiply cost or burn a rate-limited free-tier quota. A call that
fails identically three times in a row aborts the loop instead of spinning.
- **Throttle-aware.** ``call_delay_s`` spaces provider calls so a multi-turn
  loop stays under a provider's per-minute ceiling; a rate-limit failure is
  surfaced as a distinct ``throttled`` outcome rather than a silent wrong
  answer. The provider layer's own ``retry-after`` backoff still applies.
- **DAG-chained.** A tool call may carry ``input_step`` (the id shown in each
  tool result) to run on an earlier step's output table, mirroring the
  single-shot planner's ``input_step`` semantics. Without it the call runs on
  the working row-set, and ``filter_rows`` narrows it for later calls exactly
  like the sequential executor does.
- **Grounded.** Every tool call routes through :func:`insight.tools.runtime.dispatch`,
  i.e. the deterministic op registry plus column validation. The final narrative
  is produced by the *existing* explainer + grounding guard, so nothing about
  the "show your work" contract changes.

The operation-selection step is a pluggable :class:`OperationRouter`. The
default routes through the LLM (parity with today). A System One decision-model
router (Jev/Julia) can later be dropped in to classify the op, cutting LLM
calls — the seam is here so that swap is one line and needs no loop change.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from insight.analytics.operations import OPERATIONS
from insight.domain.dataset import DatasetProfile
from insight.llm.base import LLMRequest, Message, ToolCall
from insight.tools.runtime import ToolCall as RuntimeToolCall
from insight.tools.runtime import ToolResult, dispatch
from insight.tools.tooling import build_anthropic_tools, build_openai_tools

EventSink = Any


@dataclass
class LoopBudget:
    """Hard bounds that keep a multi-turn loop cheap and throttle-safe."""

    max_iterations: int = 5
    max_llm_calls: int = 9
    call_delay_s: float = 0.0


@dataclass
class LoopOutcome:
    """What the loop produced, plus the telemetry the A/B harness scores on."""

    tool_results: list[ToolResult] = field(default_factory=list)
    operations_used: list[str] = field(default_factory=list)
    iterations: int = 0
    llm_calls: int = 0
    throttled: bool = False
    error: Optional[str] = None
    final_answer: str = ""

    @property
    def success(self) -> bool:
        return (
            self.error is None
            and bool(self.tool_results)
            and all(r.ok for r in self.tool_results)
        )


class OperationRouter(Protocol):
    """Chooses the next action given the conversation so far.

    Implementations return tool calls (``{"tool_calls": [ToolCall, ...]}`` or
    ``{"tool_call": ToolCall}``), a final answer (``{"final": str}``), or a
    decision to stop (``{"stop": True}``). Keeping this an interface is what
    lets a System One decision model replace the LLM for op-selection without
    touching the loop.
    """

    async def next_action(self, state: "LoopState") -> dict: ...


@dataclass
class LoopState:
    """Everything a router needs to decide the next action."""

    question: str
    df: Any
    profile: DatasetProfile
    duck_ctx: Any = None
    messages: list[Message] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Step outputs keyed by step id — the chaining substrate (input_step).
    frames: dict[int, Any] = field(default_factory=dict)
    next_step_id: int = 0
    next_call_id: int = 0


class LLMOperationRouter:
    """Default router: the model picks the next op via tool-calling.

    Provider-agnostic across every OpenAI-compatible endpoint (Groq, NIM,
    OpenRouter, OpenAI) and Anthropic (``tool_shape="anthropic"``, matching
    the chain's primary provider). Uses ``tool_choice="auto"`` so the model
    may either call another op or answer directly.
    """

    def __init__(
        self,
        generate,
        model_name: str,
        system_prompt: str,
        tool_shape: str = "openai",
        max_tokens: int = 1500,
    ):
        self._generate = generate
        self._model = model_name
        self._system = system_prompt
        self._max_tokens = max_tokens
        if tool_shape == "anthropic":
            self._tools = build_anthropic_tools()
        else:
            self._tools = build_openai_tools()

    async def next_action(self, state: LoopState) -> dict:
        request = LLMRequest(
            messages=state.messages,
            model=self._model,
            temperature=0.0,
            max_tokens=self._max_tokens,
            tools=self._tools,
            tool_choice="auto",
        )
        response = await self._generate(request)
        tool_calls = getattr(response, "tool_calls", None) or []
        if tool_calls:
            calls = [
                RuntimeToolCall(
                    id=getattr(first, "id", None) or f"call_{state.next_call_id}",
                    name=first.name,
                    arguments=dict(getattr(first, "arguments", {}) or {}),
                )
                for first in tool_calls
            ]
            state.next_call_id += len(calls)
            return {"tool_calls": calls}
        content = (getattr(response, "content", "") or "").strip()
        return {"final": content}


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__
    if name == "RateLimitError":
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate_limit" in text or "tpm" in text


async def run_loop(
    question: str,
    df,
    profile: DatasetProfile,
    generate,
    model_name: str,
    duck_ctx: Any = None,
    budget: Optional[LoopBudget] = None,
    router: Optional[OperationRouter] = None,
    system_prompt: str = "",
    tool_shape: str = "openai",
    on_event: Optional[EventSink] = None,
    initial_messages: Optional[list[Message]] = None,
) -> LoopOutcome:
    """Run the bounded agentic loop and return a :class:`LoopOutcome`.

    ``generate`` is the chain's async generate callable (same seam the planner
    uses), which keeps this testable with the existing scripted-LLM fixtures.
    ``on_event`` — when given — is awaited with a dict for every tool call and
    result, which is how the SSE stream surfaces the agent trail live.
    ``initial_messages`` seeds the transcript (system + context + question);
    when omitted the loop builds it from ``system_prompt`` and ``question``.
    """
    budget = budget or LoopBudget()
    router = router or LLMOperationRouter(
        generate, model_name, system_prompt, tool_shape
    )
    outcome = LoopOutcome()

    if initial_messages is not None:
        state = LoopState(
            question=question,
            df=df,
            profile=profile,
            duck_ctx=duck_ctx,
            messages=list(initial_messages),
        )
    else:
        state = LoopState(
            question=question,
            df=df,
            profile=profile,
            duck_ctx=duck_ctx,
            messages=[Message(role="system", content=system_prompt)] if system_prompt else [],
        )
        state.messages.append(Message(role="user", content=question))

    working = df
    recent_failures: list[str] = []

    for _ in range(max(1, budget.max_iterations)):
        if outcome.llm_calls >= budget.max_llm_calls:
            outcome.error = "llm_call_budget_exhausted"
            break
        outcome.iterations += 1

        if budget.call_delay_s > 0 and outcome.llm_calls > 0:
            await asyncio.sleep(budget.call_delay_s)

        try:
            action = await router.next_action(state)
            outcome.llm_calls += 1
        except Exception as exc:
            if _is_rate_limit(exc):
                outcome.throttled = True
                outcome.error = f"throttled: {exc}"
            else:
                outcome.error = f"{type(exc).__name__}: {exc}"
            break

        raw_calls = action.get("tool_calls") or (
            [action["tool_call"]] if action.get("tool_call") else []
        )
        if not raw_calls:
            outcome.final_answer = action.get("final", "") or ""
            break

        turn_results: list[ToolResult] = []
        assistant_calls: list[dict] = []
        for call in raw_calls:
            step_id = state.next_step_id
            call_id = call.id or f"call_{state.next_call_id}"
            state.next_call_id += 1

            arguments = dict(call.arguments or {})
            input_step = arguments.pop("input_step", None)
            if input_step is not None:
                try:
                    input_step = int(input_step)
                except (TypeError, ValueError):
                    input_step = None

            if input_step is not None and input_step not in state.frames:
                result = ToolResult(
                    id=call_id,
                    name=call.name,
                    ok=False,
                    content=(
                        f"input_step {input_step} produced no table to "
                        f"consume; steps with tables so far: "
                        f"{sorted(state.frames) or 'none'}"
                    ),
                )
            else:
                frame = state.frames[input_step] if input_step is not None else working
                run_call = RuntimeToolCall(id=call_id, name=call.name, arguments=arguments)
                result = dispatch(
                    run_call,
                    frame,
                    profile,
                    duck_ctx,
                    step_id=step_id,
                    validate_columns=input_step is None,
                )
                result.input_step = input_step
                if result.ok and result.frame is not None:
                    state.frames[step_id] = result.frame
                    spec = OPERATIONS.get(call.name)
                    if spec is not None and spec.mutates and input_step is None:
                        working = result.frame
                if result.ok:
                    state.next_step_id += 1

            turn_results.append(result)
            outcome.tool_results.append(result)
            outcome.operations_used.append(call.name)
            state.tool_results.append(result)
            assistant_calls.append(
                {"id": call_id, "name": call.name, "arguments": arguments}
            )

            if on_event is not None:
                await on_event({
                    "type": "tool_call",
                    "name": call.name,
                    "step_id": step_id,
                    "call_id": call_id,
                    "arguments": arguments,
                    "input_step": input_step,
                })
                await on_event({
                    "type": "tool_result",
                    "name": call.name,
                    "step_id": step_id,
                    "call_id": call_id,
                    "ok": result.ok,
                    "preview": (result.content or "")[:240],
                })

            if not result.ok:
                recent_failures.append(json.dumps(result.content, sort_keys=True))
                if len(recent_failures) >= 3 and len(set(recent_failures[-3:])) == 1:
                    outcome.error = "repeated_failing_call"
            else:
                recent_failures.clear()

        state.messages.append(
            Message(role="assistant", content="", tool_calls=assistant_calls)
        )
        for result in turn_results:
            prefix = f"[step {result.step_id}] " if result.step_id is not None else ""
            state.messages.append(
                Message(
                    role="tool",
                    content=prefix + result.content,
                    tool_call_id=result.id,
                    name=result.name,
                )
            )

        if outcome.error:
            break
    else:
        if not outcome.final_answer:
            outcome.error = "iteration_cap_reached"

    return outcome
