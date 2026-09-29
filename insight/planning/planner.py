from __future__ import annotations

from typing import Awaitable, Callable, Optional

from pydantic import ValidationError

from insight.analytics.operations import operation_catalog, validate_plan_columns
from insight.domain.analysis import AnalysisPlan
from insight.domain.dataset import DatasetProfile
from insight.domain.errors import PlanValidationError
from insight.llm.base import LLMRequest
from insight.llm.registry import extract_json
from insight.observability import attach_metadata, trace_span
from insight.planning.prompts import build_planner_messages, build_repair_user_message


class Planner:
    def __init__(
        self,
        generate: Callable[[LLMRequest], Awaitable],
        model_name: str,
        max_repair_attempts: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 3000,
        reasoning_effort: Optional[str] = None,
    ):
        self._generate = generate
        self._model_name = model_name
        self._max_repair_attempts = max_repair_attempts
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort

    @trace_span("planning.plan", run_type="chain")
    async def plan(
        self,
        question: str,
        profile: DatasetProfile,
        history: list[tuple[str, str]] | None = None,
        session_context: str | None = None,
        sql_schema: str | None = None,
    ) -> AnalysisPlan:
        messages = build_planner_messages(
            question=question,
            profile=profile,
            catalog=operation_catalog(),
            history=history or [],
            session_context=session_context,
            sql_schema=sql_schema,
        )

        attempts = 1 + max(0, self._max_repair_attempts)
        issues: list[str] = []

        for attempt in range(attempts):
            request = LLMRequest(
                messages=[{"role": m["role"], "content": m["content"]} for m in messages],
                model=self._model_name,
                temperature=self._temperature,
                json_mode=True,
                max_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                cache_prompt=True,
            )
            response = await self._generate(request)
            raw_content = response.content

            try:
                data = extract_json(raw_content)
            except ValueError as exc:
                issues = [f"response was not valid JSON: {exc}"]
            else:
                try:
                    plan = AnalysisPlan.model_validate(data)
                    plan, column_issues = validate_plan_columns(plan, profile)
                    if not column_issues:
                        attach_metadata(
                            planner_model=self._model_name,
                            planner_attempts=attempt + 1,
                            step_count=len(plan.steps),
                            chart_count=len(plan.charts),
                        )
                        return plan
                    issues = column_issues
                except ValidationError as exc:
                    issues = [
                        f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                        for err in exc.errors()[:8]
                    ]

            messages.append({"role": "assistant", "content": raw_content})
            messages.append({"role": "user", "content": build_repair_user_message(issues)})

        raise PlanValidationError(issues)
