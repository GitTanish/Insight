from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError

from insight.analytics.operations import OPERATIONS, OperationError
from insight.domain.analysis import AnalysisPlan
from insight.domain.execution import ExecutionResult, StepResult


def execute_plan_sync(df, plan: AnalysisPlan, duck_ctx=None) -> ExecutionResult:
    started = time.monotonic()
    working = df.copy()
    frames: dict[int, object] = {}
    step_results: list[StepResult] = []

    for step in sorted(plan.steps, key=lambda s: s.step_id):
        spec = OPERATIONS.get(step.operation)
        if spec is None:
            step_results.append(
                StepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=False,
                    error=f"unknown operation '{step.operation}'",
                )
            )
            continue

        input_frame = working
        if step.input_step is not None:
            source = frames.get(step.input_step)
            if source is None:
                step_results.append(
                    StepResult(
                        step_id=step.step_id,
                        operation=step.operation,
                        success=False,
                        error=(
                            f"input_step {step.input_step} produced no table to "
                            f"consume (it failed or emitted none)"
                        ),
                    )
                )
                continue
            input_frame = source.copy()

        try:
            params = spec.params_model.model_validate(step.params)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in exc.errors()[:5]
            )
            step_results.append(
                StepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=False,
                    error=f"invalid params: {detail}"[:500],
                )
            )
            continue

        try:
            output = spec.run(input_frame, params, step.step_id, duck_ctx)
            if spec.mutates and spec.mutate is not None:
                mutated = spec.mutate(input_frame, params)
                frames[step.step_id] = mutated.copy()
                if step.input_step is None:
                    working = mutated
            elif output.table is not None:
                frames[step.step_id] = output.table.to_dataframe()
            step_results.append(
                StepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=True,
                    error=None,
                    output=output,
                )
            )
        except OperationError as exc:
            step_results.append(
                StepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=False,
                    error=str(exc),
                )
            )
        except Exception as exc:
            step_results.append(
                StepResult(
                    step_id=step.step_id,
                    operation=step.operation,
                    success=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    return ExecutionResult(
        success=bool(step_results) and all(s.success for s in step_results),
        steps=step_results,
        final_row_count=len(working),
        duration_ms=int((time.monotonic() - started) * 1000),
    )


async def execute_plan(df, plan: AnalysisPlan, duck_ctx=None) -> ExecutionResult:
    return await asyncio.to_thread(execute_plan_sync, df, plan, duck_ctx)
