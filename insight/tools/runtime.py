"""Dispatch agentic tool calls through the deterministic op registry.

This is the seam that keeps the "LLM never executes code and never invents
numbers" promise intact even inside an agentic loop. A tool call is *not* free
execution: it is validated into the op's Pydantic params model, canonicalised
against the dataset profile (the same :func:`validate_plan_columns` path the
single-shot planner uses), and then run by the exact same ``OperationSpec.run``
the executor uses. The model can only pick a registered op and supply params;
it cannot run arbitrary code.

Results are serialised back into compact markdown so they can be appended as a
``tool`` role message — mirroring what the explainer already receives (a small
preview of each table plus all calculations), never the raw dataframe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd
from pydantic import ValidationError

from insight.analytics.operations import (
    OPERATIONS,
    OperationError,
    validate_plan_columns,
)
from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.domain.dataset import DatasetProfile
from insight.domain.execution import OperationOutput

_MAX_TABLE_ROWS = 20


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Outcome of dispatching one :class:`ToolCall`."""

    id: str
    name: str
    ok: bool
    content: str
    step_id: Optional[int] = None
    # Structured forms retained so the eval harness can score the agentic run
    # with the exact same grader the single-shot harness uses (score_case reads
    # evidence/tables), while ``content`` is what the model actually sees.
    table: Optional[Any] = None
    calculations: list = field(default_factory=list)
    # The validated op output (tables + calculations + notes) so the caller can
    # rebuild an ExecutionResult for the standard validation/verify/render path.
    output: Optional[OperationOutput] = None
    notes: Optional[str] = None
    # Validated params (post column-canonicalisation) for plan reconstruction.
    params: dict[str, Any] = field(default_factory=dict)
    # input_step this call ran on, when the model chained from an earlier table.
    input_step: Optional[int] = None
    # The frame this step produced (table df, or the mutated row-set) so the
    # loop can chain later calls off it.
    frame: Optional[Any] = None


def _table_to_markdown(output: OperationOutput, max_rows: int = _MAX_TABLE_ROWS) -> str:
    if output.table is None:
        return ""
    table = output.table
    columns = [str(c) for c in table.columns]
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, rule]
    for row in table.rows[:max_rows]:
        cells = ["" if v is None else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    shown = min(len(table.rows), max_rows)
    if table.truncated or shown < len(table.rows):
        lines.append(f"_({table.total_rows:,} total rows; showing {shown})_")
    return "\n".join(lines)


def _format_output(name: str, output: OperationOutput) -> str:
    parts: list[str] = []
    if output.table is not None:
        parts.append(_table_to_markdown(output))
    if output.calculations:
        calc_lines = [
            f"- {c.label}: {c.value}" + (f" ({c.detail})" if c.detail else "")
            for c in output.calculations
        ]
        parts.append("Computed values:\n" + "\n".join(calc_lines))
    if output.notes:
        parts.append(f"Notes: {output.notes}")
    body = "\n\n".join(p for p in parts if p)
    return body or f"Operation '{name}' completed with no tabular output."


def dispatch(
    call: ToolCall,
    df: pd.DataFrame,
    profile: DatasetProfile,
    duck_ctx: Any = None,
    step_id: int = 0,
    validate_columns: bool = True,
) -> ToolResult:
    """Validate and run one tool call, returning a :class:`ToolResult`.

    Validation reuses the planner's column-canonicalisation so a tool call that
    names a hallucinated column is rejected with the available-column list,
    exactly like a bad plan step. Chained calls (``input_step``) run on an
    earlier step's output table, whose columns are derived — they are validated
    by the op itself at runtime instead of against the raw profile.
    """
    spec = OPERATIONS.get(call.name)
    if spec is None:
        return ToolResult(
            id=call.id,
            name=call.name,
            ok=False,
            content=(
                f"Unknown operation '{call.name}'. "
                f"Available: {sorted(OPERATIONS)}"
            ),
        )

    arguments = dict(call.arguments or {})
    arguments.pop("input_step", None)
    if validate_columns:
        step = PlanStep(step_id=step_id, operation=call.name, params=arguments)
        plan = AnalysisPlan(objective=f"tool:{call.name}", steps=[step], charts=[])
        try:
            plan, issues = validate_plan_columns(plan, profile)
        except Exception as exc:  # defensive: canonicalisation must never crash the loop
            return ToolResult(
                id=call.id, name=call.name, ok=False,
                content=f"validation error: {type(exc).__name__}: {exc}",
            )
        if issues:
            return ToolResult(
                id=call.id, name=call.name, ok=False,
                content="Invalid arguments: " + "; ".join(issues),
            )
        validated_params = dict(plan.steps[0].params)
    else:
        validated_params = arguments
    validated_params.pop("input_step", None)
    try:
        params = spec.params_model.model_validate(validated_params)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        return ToolResult(
            id=call.id, name=call.name, ok=False,
            content=f"invalid params: {detail}",
        )

    try:
        output = spec.run(df, params, step_id, duck_ctx)
    except OperationError as exc:
        return ToolResult(id=call.id, name=call.name, ok=False, content=f"operation error: {exc}")
    except Exception as exc:  # keep the loop alive on any unexpected failure
        return ToolResult(
            id=call.id, name=call.name, ok=False,
            content=f"{type(exc).__name__}: {exc}",
        )

    frame: Optional[Any] = None
    if output.table is not None:
        frame = output.table.to_dataframe()
    if spec.mutates and spec.mutate is not None:
        try:
            frame = spec.mutate(df, params)
        except Exception:
            frame = None

    return ToolResult(
        id=call.id, name=call.name, ok=True,
        content=_format_output(call.name, output), step_id=step_id,
        table=output.table, calculations=output.calculations,
        output=output, notes=output.notes, params=validated_params,
        frame=frame,
    )

