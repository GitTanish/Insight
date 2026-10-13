"""Rebuild pipeline objects from an agentic loop outcome.

Everything after planning in the single-shot flow — the validator, the
independent recomputation, chart rendering, the explainer and the grounding
guard — consumes an ``AnalysisPlan`` + ``ExecutionResult``. Rebuilding those
from the loop's tool results means an agentic answer flows through EXACTLY the
same verification machinery as a planned answer, with no second implementation
of the guarantees. Failed tool calls stay out of the plan (the model already
saw the error and recovered, or the run ends); they are kept as a trail in
:func:`agentic_meta` for observability instead.
"""

from __future__ import annotations

from typing import Any, Optional

from insight.domain.analysis import AnalysisPlan, ChartSpec, ChartType, PlanStep
from insight.domain.execution import DataTable, ExecutionResult, OperationOutput, StepResult
from insight.tools.loop import LoopOutcome
from insight.tools.runtime import ToolResult

_MAX_CHARTS = 2

_TABLE_CHART_TITLES = {
    "group_aggregate": "{y} by {x}",
    "value_counts": "Frequency of {x}",
    "compare_subsets": "{y} by subset",
}


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _first_numeric_column(table: DataTable, skip: int = 1) -> Optional[str]:
    for name in table.columns[skip:]:
        for row in table.rows[:5]:
            idx = table.columns.index(name)
            if _is_num(row[idx]):
                return name
    return None


def _chart_for(op_name: str, step_id: int, table: DataTable) -> Optional[ChartSpec]:
    if not table.rows or len(table.columns) < 2:
        return None
    x = table.columns[0]

    if op_name == "distribution":
        return ChartSpec(
            chart_type=ChartType.hist,
            title=f"Distribution of {x}",
            source_step=step_id,
            x=x,
        )
    if op_name == "trend":
        y = _first_numeric_column(table)
        if y is None:
            return None
        return ChartSpec(
            chart_type=ChartType.line,
            title=f"{y} over time",
            source_step=step_id,
            x=x,
            y=y,
        )

    y = _first_numeric_column(table)
    if y is None:
        return None
    template = _TABLE_CHART_TITLES.get(op_name, "{y} by {x}")
    return ChartSpec(
        chart_type=ChartType.bar,
        title=template.format(x=x, y=y),
        source_step=step_id,
        x=x,
        y=y,
    )


def chart_specs_from_results(results: list[ToolResult]) -> list[ChartSpec]:
    """Deterministic editorial charts from the loop's table-producing steps.

    Most recent tables win (the final answer is usually built on them), and
    every spec is validated against the actual table columns downstream, so a
    bad synthesis is rejected by the ordinary validator instead of trusted.
    """
    charts: list[ChartSpec] = []
    for result in reversed(results):
        if not result.ok or result.output is None or result.output.table is None:
            continue
        spec = _chart_for(result.name, result.step_id or 0, result.output.table)
        if spec is not None:
            charts.append(spec)
        if len(charts) >= _MAX_CHARTS:
            break
    return charts


def plan_from_outcome(outcome: LoopOutcome, question: str) -> AnalysisPlan:
    """The rebuilt plan: one step per successful tool call, charts attached."""
    steps = [
        PlanStep(
            step_id=r.step_id,
            operation=r.name,
            params=dict(r.params),
            reason="agentic tool call",
            input_step=r.input_step,
        )
        for r in outcome.tool_results
        if r.ok
    ]
    if not steps:
        raise ValueError("agentic loop produced no successful tool calls")
    return AnalysisPlan(
        objective=f"agentic: {question[:120]}",
        steps=steps,
        charts=chart_specs_from_results(outcome.tool_results),
    )


def execution_from_outcome(outcome: LoopOutcome, df) -> ExecutionResult:
    """The rebuilt execution: successful steps only, in loop order."""
    steps = [
        StepResult(
            step_id=r.step_id if r.step_id is not None else index,
            operation=r.name,
            success=True,
            error=None,
            output=r.output or OperationOutput(),
        )
        for index, r in enumerate(
            (r for r in outcome.tool_results if r.ok), start=0
        )
    ]
    row_count = getattr(df, "__len__", lambda: None)()
    return ExecutionResult(
        success=bool(steps),
        steps=steps,
        final_row_count=row_count,
        duration_ms=0,
    )


def agentic_meta(outcome: LoopOutcome) -> dict:
    """Telemetry merged into the response meta (observability + UI chips)."""
    return {
        "agentic": True,
        "agentic_iterations": outcome.iterations,
        "agentic_llm_calls": outcome.llm_calls,
        "agentic_throttled": outcome.throttled,
        "agentic_ops": outcome.operations_used,
        "agentic_final_summary": (outcome.final_answer or "")[:400],
        "agentic_trail": [
            {
                "operation": r.name,
                "ok": r.ok,
                "step_id": r.step_id,
                "error": None if r.ok else r.content[:200],
            }
            for r in outcome.tool_results
        ],
    }
