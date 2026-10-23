"""Agent-facing toolkit: deterministic analytics + claim verification as a library.

This module is the dependency-light seam that lets external agents (MCP clients,
cron workers, other coding agents) use Insight's engine *without* the LLM
planner, while keeping the trust guarantees of the main product:

- every operation runs through the same ``OPERATIONS`` registry, Pydantic param
  models and column canonicalisation the webapp/orchestrator use;
- every run is recorded, so a later ``verify_claims`` call can independently
  recompute the headline aggregates (``independently_verify``) and trace every
  number in an arbitrary narrative back to computed evidence
  (``find_ungrounded_numbers``).

No LLM and no network access are involved anywhere in this module.
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import ValidationError

from insight.analytics.operations import (
    CHAINABLE_OPS,
    OPERATIONS,
    operation_catalog,
    validate_plan_columns,
)
from insight.analytics.sql_engine import DuckSession, sanitize_table_name
from insight.domain.analysis import AnalysisPlan, PlanStep
from insight.domain.dataset import DatasetProfile
from insight.domain.execution import (
    Calculation,
    DataTable,
    ExecutionResult,
    StepResult,
)
from insight.domain.visualization import ValidationIssue
from insight.ingest import EXCEL_EXTENSIONS, parse_csv_content, parse_tables
from insight.profiling.profiler import profile_dataframe
from insight.tools.runtime import ToolCall, dispatch
from insight.validation.grounding import find_ungrounded_numbers
from insight.validation.verification import independently_verify


class ToolkitError(Exception):
    """Invalid toolkit input; the message is safe to return to a caller."""


@dataclass
class _RunRecord:
    run_id: str
    table: str
    plan: AnalysisPlan
    execution: ExecutionResult
    df: pd.DataFrame


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, bool) or value is None:
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def _calc_payload(calc: Calculation) -> dict:
    return {"label": calc.label, "value": _jsonable(calc.value), "detail": calc.detail}


def _table_payload(table: Optional[DataTable]) -> Optional[dict]:
    if table is None:
        return None
    return {
        "name": table.name,
        "columns": [str(c) for c in table.columns],
        "rows": [[_jsonable(cell) for cell in row] for row in table.rows],
        "total_rows": table.total_rows,
        "truncated": table.truncated,
    }


def _issue_payload(issue) -> dict:
    return {
        "severity": issue.severity,
        "message": issue.message,
        "step_id": issue.step_id,
    }


class AnalysisToolkit:
    """Stateful, single-user session: load tables, run ops, verify claims.

    All mutating methods take an internal lock, so a toolkit instance is safe
    to share between the concurrent tool calls of one MCP client.

    ``file_roots`` restricts ``load_table(path=...)`` to the given directories
    (``None`` = unrestricted, the webapp/test default; ``[]`` = deny all file
    loads). ``max_file_bytes`` caps the size read from disk.
    """

    def __init__(
        self,
        *,
        file_roots: Optional[list[str | Path]] = None,
        max_file_bytes: Optional[int] = None,
        max_runs: int = 100,
        max_tables: int = 64,
    ) -> None:
        self._tables: dict[str, pd.DataFrame] = {}
        self._profiles: dict[str, DatasetProfile] = {}
        self._runs: "OrderedDict[str, _RunRecord]" = OrderedDict()
        self._last_run_id: Optional[str] = None
        self._lock = threading.Lock()
        self._file_roots = (
            None
            if file_roots is None
            else [Path(root).expanduser().resolve() for root in file_roots]
        )
        self._max_file_bytes = max_file_bytes
        self._max_runs = max(1, int(max_runs))
        self._max_tables = max(1, int(max_tables))

    # ------------------------------------------------------------------ data

    def load_table(
        self,
        name: str,
        content: Optional[str] = None,
        path: Optional[str] = None,
    ) -> dict:
        """Register a table from inline CSV text (``content``) or a file path.

        Paths ending in .xlsx/.xls/.xlsm are expanded to one table per sheet.
        Returns the loaded table summaries. Re-loading a name replaces it.
        """
        with self._lock:
            if (content is None) == (path is None):
                raise ToolkitError("provide exactly one of 'content' (CSV text) or 'path'")
            table_name = sanitize_table_name(str(name or ""))
            if content is not None:
                raw = content.encode("utf-8")
                frames, err = self._parse_single(table_name, raw)
            else:
                source = self._resolve_source(str(path))
                raw = self._read_source(source)
                if source.suffix.lower() in EXCEL_EXTENSIONS:
                    frames, err = parse_tables(source.name, raw)
                else:
                    frames, err = self._parse_single(table_name, raw)
            if frames is None:
                raise ToolkitError(err or "could not parse input")

            new_tables = [tname for tname in frames if tname not in self._tables]
            if len(self._tables) + len(new_tables) > self._max_tables:
                raise ToolkitError(
                    f"too many tables (max {self._max_tables}); reload existing "
                    "names instead of adding new ones"
                )

            summaries = []
            for tname, df in frames.items():
                self._tables[tname] = df
                self._profiles[tname] = profile_dataframe(df, tname)
                summaries.append(self._summary(tname))
            return {"loaded": summaries}

    def _resolve_source(self, path_value: str) -> Path:
        source = Path(path_value).expanduser()
        try:
            resolved = source.resolve(strict=True)
        except OSError:
            raise ToolkitError(f"file not found: {source}")
        if not resolved.is_file():
            raise ToolkitError(f"not a file: {source}")
        if self._file_roots is not None and not any(
            resolved == root or resolved.is_relative_to(root)
            for root in self._file_roots
        ):
            raise ToolkitError(
                "path is outside the allowed roots: "
                + ", ".join(str(root) for root in self._file_roots)
                if self._file_roots
                else "file loading is disabled (no file roots configured)"
            )
        if self._max_file_bytes is not None:
            try:
                size = resolved.stat().st_size
            except OSError:
                raise ToolkitError(f"file not readable: {source}")
            if size > self._max_file_bytes:
                raise ToolkitError(
                    f"file exceeds the {self._max_file_bytes} byte limit: {source}"
                )
        return resolved

    def _read_source(self, source: Path) -> bytes:
        """Read at most max_file_bytes from the already-resolved path.

        The size check is repeated on the open handle so a file growing between
        ``stat`` and ``read`` cannot bypass the cap.
        """
        if self._max_file_bytes is None:
            return source.read_bytes()
        with source.open("rb") as handle:
            data = handle.read(self._max_file_bytes + 1)
        if len(data) > self._max_file_bytes:
            raise ToolkitError(
                f"file exceeds the {self._max_file_bytes} byte limit: {source}"
            )
        return data

    @staticmethod
    def _parse_single(
        name: str, raw: bytes
    ) -> tuple[Optional[dict[str, pd.DataFrame]], Optional[str]]:
        df, err = parse_csv_content(raw)
        if df is None:
            return None, err
        return {name: df}, None

    def _summary(self, name: str) -> dict:
        df = self._tables[name]
        return {
            "name": name,
            "rows": len(df),
            "columns": [str(c) for c in df.columns],
        }

    def list_tables(self) -> dict:
        with self._lock:
            return {"tables": [self._summary(name) for name in self._tables]}

    def profile_table(self, name: str) -> dict:
        with self._lock:
            profile = self._profile(name)
            return profile.model_dump(mode="json")

    def list_operations(self) -> dict:
        return {"catalog": operation_catalog(), "names": sorted(OPERATIONS)}

    # ------------------------------------------------------------------- ops

    def execute_operation(
        self, table: str, operation: str, params: Optional[dict] = None
    ) -> dict:
        """Run one registered operation against a loaded table."""
        with self._lock:
            df = self._table(table)
            profile = self._profile(table)
            duck = self._duck_for(operation)
            try:
                result = dispatch(
                    ToolCall(id="toolkit-op", name=operation, arguments=dict(params or {})),
                    df,
                    profile,
                    duck_ctx=duck,
                    step_id=0,
                )
            finally:
                if duck is not None:
                    duck.close()
            if not result.ok:
                raise ToolkitError(result.content)

            step = PlanStep(step_id=0, operation=operation, params=result.params)
            plan = AnalysisPlan(objective=f"tool:{operation}", steps=[step])
            execution = _single_step_execution(operation, result.output, result.frame, df)
            run_id = self._record(table, plan, execution, df)
            return {
                "run_id": run_id,
                "operation": operation,
                "calculations": [_calc_payload(c) for c in result.calculations],
                "table": _table_payload(result.output.table if result.output else None),
                "notes": result.notes,
            }

    def run_sql(self, sql: str, tables: Optional[list[str]] = None) -> dict:
        """Run one read-only SELECT/WITH statement through the SQL sandbox."""
        with self._lock:
            selected = list(tables) if tables else list(self._tables)
            if not selected:
                raise ToolkitError("no tables loaded")
            for name in selected:
                if name not in self._tables:
                    raise ToolkitError(
                        f"unknown table '{name}'. loaded: {sorted(self._tables)}"
                    )
            base = selected[0]
            df = self._tables[base]
            profile = self._profiles[base]
            duck = DuckSession({name: self._tables[name] for name in selected})
            try:
                result = dispatch(
                    ToolCall(id="toolkit-sql", name="sql_query", arguments={"query": sql}),
                    df,
                    profile,
                    duck_ctx=duck,
                    step_id=0,
                )
            finally:
                duck.close()
            if not result.ok:
                raise ToolkitError(result.content)

            step = PlanStep(step_id=0, operation="sql_query", params=result.params)
            plan = AnalysisPlan(objective="tool:sql_query", steps=[step])
            execution = _single_step_execution("sql_query", result.output, result.frame, df)
            run_id = self._record(base, plan, execution, df)
            return {
                "run_id": run_id,
                "operation": "sql_query",
                "calculations": [_calc_payload(c) for c in result.calculations],
                "table": _table_payload(result.output.table if result.output else None),
                "notes": result.notes,
            }

    def run_plan(self, table: str, plan: Any) -> dict:
        """Validate and execute a full AnalysisPlan (DAG) against a table.

        ``plan`` may be a JSON string or a dict shaped like the planner's
        ``AnalysisPlan`` schema. A plan that validates but fails at execution
        returns ``success: false`` with per-step errors (it is not an error to
        call this); only structurally invalid plans raise ToolkitError.
        """
        from insight.execution.executor import execute_plan_sync

        with self._lock:
            df = self._table(table)
            profile = self._profile(table)
            parsed = _parse_plan(plan)
            parsed, issues = _validate_plan_for_table(parsed, profile)
            if issues:
                raise ToolkitError("plan validation failed: " + "; ".join(issues))

            needs_sql = any(s.operation == "sql_query" for s in parsed.steps)
            duck = DuckSession(dict(self._tables)) if needs_sql and self._tables else None
            try:
                execution = execute_plan_sync(df, parsed, duck_ctx=duck)
            finally:
                if duck is not None:
                    duck.close()

            run_id = self._record(table, parsed, execution, df)
            return {
                "run_id": run_id,
                "success": execution.success,
                "final_row_count": execution.final_row_count,
                "duration_ms": execution.duration_ms,
                "failed_messages": execution.failed_messages(),
                "steps": [
                    {
                        "step_id": s.step_id,
                        "operation": s.operation,
                        "success": s.success,
                        "error": s.error,
                        "calculations": [
                            _calc_payload(c) for c in (s.output.calculations if s.output else [])
                        ],
                        "table": _table_payload(s.output.table if s.output else None),
                        "notes": s.output.notes if s.output else None,
                    }
                    for s in execution.steps
                ],
            }

    # ------------------------------------------------------------ verification

    def verify_claims(
        self,
        answer: str,
        run_id: Optional[str] = None,
        evidence_tables: Optional[list[dict]] = None,
        evidence_numbers: Optional[list[float]] = None,
        evidence_notes: Optional[str] = None,
    ) -> dict:
        """Check a narrative against computed evidence.

        Uses the most recent run unless ``run_id`` is given. Callers may add
        extra evidence (tables as ``{"columns": [...], "rows": [...]}`` and raw
        numbers) for figures produced outside the toolkit. Returns ungrounded
        numbers (no matching evidence) plus independent-recomputation issues.
        """
        with self._lock:
            record = self._run(run_id)
            calcs: list[Calculation] = []
            tables: list[DataTable] = []
            notes: list[str] = []
            warnings: list[str] = []

            if record is not None:
                calcs.extend(record.execution.all_calculations())
                tables.extend(record.execution.tables_by_step().values())
                for step in record.execution.steps:
                    if step.output is not None and step.output.notes:
                        notes.append(step.output.notes)
                warnings.extend(record.execution.failed_messages())

            for i, value in enumerate(evidence_numbers or []):
                calcs.append(Calculation(label=f"caller_evidence[{i}]", value=value))
            for i, table in enumerate(evidence_tables or []):
                try:
                    tables.append(
                        DataTable(
                            step_id=-1000 - i,
                            name=str(table.get("name", f"evidence_{i}")),
                            columns=[str(c) for c in table["columns"]],
                            rows=[list(r) for r in table["rows"]],
                            total_rows=len(table["rows"]),
                        )
                    )
                except (KeyError, TypeError) as exc:
                    raise ToolkitError(
                        "evidence_tables entries need 'columns' and 'rows'"
                    ) from exc
            if evidence_notes:
                notes.append(str(evidence_notes))

            if record is None and not calcs and not tables:
                raise ToolkitError(
                    "nothing to verify against: run an operation/plan first, "
                    "or pass evidence_numbers/evidence_tables"
                )

            ungrounded = find_ungrounded_numbers(
                answer or "",
                calcs,
                tables,
                notes="; ".join(notes),
                warnings=warnings,
            )
            issues = []
            if record is not None:
                try:
                    issues = independently_verify(record.df, record.plan, record.execution)
                except Exception as exc:
                    issues = [
                        ValidationIssue(
                            severity="warning",
                            message=f"independent verification failed: {exc}",
                        )
                    ]
            errors = [i for i in issues if i.severity == "error"]
            return {
                "run_id": record.run_id if record else None,
                "grounded": not ungrounded,
                "verified": not errors,
                "ungrounded_numbers": ungrounded,
                "verification_issues": [_issue_payload(i) for i in issues],
            }

    # ---------------------------------------------------------------- helpers

    def _table(self, name: str) -> pd.DataFrame:
        if name not in self._tables:
            raise ToolkitError(f"unknown table '{name}'. loaded: {sorted(self._tables)}")
        return self._tables[name]

    def _profile(self, name: str) -> DatasetProfile:
        self._table(name)
        return self._profiles[name]

    def _duck_for(self, operation: str) -> Optional[DuckSession]:
        if operation != "sql_query" or not self._tables:
            return None
        return DuckSession(dict(self._tables))

    def _record(
        self,
        table: str,
        plan: AnalysisPlan,
        execution: ExecutionResult,
        df: pd.DataFrame,
    ) -> str:
        run_id = uuid.uuid4().hex[:12]
        self._runs[run_id] = _RunRecord(
            run_id=run_id, table=table, plan=plan, execution=execution, df=df
        )
        self._last_run_id = run_id
        while len(self._runs) > self._max_runs:
            self._runs.popitem(last=False)
        return run_id

    def _run(self, run_id: Optional[str]) -> Optional[_RunRecord]:
        target = run_id or self._last_run_id
        if target is None:
            return None
        record = self._runs.get(target)
        if record is None:
            raise ToolkitError(
                f"unknown run_id '{target}'. recent runs: {list(self._runs)[-5:]}"
            )
        self._runs.move_to_end(target)
        return record


def _validate_plan_for_table(
    plan: AnalysisPlan, profile: DatasetProfile
) -> tuple[AnalysisPlan, list[str]]:
    """Validate a plan against the base table's schema.

    Steps without ``input_step`` are canonicalised and checked against the
    loaded table exactly like the planner path. Chained steps run on a derived
    frame whose schema is unknown up front, so only their DAG references are
    checked here; their columns are validated by the op at runtime (the same
    rule the agentic tool loop applies to chained calls).
    """
    issues: list[str] = []
    ids = {s.step_id for s in plan.steps}
    order = {s.step_id: i for i, s in enumerate(plan.steps)}
    for step in plan.steps:
        if step.input_step is None:
            continue
        if step.input_step == step.step_id:
            issues.append(f"step {step.step_id}: input_step cannot reference itself")
        elif step.input_step not in ids:
            issues.append(
                f"step {step.step_id}: input_step references unknown step {step.input_step}"
            )
        else:
            if order[step.input_step] > order[step.step_id]:
                issues.append(
                    f"step {step.step_id}: input_step {step.input_step} comes later in "
                    "execution order (DAG must be acyclic)"
                )
            target = next(s for s in plan.steps if s.step_id == step.input_step)
            if target.operation not in CHAINABLE_OPS:
                issues.append(
                    f"step {step.step_id}: input_step target ({target.operation}) "
                    f"produces no table; choose one of {sorted(CHAINABLE_OPS)}"
                )

    base_steps = [s for s in plan.steps if s.input_step is None]
    if base_steps:
        probe = AnalysisPlan(objective=plan.objective, steps=base_steps, charts=[])
        probe, base_issues = validate_plan_columns(probe, profile)
        issues.extend(base_issues)
        validated = {s.step_id: s for s in probe.steps}
        plan = plan.model_copy(
            update={"steps": [validated.get(s.step_id, s) for s in plan.steps]}
        )
    return plan, issues


def _parse_plan(plan: Any) -> AnalysisPlan:
    try:
        if isinstance(plan, AnalysisPlan):
            return plan
        if isinstance(plan, str):
            plan = json.loads(plan)
        return AnalysisPlan.model_validate(plan)
    except json.JSONDecodeError as exc:
        raise ToolkitError(f"plan is not valid JSON: {exc}") from exc
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
            for err in exc.errors()[:5]
        )
        raise ToolkitError(f"invalid plan: {detail}") from exc


def _single_step_execution(
    operation: str,
    output,
    frame: Optional[pd.DataFrame],
    df: pd.DataFrame,
) -> ExecutionResult:
    return ExecutionResult(
        success=True,
        steps=[
            StepResult(step_id=0, operation=operation, success=True, output=output)
        ],
        final_row_count=len(frame) if frame is not None else len(df),
    )
