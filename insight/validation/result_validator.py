from __future__ import annotations

import math

from insight.domain.analysis import AnalysisPlan
from insight.domain.execution import ExecutionResult
from insight.domain.visualization import ValidationIssue, ValidationResult

_SMALL_SAMPLE_THRESHOLD = 30


def _scan_numeric_issues(table) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col_idx, col_name in enumerate(table.columns):
        for row_idx, row in enumerate(table.rows[:200]):
            value = row[col_idx]
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                issues.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"table '{table.name}' has non-finite value "
                            f"({value}) at [{row_idx}][{col_name}]"
                        ),
                        step_id=table.step_id,
                    )
                )
                break
    return issues


_MISSINGNESS_WARN_THRESHOLD = 0.05


def _referenced_columns(plan: AnalysisPlan, profile) -> list[str]:
    known = {c.name.lower(): c.name for c in profile.columns}
    found: set[str] = set()

    def walk(value) -> None:
        if isinstance(value, str):
            canonical = known.get(value.strip().lower())
            if canonical:
                found.add(canonical)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    for step in plan.steps:
        if isinstance(step.params, dict):
            walk(step.params)
    return sorted(found)


def _missingness_warnings(plan: AnalysisPlan, profile) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col_name in _referenced_columns(plan, profile):
        column = profile.get_column(col_name)
        if column is not None and column.missing_pct > _MISSINGNESS_WARN_THRESHOLD:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=(
                        f"column '{col_name}' is {column.missing_pct:.0%} missing; rows with "
                        "incomplete values are excluded from calculations, which assumes data "
                        "are Missing Completely at Random (MCAR)"
                    ),
                )
            )
    return issues


def validate_execution(
    plan: AnalysisPlan,
    execution: ExecutionResult,
    df=None,
    profile=None,
) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    checks_run: list[str] = []

    tables = execution.tables_by_step()

    checks_run.append("step_success")
    for result in execution.steps:
        if not result.success:
            severity = "error"
            errors.append(
                ValidationIssue(
                    severity=severity,
                    message=f"step {result.step_id} ({result.operation}) failed: {result.error}",
                    step_id=result.step_id,
                )
            )

    checks_run.append("tables_present")
    from insight.analytics.operations import TABLE_PRODUCING_OPS

    for step in plan.steps:
        if step.operation in TABLE_PRODUCING_OPS and step.step_id not in tables:
            if any(
                r.step_id == step.step_id and r.success for r in execution.steps
            ):
                warnings.append(
                    ValidationIssue(
                        severity="warning",
                        message=(
                            f"step {step.step_id} ({step.operation}) succeeded but "
                            f"produced no table"
                        ),
                        step_id=step.step_id,
                    )
                )

    checks_run.append("numeric_sanity")
    for step_id, table in tables.items():
        errors.extend(_scan_numeric_issues(table))
        if table.total_rows == 0 or not table.rows:
            source = next((s for s in plan.steps if s.step_id == step_id), None)
            op_name = source.operation if source else "?"
            filters_used = bool(source and source.params.get("filters"))
            severity = "warning" if filters_used else "error"
            hint = " (filters matched no rows)" if filters_used else ""
            (
                warnings if severity == "warning" else errors
            ).append(
                ValidationIssue(
                    severity=severity,
                    message=f"table from step {step_id} ({op_name}) is empty{hint}",
                    step_id=step_id,
                )
            )

    checks_run.append("small_sample")
    small_groups: list[str] = []
    for step_id, table in tables.items():
        if not table.columns:
            continue
        count_cols = [
            i for i, c in enumerate(table.columns)
            if c.lower() in {"count", "row_count"} or c.lower().startswith(("count_", "sum_"))
        ]
        for row in table.rows[:100]:
            for idx in count_cols:
                v = row[idx]
                if isinstance(v, int) and 0 < v < _SMALL_SAMPLE_THRESHOLD:
                    label = str(row[0])[:30]
                    small_groups.append(f"{label} (n={v})")
    if small_groups:
        sample = ", ".join(small_groups[:5])
        more = f" (+{len(small_groups) - 5} more)" if len(small_groups) > 5 else ""
        warnings.append(
            ValidationIssue(
                severity="warning",
                message=(
                    f"small sample sizes detected (< {_SMALL_SAMPLE_THRESHOLD}): "
                    f"{sample}{more}. Interpret with caution."
                ),
            )
        )

    if profile is not None:
        checks_run.append("missingness_caveats")
        warnings.extend(_missingness_warnings(plan, profile))

    if df is not None:
        from insight.validation.verification import independently_verify

        checks_run.append("independent_recompute")
        errors.extend(independently_verify(df, plan, execution))

    checks_run.append("chart_sources")
    for chart in plan.charts:
        source_table = tables.get(chart.source_step)
        if source_table is None:
            errors.append(
                ValidationIssue(
                    severity="error",
                    message=(
                        f"chart '{chart.title}' references step {chart.source_step} "
                        f"which produced no table"
                    ),
                    step_id=chart.source_step,
                )
            )
            continue
        available = set(source_table.columns)
        for axis in (chart.x, chart.y):
            if axis is not None and axis not in available:
                errors.append(
                    ValidationIssue(
                        severity="error",
                        message=(
                            f"chart '{chart.title}' references column '{axis}' "
                            f"not present in step {chart.source_step} output "
                            f"(available: {sorted(available)})"
                        ),
                        step_id=chart.source_step,
                    )
                )

    return ValidationResult(
        valid=not errors,
        errors=errors,
        warnings=warnings,
        checks_run=checks_run,
    )
