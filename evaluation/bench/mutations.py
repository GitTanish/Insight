"""Mutation catalogue for the adversarial verifier matrix.

A mutation is described declaratively (``MutationSpec``) and applied to a deep
copy of a clean ``ExecutionResult``. The benchmark records both the mutation and
the verdicts, never the mutated object itself.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any

from insight.domain.execution import DataTable, ExecutionResult


@dataclass(frozen=True)
class MutationSpec:
    mutation_class: str
    target: dict
    expectation: str = "detect"
    magnitude: float | None = None
    params: dict = field(default_factory=dict)


def _table(execution: ExecutionResult, step_id: int = 0) -> DataTable | None:
    return execution.tables_by_step().get(step_id)


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    if not _is_number(value):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _first_row(table: DataTable, column: str) -> int | None:
    if column not in table.columns:
        return None
    index = table.columns.index(column)
    for row_index, row in enumerate(table.rows):
        if _finite_number(row[index]):
            return row_index
    return None


def _bounded_new_value(
    value: float, magnitude: float, bounds: tuple[float, float] | None
) -> tuple[float, float, bool]:
    """Return (new_value, effective_delta, clamped)."""
    delta = max(abs(value), 1.0) * magnitude
    if not bounds:
        return value + delta, delta, False
    low, high = bounds
    candidates = [value + delta, value - delta]
    for candidate in candidates:
        if low <= candidate <= high:
            return candidate, abs(candidate - value), False
    # The requested relative magnitude cannot fit inside the field's domain.
    candidate = high if (high - value) >= (value - low) else low
    if candidate == value:
        candidate = low if value != low else high
    return candidate, abs(candidate - value), True


def _calc(execution: ExecutionResult, step_id: int, label: str):
    for result in execution.steps:
        if result.step_id != step_id or not result.success or result.output is None:
            continue
        for calculation in result.output.calculations:
            if calculation.label == label:
                return calculation
    return None


def _step_table_metadata(table: DataTable, column: str, row: int) -> dict:
    return {
        "kind": "table_cell",
        "step_id": table.step_id,
        "table": table.name,
        "column": column,
        "row": row,
    }


def _apply_table_numeric(
    execution: ExecutionResult,
    spec: MutationSpec,
) -> tuple[bool, Any, Any, dict]:
    table = _table(execution, spec.target.get("step_id", 0))
    if table is None:
        return False, None, None, {}
    column = str(spec.params.get("column"))
    row = spec.params.get("row")
    if column not in table.columns:
        return False, None, None, {}
    index = table.columns.index(column)
    if row is None:
        row = _first_row(table, column)
    if row is None or not (0 <= int(row) < len(table.rows)):
        return False, None, None, {}
    row = int(row)
    before = table.rows[row][index]
    if not _is_number(before):
        return False, None, None, {}
    before_f = float(before)
    magnitude = float(spec.magnitude or 0.0)
    operation = spec.params.get("operation", "delta")
    clamped = False
    if operation == "zero":
        after_f = 0.0
    elif operation == "sign_flip":
        after_f = -before_f
    elif operation == "nan":
        after_f = float("nan")
    elif operation == "inf":
        after_f = float("inf")
    else:
        after_f, _effective, clamped = _bounded_new_value(
            before_f, magnitude, spec.params.get("bounds")
        )
    if operation not in {"nan", "inf"} and after_f == before_f:
        return False, None, None, {}
    table.rows[row][index] = after_f
    target = _step_table_metadata(table, column, row)
    details = {"clamped_to_domain": clamped, "effective_delta": after_f - before_f}
    return True, before, after_f, {"target": target, **details}


def _apply_count_delta(
    execution: ExecutionResult,
    spec: MutationSpec,
) -> tuple[bool, Any, Any, dict]:
    table = _table(execution, spec.target.get("step_id", 0))
    if table is None:
        return False, None, None, {}
    column = str(spec.params.get("column"))
    if column not in table.columns:
        return False, None, None, {}
    index = table.columns.index(column)
    row = spec.params.get("row")
    if row is None:
        row = _first_row(table, column)
    if row is None or not (0 <= int(row) < len(table.rows)):
        return False, None, None, {}
    row = int(row)
    before = table.rows[row][index]
    if not _is_number(before):
        return False, None, None, {}
    delta = int(spec.params.get("delta", 1))
    after = int(before) + delta
    if after < 0:
        after = int(before) - delta
    if after == before:
        return False, None, None, {}
    table.rows[row][index] = after
    target = _step_table_metadata(table, column, row)
    return True, before, after, {"target": target}


def _apply_calculation(
    execution: ExecutionResult,
    spec: MutationSpec,
) -> tuple[bool, Any, Any, dict]:
    label = str(spec.params.get("label"))
    calculation = _calc(execution, spec.target.get("step_id", 0), label)
    if calculation is None:
        return False, None, None, {}
    before = calculation.value
    if not _is_number(before):
        return False, None, None, {}
    operation = spec.params.get("operation", "delta")
    before_f = float(before)
    clamped = False
    if isinstance(before, int) and not isinstance(before, bool) and operation == "count_delta":
        delta = int(spec.params.get("delta", 1))
        after: Any = int(before) + delta
        if after < 0:
            after = int(before) - delta
        if after == before:
            return False, None, None, {}
    elif operation == "zero":
        after = 0.0
    elif operation == "sign_flip":
        after = -before_f
    elif operation == "nan":
        after = float("nan")
    elif operation == "inf":
        after = float("inf")
    else:
        after, _effective, clamped = _bounded_new_value(
            before_f, float(spec.magnitude or 0.0), spec.params.get("bounds")
        )
    if operation not in {"nan", "inf"} and after == before:
        return False, None, None, {}
    calculation.value = after
    target = {
        "kind": "calculation",
        "step_id": spec.target.get("step_id", 0),
        "label": label,
    }
    details = {"clamped_to_domain": clamped}
    return True, before, after, {"target": target, **details}


def _correlation_cell(
    table: DataTable, row_label: str, column_label: str
) -> tuple[int, int] | None:
    if row_label not in table.columns or column_label not in table.columns:
        return None
    # Correlation tables are square: first column holds the row labels.
    row_index = None
    for index, row in enumerate(table.rows):
        if str(row[0]) == row_label:
            row_index = index
            break
    if row_index is None:
        return None
    return row_index, table.columns.index(column_label)


def _apply_correlation(
    execution: ExecutionResult,
    spec: MutationSpec,
) -> tuple[bool, Any, Any, dict]:
    table = _table(execution, spec.target.get("step_id", 0))
    if table is None:
        return False, None, None, {}
    row_label = str(spec.params.get("row"))
    column_label = str(spec.params.get("column"))
    location = _correlation_cell(table, row_label, column_label)
    if location is None:
        return False, None, None, {}
    row_index, column_index = location
    before = table.rows[row_index][column_index]
    if not _is_number(before):
        return False, None, None, {}
    before_f = float(before)
    operation = spec.params.get("operation", "delta")
    if operation == "zero":
        after = 0.0
    elif operation == "sign_flip":
        after = -before_f
    elif operation == "nan":
        after = float("nan")
    elif operation == "inf":
        after = float("inf")
    else:
        after, _effective, clamped = _bounded_new_value(
            before_f, float(spec.magnitude or 0.0), spec.params.get("bounds", (-1.0, 1.0))
        )
    if operation not in {"nan", "inf"} and after == before_f:
        return False, None, None, {}
    cells = [(row_index, column_index)]
    if spec.params.get("symmetric_pair") and row_label != column_label:
        mirrored = _correlation_cell(table, column_label, row_label)
        if mirrored is not None:
            cells.append(mirrored)
    for r_index, c_index in cells:
        table.rows[r_index][c_index] = after
    target = {
        "kind": "correlation_cell",
        "step_id": spec.target.get("step_id", 0),
        "row": row_label,
        "column": column_label,
        "symmetric_pair": bool(spec.params.get("symmetric_pair")),
    }
    return True, before, after, {
        "target": target,
        "clamped_to_domain": bool(spec.params.get("clamped_to_domain")),
    }


def _apply_structural(
    execution: ExecutionResult,
    spec: MutationSpec,
) -> tuple[bool, Any, Any, dict]:
    table = _table(execution, spec.target.get("step_id", 0))
    if table is None:
        return False, None, None, {}
    operation = spec.params.get("operation")
    target_column = spec.params.get("column")

    if operation == "drop_row":
        if not table.rows:
            return False, None, None, {}
        removed = table.rows.pop(0)
        target = {"kind": "table_rows", "step_id": table.step_id, "operation": operation}
        return True, removed, None, {"target": target}

    if operation == "duplicate_row":
        if not table.rows:
            return False, None, None, {}
        duplicated = list(table.rows[0])
        table.rows.append(duplicated)
        target = {"kind": "table_rows", "step_id": table.step_id, "operation": operation}
        return True, duplicated, duplicated, {"target": target}

    if operation == "swap_rows":
        if len(table.rows) < 2:
            return False, None, None, {}
        table.rows[0], table.rows[1] = table.rows[1], table.rows[0]
        target = {"kind": "table_rows", "step_id": table.step_id, "operation": operation}
        return True, table.rows[1], table.rows[0], {"target": target}

    if operation == "wrong_key":
        if not table.rows or not table.columns:
            return False, None, None, {}
        before = table.rows[0][0]
        if isinstance(before, str):
            after = f"{before}__tampered"
        elif _is_number(before):
            after = float(before) + 999.0
        else:
            after = "__tampered__"
        table.rows[0][0] = after
        target = {
            "kind": "table_cell",
            "step_id": table.step_id,
            "column": table.columns[0],
            "row": 0,
        }
        return True, before, after, {"target": target}

    if operation == "extra_row":
        if not table.rows:
            return False, None, None, {}
        extra = list(table.rows[0])
        if isinstance(extra[0], str):
            extra[0] = f"{extra[0]}__extra"
        elif _is_number(extra[0]):
            extra[0] = float(extra[0]) + 999.0
        else:
            extra[0] = "__extra__"
        table.rows.append(extra)
        target = {"kind": "table_rows", "step_id": table.step_id, "operation": operation}
        return True, None, extra, {"target": target}

    if operation == "drop_column":
        if not target_column or target_column not in table.columns:
            return False, None, None, {}
        index = table.columns.index(target_column)
        removed = [row[index] for row in table.rows]
        table.columns.pop(index)
        for row in table.rows:
            row.pop(index)
        target = {
            "kind": "table_column",
            "step_id": table.step_id,
            "column": target_column,
        }
        return True, removed, None, {"target": target}

    if operation == "false_truncation":
        before = table.truncated
        table.truncated = True
        target = {"kind": "table_flag", "step_id": table.step_id, "flag": "truncated"}
        return True, before, True, {"target": target}

    if operation == "truncated_hide_missing":
        if not table.rows:
            return False, None, None, {}
        removed = table.rows.pop(0)
        table.truncated = True
        target = {"kind": "table_rows", "step_id": table.step_id, "operation": operation}
        return True, removed, None, {"target": target}

    return False, None, None, {}


def apply_mutation(
    execution: ExecutionResult, spec: MutationSpec
) -> tuple[bool, Any, Any, dict]:
    """Apply a mutation in place. Returns (applied, before, after, metadata)."""
    if spec.mutation_class in {
        "numeric_delta",
        "zero",
        "sign_flip",
        "nonfinite",
    }:
        return _apply_table_numeric(execution, spec)
    if spec.mutation_class == "count_delta":
        return _apply_count_delta(execution, spec)
    if spec.mutation_class == "calculation_delta":
        return _apply_calculation(execution, spec)
    if spec.mutation_class == "correlation_delta":
        return _apply_correlation(execution, spec)
    if spec.mutation_class == "structural":
        return _apply_structural(execution, spec)
    raise ValueError(f"unknown mutation class: {spec.mutation_class}")


def enumerate_mutations(
    fixture,
    execution: ExecutionResult,
    *,
    magnitudes: tuple[float, ...],
    count_deltas: tuple[int, ...],
    include_fixed_numeric: bool = True,
    include_structural: bool = True,
    include_calculation: bool = True,
    include_correlation: bool = True,
    max_numeric_targets: int | None = None,
    max_calculation_targets: int | None = None,
) -> list[MutationSpec]:
    specs: list[MutationSpec] = []
    table = _table(execution)
    if table is None:
        return specs

    numeric_targets = fixture.numeric_targets
    if max_numeric_targets is not None:
        numeric_targets = numeric_targets[:max_numeric_targets]
    for target in numeric_targets:
        row = _first_row(table, target.column)
        if row is None:
            continue
        base = {
            "step_id": 0,
            "column": target.column,
            "row": row,
            "bounds": target.bounds,
        }
        for magnitude in magnitudes:
            specs.append(
                MutationSpec(
                    mutation_class="numeric_delta",
                    target={"kind": "table_column", "step_id": 0, "column": target.column},
                    expectation=target.expectation,
                    magnitude=magnitude,
                    params={**base, "operation": "delta"},
                )
            )
        if include_fixed_numeric:
            specs.append(
                MutationSpec(
                    mutation_class="zero",
                    target={"kind": "table_column", "step_id": 0, "column": target.column},
                    expectation=target.expectation,
                    params={**base, "operation": "zero"},
                )
            )
            specs.append(
                MutationSpec(
                    mutation_class="sign_flip",
                    target={"kind": "table_column", "step_id": 0, "column": target.column},
                    expectation=target.expectation,
                    params={**base, "operation": "sign_flip"},
                )
            )
            for nonfinite_name in ("nan", "inf"):
                specs.append(
                    MutationSpec(
                        mutation_class="nonfinite",
                        target={"kind": "table_column", "step_id": 0, "column": target.column},
                        expectation=target.expectation,
                        params={**base, "operation": nonfinite_name},
                    )
                )

    for target in fixture.count_targets:
        row = _first_row(table, target.column)
        if row is None:
            continue
        for delta in count_deltas:
            specs.append(
                MutationSpec(
                    mutation_class="count_delta",
                    target={"kind": "table_column", "step_id": 0, "column": target.column},
                    expectation=target.expectation,
                    magnitude=float(delta),
                    params={
                        "step_id": 0,
                        "column": target.column,
                        "row": row,
                        "delta": delta,
                    },
                )
            )

    if include_calculation:
        calculation_targets = fixture.calculation_targets
        if max_calculation_targets is not None:
            calculation_targets = calculation_targets[:max_calculation_targets]
        for target in calculation_targets:
            calculation = _calc(execution, 0, target.label)
            if calculation is None or not _is_number(calculation.value):
                continue
            base = {"step_id": 0, "label": target.label, "bounds": target.bounds}
            if isinstance(calculation.value, int) and not isinstance(calculation.value, bool):
                for delta in count_deltas:
                    specs.append(
                        MutationSpec(
                            mutation_class="calculation_delta",
                            target={"kind": "calculation", "step_id": 0, "label": target.label},
                            expectation=target.expectation,
                            magnitude=float(delta),
                            params={**base, "operation": "count_delta", "delta": delta},
                        )
                    )
                continue
            for magnitude in magnitudes:
                specs.append(
                    MutationSpec(
                        mutation_class="calculation_delta",
                        target={"kind": "calculation", "step_id": 0, "label": target.label},
                        expectation=target.expectation,
                        magnitude=magnitude,
                        params={**base, "operation": "delta"},
                    )
                )
            if include_fixed_numeric:
                specs.append(
                    MutationSpec(
                        mutation_class="calculation_delta",
                        target={"kind": "calculation", "step_id": 0, "label": target.label},
                        expectation=target.expectation,
                        params={**base, "operation": "zero"},
                    )
                )
                specs.append(
                    MutationSpec(
                        mutation_class="calculation_delta",
                        target={"kind": "calculation", "step_id": 0, "label": target.label},
                        expectation=target.expectation,
                        params={**base, "operation": "sign_flip"},
                    )
                )
                for nonfinite_name in ("nan", "inf"):
                    specs.append(
                        MutationSpec(
                            mutation_class="calculation_delta",
                            target={"kind": "calculation", "step_id": 0, "label": target.label},
                            expectation=target.expectation,
                            params={**base, "operation": nonfinite_name},
                        )
                    )

    if include_correlation:
        for target in fixture.correlation_targets:
            base = {
                "step_id": 0,
                "row": target.row,
                "column": target.column,
                "symmetric_pair": target.symmetric_pair,
                "bounds": (-1.0, 1.0),
            }
            for magnitude in magnitudes:
                specs.append(
                    MutationSpec(
                        mutation_class="correlation_delta",
                        target={"kind": "correlation_cell", "step_id": 0},
                        expectation=target.expectation,
                        magnitude=magnitude,
                        params={**base, "operation": "delta"},
                    )
                )
            if include_fixed_numeric:
                for operation in ("zero", "nan", "inf"):
                    specs.append(
                        MutationSpec(
                            mutation_class="correlation_delta",
                            target={"kind": "correlation_cell", "step_id": 0},
                            expectation=target.expectation,
                            params={**base, "operation": operation},
                        )
                    )

    if include_structural:
        specs.extend(_structural_mutations(fixture, table))
    return specs


def _structural_mutations(fixture, table: DataTable) -> list[MutationSpec]:
    structural = ("drop_row", "duplicate_row", "swap_rows", "wrong_key", "extra_row")
    drop_column = None
    for target in fixture.numeric_targets:
        if target.column in table.columns:
            drop_column = target.column
            break
    if drop_column is None and len(table.columns) > 1:
        drop_column = table.columns[1]

    specs = []
    for operation in structural:
        expectation = "detect"
        if operation == "swap_rows" and fixture.operation not in {"top_n", "compare_subsets"}:
            # Keyed tables are semantically order-insensitive; a swap is still
            # recorded but not scored as a correctness failure.
            expectation = "informational"
        specs.append(
            MutationSpec(
                mutation_class="structural",
                target={"kind": "table", "step_id": 0},
                expectation=expectation,
                params={"operation": operation},
            )
        )
    if drop_column:
        specs.append(
            MutationSpec(
                mutation_class="structural",
                target={"kind": "table_column", "step_id": 0, "column": drop_column},
                expectation="detect",
                params={"operation": "drop_column", "column": drop_column},
            )
        )
    specs.extend(
        [
            MutationSpec(
                mutation_class="structural",
                target={"kind": "table_flag", "step_id": 0, "flag": "truncated"},
                expectation="known_bypass",
                params={"operation": "false_truncation"},
            ),
            MutationSpec(
                mutation_class="structural",
                target={"kind": "table_rows", "step_id": 0},
                expectation="known_bypass",
                params={"operation": "truncated_hide_missing"},
            ),
        ]
    )
    return specs
