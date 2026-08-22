from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field


class DataTable(BaseModel):
    step_id: int
    name: str
    columns: list[str]
    rows: list[list[Any]]
    total_rows: int = 0
    truncated: bool = False

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)


class Calculation(BaseModel):
    label: str
    value: Any
    detail: Optional[str] = None


class OperationOutput(BaseModel):
    table: Optional[DataTable] = None
    calculations: list[Calculation] = Field(default_factory=list)
    notes: Optional[str] = None


class StepResult(BaseModel):
    step_id: int
    operation: str
    success: bool
    error: Optional[str] = None
    output: Optional[OperationOutput] = None


class ExecutionResult(BaseModel):
    success: bool
    steps: list[StepResult] = Field(default_factory=list)
    final_row_count: Optional[int] = None
    duration_ms: Optional[int] = None

    def tables_by_step(self) -> dict[int, DataTable]:
        tables: dict[int, DataTable] = {}
        for step in self.steps:
            if step.success and step.output is not None and step.output.table:
                tables[step.step_id] = step.output.table
        return tables

    def get_table(self, step_id: int) -> DataTable | None:
        return self.tables_by_step().get(step_id)

    def all_calculations(self) -> list[Calculation]:
        calcs: list[Calculation] = []
        for step in self.steps:
            if step.success and step.output is not None:
                calcs.extend(step.output.calculations)
        return calcs

    def failed_messages(self) -> list[str]:
        return [
            f"step {s.step_id} ({s.operation}): {s.error}"
            for s in self.steps
            if not s.success
        ]
