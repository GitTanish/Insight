from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class FilterOperator(str, Enum):
    eq = "eq"
    ne = "ne"
    gt = "gt"
    gte = "gte"
    lt = "lt"
    lte = "lte"
    in_ = "in"
    not_in = "not_in"
    contains = "contains"
    between = "between"
    isna = "isna"
    notna = "notna"


class Filter(BaseModel):
    column: str
    op: FilterOperator
    value: Any = None


class AggFunc(str, Enum):
    count = "count"
    nunique = "nunique"
    sum = "sum"
    mean = "mean"
    median = "median"
    min = "min"
    max = "max"
    std = "std"
    var = "var"


class MetricSpec(BaseModel):
    column: Optional[str] = None
    agg: AggFunc = AggFunc.count
    alias: Optional[str] = None

    def output_name(self) -> str:
        if self.alias:
            return self.alias
        if self.column:
            return f"{self.agg.value}_{self.column}"
        return "row_count"

    @model_validator(mode="after")
    def _require_column(self) -> "MetricSpec":
        if self.agg != AggFunc.count and not self.column:
            raise ValueError(f"metric '{self.output_name()}' requires a column")
        return self


class SortDirection(str, Enum):
    asc = "asc"
    desc = "desc"


class ChartType(str, Enum):
    bar = "bar"
    line = "line"
    scatter = "scatter"
    hist = "hist"
    pie = "pie"


class ChartSpec(BaseModel):
    chart_type: ChartType
    title: str
    source_step: int
    x: Optional[str] = None
    y: Optional[str] = None
    sort_desc: Optional[bool] = None
    top_n: Optional[int] = Field(default=None, ge=1, le=50)
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    def resolved_top_n(self) -> int:
        return self.top_n if self.top_n is not None else 12

    def resolved_sort_desc(self) -> bool:
        return self.sort_desc if self.sort_desc is not None else True


class PlanStep(BaseModel):
    step_id: int
    operation: str
    params: dict[str, Any] = Field(default_factory=dict)
    reason: Optional[str] = None
    input_step: Optional[int] = Field(
        default=None,
        description=(
            "optional id of an earlier step whose output table becomes the "
            "dataset for this operation (DAG chaining)"
        ),
    )


class AnalysisPlan(BaseModel):
    objective: str
    steps: list[PlanStep] = Field(min_length=1)
    charts: list[ChartSpec] = Field(default_factory=list)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _unique_step_ids(self) -> "AnalysisPlan":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step_ids must be unique")
        return self
