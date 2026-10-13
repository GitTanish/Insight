"""Convert the deterministic op registry into provider tool-calling schemas.

Each ``OperationSpec`` already carries a Pydantic ``params_model`` with
``Field`` descriptions, so a valid JSON Schema can be derived mechanically —
no hand-written tool definitions to drift out of sync with the registry. When a
new op is added to ``insight.analytics.operations`` it becomes callable as a
tool for free, exactly like it becomes plannable for free today.

Two wire shapes are emitted because the providers differ:

- OpenAI / OpenAI-compatible (Groq, NIM, OpenRouter): ``{"type": "function",
  "function": {"name", "description", "parameters"}}``.
- Anthropic: ``{"name", "description", "input_schema"}``.

Nested Pydantic models (``MetricSpec``, ``Filter``) and the ``AggFunc`` /
``FilterOperator`` enums are handled by ``model_json_schema`` with inlined
``$defs`` so the schema is self-contained (no external ``$ref`` targets, which
several strict endpoints reject).
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel

from insight.analytics.operations import OPERATIONS


def _inline_defs(schema: dict) -> dict:
    """Resolve ``$defs``/``$ref`` into inline subschemas.

    Provider function-calling endpoints vary in ``$ref`` support; the safe
    common denominator is a fully inlined schema. Cycles are not expected here
    (op params are trees, not graphs), so a depth guard is belt-and-braces.
    """
    schema = copy.deepcopy(schema)
    defs: dict[str, Any] = schema.pop("$defs", {}) or schema.pop("definitions", {}) or {}

    def _resolve(node: Any, depth: int = 0) -> Any:
        if depth > 25:
            return {}
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                name = ref.split("/")[-1]
                target = defs.get(name)
                if target is not None:
                    merged = copy.deepcopy(target)
                    for key, value in node.items():
                        if key != "$ref":
                            merged[key] = _resolve(value, depth + 1)
                    return _resolve(merged, depth + 1)
            return {key: _resolve(value, depth + 1) for key, value in node.items()}
        if isinstance(node, list):
            return [_resolve(item, depth + 1) for item in node]
        return node

    return _resolve(schema)


def parameters_schema(params_model: type[BaseModel]) -> dict:
    """Return a self-contained JSON Schema for an op's parameters."""
    raw = params_model.model_json_schema(ref_template="#/$defs/{model}")
    return _inline_defs(raw)


def openai_tool_schema(name: str, description: str, params_model: type[BaseModel]) -> dict:
    """OpenAI / OpenAI-compatible function-calling tool definition."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_schema(params_model),
        },
    }


def anthropic_tool_schema(name: str, description: str, params_model: type[BaseModel]) -> dict:
    """Anthropic tool definition."""
    return {
        "name": name,
        "description": description,
        "input_schema": parameters_schema(params_model),
    }


def _with_chaining_property(tool: dict) -> dict:
    """Add the loop-level ``input_step`` field to a tool schema.

    Op params describe only the op itself; chaining is a loop-level concern
    (the executor's ``input_step`` DAG), so it is layered on here rather than
    duplicated across every op's Pydantic model.
    """
    tool = copy.deepcopy(tool)
    holder = tool["function"] if "function" in tool else tool
    params = holder.setdefault("parameters", holder.setdefault("input_schema", {}))
    params.setdefault("type", "object")
    params.setdefault("properties", {})["input_step"] = {
        "type": "integer",
        "description": (
            "optional: id of an earlier tool step whose output table this call "
            "should run on (DAG chaining). Each tool result states its step id. "
            "Omit to run on the dataset itself."
        ),
    }
    return tool


def build_openai_tools(names: list[str] | None = None, chaining: bool = True) -> list[dict]:
    """Tool schemas for the given op names (all ops when ``names`` is None)."""
    selected = names if names is not None else list(OPERATIONS)
    tools: list[dict] = []
    for name in selected:
        spec = OPERATIONS.get(name)
        if spec is None:
            continue
        tool = openai_tool_schema(spec.name, spec.description, spec.params_model)
        tools.append(_with_chaining_property(tool) if chaining else tool)
    return tools


def build_anthropic_tools(names: list[str] | None = None, chaining: bool = True) -> list[dict]:
    """Anthropic tool schemas for the given op names (all ops when None)."""
    selected = names if names is not None else list(OPERATIONS)
    tools: list[dict] = []
    for name in selected:
        spec = OPERATIONS.get(name)
        if spec is None:
            continue
        tool = anthropic_tool_schema(spec.name, spec.description, spec.params_model)
        tools.append(_with_chaining_property(tool) if chaining else tool)
    return tools
