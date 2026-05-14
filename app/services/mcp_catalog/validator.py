"""
Cached JSON-schema validation for direct MCP tool calls.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from app.services.mcp_catalog.models import McpCatalogSnapshot


@dataclass(frozen=True)
class ToolValidationResult:
    arguments: dict[str, Any]
    local_payload: dict[str, Any] | None = None
    charge_budget: bool = True
    repair_notes: list[str] = field(default_factory=list)


def validate_tool_call(
    *,
    snapshot: McpCatalogSnapshot,
    tool_name: str,
    arguments: dict[str, Any],
) -> ToolValidationResult:
    normalized_tool = str(tool_name or "").strip()
    normalized_arguments = copy.deepcopy(arguments or {})
    tool = snapshot.get_tool(normalized_tool)
    if tool is None:
        return ToolValidationResult(
            arguments=normalized_arguments,
            local_payload=_contract_error_payload(
                tool_name=normalized_tool,
                summary=f"{normalized_tool or 'tool'} is not present in the live MCP catalog",
                json_path="",
                schema_path="",
                extra={"available_tools": snapshot.tool_names()},
            ),
            charge_budget=False,
        )
    schema = tool.input_schema or {"type": "object", "properties": {}}
    validator = _validator_for_schema(snapshot.schema_hash, normalized_tool, schema)
    errors = sorted(validator.iter_errors(normalized_arguments), key=_error_sort_key)
    if not errors:
        return ToolValidationResult(arguments=normalized_arguments)
    first_error = errors[0]
    return ToolValidationResult(
        arguments=normalized_arguments,
        local_payload=_payload_from_validation_error(tool_name=normalized_tool, error=first_error),
        charge_budget=False,
    )


_SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {}


def _error_sort_key(error: ValidationError) -> tuple[int, int, str]:
    return (len(list(error.path)), len(list(error.schema_path)), error.message)


def _payload_from_validation_error(*, tool_name: str, error: ValidationError) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if error.validator == "enum" and isinstance(error.validator_value, list):
        extra["allowed_values"] = list(error.validator_value)
    if error.validator == "required":
        missing_field = _missing_required_field_from_message(error.message)
        if missing_field:
            extra["missing_field"] = missing_field
    return _contract_error_payload(
        tool_name=tool_name,
        summary=error.message,
        json_path=_render_path(error.path),
        schema_path=_render_path(error.schema_path),
        extra=extra,
    )


def _contract_error_payload(
    *,
    tool_name: str,
    summary: str,
    json_path: str,
    schema_path: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = {
        "tool_name": tool_name,
        "json_path": json_path,
        "schema_path": schema_path,
    }
    if extra:
        details.update(extra)
    return {
        "ok": False,
        "error_class": "agent_contract_misuse",
        "summary": summary,
        "details": details,
    }


def _missing_required_field_from_message(message: str) -> str:
    if "'" not in message:
        return ""
    parts = message.split("'")
    if len(parts) < 2:
        return ""
    return str(parts[1]).strip()


def _render_path(path: Any) -> str:
    parts = [str(item) for item in list(path)]
    return ".".join(parts)


def _validator_for_schema(schema_hash: str, tool_name: str, schema: dict[str, Any]):
    key = f"{schema_hash}:{tool_name}"
    _SCHEMA_REGISTRY[key] = schema
    return _build_validator_cached(schema_hash, tool_name, key)


@lru_cache(maxsize=256)
def _build_validator_cached(schema_hash: str, tool_name: str, schema_key: str):
    del schema_hash, tool_name
    schema = _SCHEMA_REGISTRY[schema_key]
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    return validator_cls(schema)


__all__ = ["ToolValidationResult", "validate_tool_call"]
