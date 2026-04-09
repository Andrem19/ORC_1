"""
Fast cached JSON Schema validation for broker-side MCP tool arguments.
"""

from __future__ import annotations

from functools import lru_cache
import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


class ToolArgumentValidationError(ValueError):
    """Raised when a tool call violates the discovered JSON schema."""


def validate_arguments(*, schema: dict[str, Any], arguments: dict[str, Any], tool_name: str) -> None:
    if not schema:
        return
    try:
        validator = _validator_for_schema(json.dumps(schema, sort_keys=True, ensure_ascii=True))
        error = next(validator.iter_errors(arguments), None)
    except ValidationError as exc:  # pragma: no cover - defensive
        raise ToolArgumentValidationError(f"{tool_name}: {exc.message}") from exc
    if error is None:
        return
    if _accept_equivalent_one_of(error=error, instance=error.instance):
        return
    raise ToolArgumentValidationError(_format_validation_error(tool_name=tool_name, error=error))


@lru_cache(maxsize=256)
def _validator_for_schema(serialized_schema: str) -> Draft202012Validator:
    schema = json.loads(serialized_schema)
    return Draft202012Validator(schema)


def _format_validation_error(*, tool_name: str, error: ValidationError) -> str:
    path = ".".join(str(item) for item in error.absolute_path)
    location = f"{tool_name}.{path}" if path else tool_name
    if error.validator == "required":
        missing = list(error.validator_value or [])
        for field in missing:
            if field not in (error.instance or {}):
                return f"{location}: missing required field '{field}'"
    if error.validator == "type":
        expected = error.validator_value
        return f"{location}: expected {expected}"
    return f"{location}: {error.message}"


def _accept_equivalent_one_of(*, error: ValidationError, instance: Any) -> bool:
    if error.validator != "oneOf":
        return False
    candidate_schemas = [item for item in error.validator_value or [] if isinstance(item, dict)]
    if len(candidate_schemas) < 2:
        return False
    allowed_types = {str(item.get("type", "")).strip() for item in candidate_schemas}
    if not allowed_types.issubset({"integer", "number", "boolean", "string"}):
        return False
    validator = Draft202012Validator({"anyOf": candidate_schemas})
    return validator.is_valid(instance)


__all__ = ["ToolArgumentValidationError", "validate_arguments"]
