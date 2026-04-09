"""
Cheap schema-aware coercion for common LLM argument drift.
"""

from __future__ import annotations

from typing import Any


def coerce_arguments_to_schema(*, schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, Any]:
    if not schema or not isinstance(arguments, dict):
        return dict(arguments)
    return _coerce_value(schema, arguments)


def _coerce_value(schema: dict[str, Any], value: Any) -> Any:
    candidate_schemas = [item for item in schema.get("oneOf", []) if isinstance(item, dict)]
    candidate_schemas.extend(item for item in schema.get("anyOf", []) if isinstance(item, dict))
    if candidate_schemas:
        best = value
        for option in candidate_schemas:
            coerced = _coerce_value(option, value)
            if coerced is not value:
                return coerced
        return best

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        for option in schema_type:
            coerced = _coerce_value({"type": option, **{k: v for k, v in schema.items() if k != "type"}}, value)
            if coerced is not value:
                return coerced
        return value

    if schema_type == "object" and isinstance(value, dict):
        properties = schema.get("properties", {}) or {}
        coerced: dict[str, Any] = {}
        for key, item in value.items():
            property_schema = properties.get(key, {})
            coerced[key] = _coerce_value(property_schema, item) if isinstance(property_schema, dict) else item
        return coerced

    if schema_type == "array" and isinstance(value, list):
        item_schema = schema.get("items", {})
        if isinstance(item_schema, dict):
            return [_coerce_value(item_schema, item) for item in value]
        return list(value)

    if schema_type == "integer":
        return _coerce_integer(value)
    if schema_type == "number":
        return _coerce_number(value)
    if schema_type == "boolean":
        return _coerce_boolean(value)
    return value


def _coerce_integer(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.startswith(("+", "-")):
        sign = text[0]
        digits = text[1:]
        if digits.isdigit():
            return int(sign + digits)
        return value
    if text.isdigit():
        return int(text)
    return value


def _coerce_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        numeric = float(text)
    except ValueError:
        return value
    if numeric.is_integer() and "." not in text and "e" not in text.lower():
        return int(numeric)
    return numeric


def _coerce_boolean(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    return value


__all__ = ["coerce_arguments_to_schema"]
