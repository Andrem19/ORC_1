"""
Normalization and hashing for live MCP tool catalogs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from app.execution_models import utc_now_iso
from app.services.mcp_catalog.classifier import derive_tool_semantics
from app.services.mcp_catalog.models import McpCatalogSnapshot, McpFieldSpec, McpToolSpec


def build_catalog_snapshot(
    *,
    tools: list[dict[str, Any]],
    endpoint_url: str,
    bootstrap_manifest: dict[str, Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
    manifest_source: str = "",
    fetched_at: str | None = None,
) -> McpCatalogSnapshot:
    manifest_overrides = _tool_manifest_map(capability_manifest or bootstrap_manifest or {})
    specs = [
        normalize_tool_spec(tool, manifest_entry=manifest_overrides.get(str(tool.get("name", "") or "").strip()))
        for tool in tools
        if str(tool.get("name", "") or "").strip()
    ]
    specs.sort(key=lambda item: item.name)
    schema_hash = compute_schema_hash(specs)
    return McpCatalogSnapshot(
        server_name="dev_space1",
        endpoint_url=endpoint_url,
        schema_hash=schema_hash,
        fetched_at=fetched_at or utc_now_iso(),
        tools=specs,
        bootstrap_manifest=dict(bootstrap_manifest or {}),
        capability_manifest=dict(capability_manifest or {}),
        manifest_source=manifest_source,
    )


def normalize_tool_spec(tool: dict[str, Any], *, manifest_entry: dict[str, Any] | None = None) -> McpToolSpec:
    name = str(tool.get("name", "") or "").strip()
    description = str(tool.get("description", "") or "").strip()
    title = str(tool.get("title", "") or "").strip()
    raw_schema = tool.get("inputSchema") or tool.get("input_schema") or {"type": "object", "properties": {}}
    schema = canonicalize_schema(raw_schema if isinstance(raw_schema, dict) else {"type": "object", "properties": {}})
    required = sorted({str(item).strip() for item in schema.get("required", []) or [] if str(item).strip()})
    fields = _field_specs(schema, required_fields=set(required))
    semantics = derive_tool_semantics(
        name=name,
        description=description,
        schema=schema,
        manifest_entry=manifest_entry,
    )
    return McpToolSpec(
        name=name,
        description=description,
        title=title,
        input_schema=schema,
        required_fields=required,
        fields=fields,
        additional_properties=_normalize_additional_properties(schema.get("additionalProperties")),
        cost_class=str(semantics.get("cost_class") or "cheap"),
        side_effects=str(semantics.get("side_effects") or "read_only"),
        async_pattern=str(semantics.get("async_pattern") or ""),
        id_fields=[str(item) for item in semantics.get("id_fields", []) or []],
        recommended_discovery_flow=[str(item) for item in semantics.get("recommended_discovery_flow", []) or []],
        replaces_tools=[str(item) for item in semantics.get("replaces_tools", []) or []],
        deprecated=bool(semantics.get("deprecated", False)),
        stability=str(semantics.get("stability") or "stable"),
        family=str(semantics.get("family") or ""),
        accepted_handle_fields=[str(item) for item in semantics.get("accepted_handle_fields", []) or []],
        produced_handle_fields=[str(item) for item in semantics.get("produced_handle_fields", []) or []],
        supports_terminal_write=bool(semantics.get("supports_terminal_write", False)),
        supports_discovery=bool(semantics.get("supports_discovery", False)),
        supports_polling=bool(semantics.get("supports_polling", False)),
        async_like=bool(semantics.get("async_like", False)),
    )


def compute_schema_hash(tools: list[McpToolSpec]) -> str:
    stable_payload = [tool.contract_signature() for tool in sorted(tools, key=lambda item: item.name)]
    encoded = json.dumps(stable_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonicalize_schema(value: Any, *, parent_key: str = "") -> Any:
    if isinstance(value, dict):
        return {
            str(key): canonicalize_schema(item, parent_key=str(key))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        normalized = [canonicalize_schema(item, parent_key=parent_key) for item in value]
        if parent_key in {"required", "enum", "type"}:
            return sorted(normalized, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True))
        return normalized
    return value


def snapshot_to_dict(snapshot: McpCatalogSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def snapshot_from_dict(payload: dict[str, Any]) -> McpCatalogSnapshot:
    tools: list[McpToolSpec] = []
    for raw_tool in payload.get("tools", []) or []:
        fields = {
            str(field_name): McpFieldSpec(
                name=str(field_payload.get("name", field_name) or field_name),
                types=[str(item) for item in field_payload.get("types", []) or []],
                required=bool(field_payload.get("required", False)),
                enum=list(field_payload.get("enum", []) or []),
                default=field_payload.get("default"),
                description=str(field_payload.get("description", "") or ""),
            )
            for field_name, field_payload in dict(raw_tool.get("fields", {}) or {}).items()
            if isinstance(field_payload, dict)
        }
        tools.append(
            McpToolSpec(
                name=str(raw_tool.get("name", "") or ""),
                description=str(raw_tool.get("description", "") or ""),
                title=str(raw_tool.get("title", "") or ""),
                input_schema=dict(raw_tool.get("input_schema", {}) or {}),
                required_fields=[str(item) for item in raw_tool.get("required_fields", []) or []],
                fields=fields,
                additional_properties=raw_tool.get("additional_properties"),
                cost_class=str(raw_tool.get("cost_class", "cheap") or "cheap"),
                side_effects=str(raw_tool.get("side_effects", "read_only") or "read_only"),
                async_pattern=str(raw_tool.get("async_pattern", "") or ""),
                id_fields=[str(item) for item in raw_tool.get("id_fields", []) or []],
                recommended_discovery_flow=[str(item) for item in raw_tool.get("recommended_discovery_flow", []) or []],
                replaces_tools=[str(item) for item in raw_tool.get("replaces_tools", []) or []],
                deprecated=bool(raw_tool.get("deprecated", False)),
                stability=str(raw_tool.get("stability", "stable") or "stable"),
                family=str(raw_tool.get("family", "") or ""),
                accepted_handle_fields=[str(item) for item in raw_tool.get("accepted_handle_fields", []) or []],
                produced_handle_fields=[str(item) for item in raw_tool.get("produced_handle_fields", []) or []],
                supports_terminal_write=bool(raw_tool.get("supports_terminal_write", False)),
                supports_discovery=bool(raw_tool.get("supports_discovery", False)),
                supports_polling=bool(raw_tool.get("supports_polling", False)),
                async_like=bool(raw_tool.get("async_like", False)),
            )
        )
    return McpCatalogSnapshot(
        server_name=str(payload.get("server_name", "dev_space1") or "dev_space1"),
        endpoint_url=str(payload.get("endpoint_url", "") or ""),
        schema_hash=str(payload.get("schema_hash", "") or ""),
        fetched_at=str(payload.get("fetched_at", "") or ""),
        tools=sorted(tools, key=lambda item: item.name),
        bootstrap_manifest=dict(payload.get("bootstrap_manifest", {}) or {}),
        capability_manifest=dict(payload.get("capability_manifest", {}) or {}),
        manifest_source=str(payload.get("manifest_source", "") or ""),
    )


def _field_specs(schema: dict[str, Any], *, required_fields: set[str]) -> dict[str, McpFieldSpec]:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return {}
    result: dict[str, McpFieldSpec] = {}
    for name, raw_field in sorted(properties.items(), key=lambda item: str(item[0])):
        field_schema = raw_field if isinstance(raw_field, dict) else {}
        field_name = str(name)
        result[field_name] = McpFieldSpec(
            name=field_name,
            types=_field_types(field_schema),
            required=field_name in required_fields,
            enum=list(field_schema.get("enum", []) or []),
            default=field_schema.get("default"),
            description=str(field_schema.get("description", "") or field_schema.get("title", "") or ""),
        )
    return result


def _field_types(field_schema: dict[str, Any]) -> list[str]:
    raw_type = field_schema.get("type")
    if isinstance(raw_type, str) and raw_type.strip():
        return [raw_type.strip()]
    if isinstance(raw_type, list):
        return sorted({str(item).strip() for item in raw_type if str(item).strip()})
    if isinstance(field_schema.get("anyOf"), list):
        collected: set[str] = set()
        for item in field_schema["anyOf"]:
            if isinstance(item, dict):
                nested_type = item.get("type")
                if isinstance(nested_type, str) and nested_type.strip():
                    collected.add(nested_type.strip())
        return sorted(collected)
    return []


def _normalize_additional_properties(value: Any) -> bool | dict[str, Any] | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        return canonicalize_schema(value)
    return None


def _tool_manifest_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    candidates = manifest.get("tools") if isinstance(manifest, dict) else None
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool_name") or item.get("tool") or "").strip()
            if name:
                result[name] = dict(item)
    elif isinstance(candidates, dict):
        for key, item in candidates.items():
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or key or "").strip()
            if name:
                result[name] = dict(item)
    return result


__all__ = [
    "build_catalog_snapshot",
    "canonicalize_schema",
    "compute_schema_hash",
    "normalize_tool_spec",
    "snapshot_from_dict",
    "snapshot_to_dict",
]
