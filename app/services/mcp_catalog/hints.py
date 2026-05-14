"""
Generated prompt hints derived from the live MCP catalog snapshot.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from app.services.mcp_catalog.models import McpCatalogSnapshot, McpToolSpec

_IMPORTANT_DESC_TOKENS = ("required", "always requires", "policy-locked")
_SNAPSHOT_REGISTRY: dict[str, McpCatalogSnapshot] = {}


def build_tool_contract_hints(
    *,
    snapshot: McpCatalogSnapshot,
    allowed_tools: list[str],
    known_facts: dict[str, Any],
    provider: str = "",
) -> list[str]:
    del known_facts
    allowed = tuple(sorted({str(item).strip() for item in allowed_tools if str(item).strip()}))
    _SNAPSHOT_REGISTRY[snapshot.schema_hash] = snapshot
    return list(
        _build_tool_contract_hints_cached(
            schema_hash=snapshot.schema_hash,
            allowed_tools=allowed,
            provider=str(provider or "").strip().lower(),
        )
    )


@lru_cache(maxsize=128)
def _build_tool_contract_hints_cached(
    *,
    schema_hash: str,
    allowed_tools: tuple[str, ...],
    provider: str,
) -> tuple[str, ...]:
    del provider
    snapshot = _SNAPSHOT_REGISTRY[schema_hash]
    hints: list[str] = [
        "If a tool error gives an exact remediation, retry the same tool with corrected arguments instead of switching tools.",
    ]
    for tool_name in allowed_tools:
        tool = snapshot.get_tool(tool_name)
        if tool is None:
            continue
        first_sentence = _first_sentence(tool.description)
        if first_sentence:
            hints.append(f"{tool.name}: {first_sentence}")
        if tool.required_fields:
            hints.append(f"{tool.name}: required fields -> {', '.join(tool.required_fields[:6])}")
        for field_name, field in list(tool.fields.items())[:12]:
            lowered = field.description.lower()
            if field.enum and field_name in {"action", "view", "wait"}:
                rendered = ", ".join(str(item) for item in field.enum[:8])
                hints.append(f"{tool.name}.{field_name}: allowed values -> {rendered}")
            if any(token in lowered for token in _IMPORTANT_DESC_TOKENS):
                snippet = _clean_fragment(field.description)
                if snippet:
                    hints.append(f"{tool.name}.{field_name}: {snippet}")
        hints.extend(_bootstrap_tool_hints(snapshot=snapshot, tool=tool))
    return tuple(_dedupe_preserve_order(hints))


def _bootstrap_tool_hints(*, snapshot: McpCatalogSnapshot, tool: McpToolSpec) -> list[str]:
    manifest = snapshot.capability_manifest or snapshot.bootstrap_manifest
    if not manifest:
        return []
    lowered_name = tool.name.lower()
    hints: list[str] = []
    for text in _flatten_manifest_strings(manifest):
        lowered = text.lower()
        if lowered_name not in lowered:
            continue
        cleaned = _clean_fragment(text)
        if cleaned:
            hints.append(cleaned)
        if len(hints) >= 2:
            break
    return hints


def _flatten_manifest_strings(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            result.append(text)
        return result
    if isinstance(value, dict):
        for item in value.values():
            result.extend(_flatten_manifest_strings(item))
        return result
    if isinstance(value, list):
        for item in value:
            result.extend(_flatten_manifest_strings(item))
    return result


def _first_sentence(text: str) -> str:
    cleaned = _clean_fragment(text)
    if not cleaned:
        return ""
    match = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)
    return match[0].strip()


def _clean_fragment(text: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) > 220:
        cleaned = cleaned[:217] + "..."
    return cleaned


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = str(item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


__all__ = ["build_tool_contract_hints"]
