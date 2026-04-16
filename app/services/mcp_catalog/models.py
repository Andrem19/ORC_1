"""
Runtime models for the live dev_space1 MCP catalog snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class McpFieldSpec:
    name: str
    types: list[str] = field(default_factory=list)
    required: bool = False
    enum: list[Any] = field(default_factory=list)
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    description: str = ""
    title: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    required_fields: list[str] = field(default_factory=list)
    fields: dict[str, McpFieldSpec] = field(default_factory=dict)
    additional_properties: bool | dict[str, Any] | None = None
    cost_class: str = "cheap"
    side_effects: str = "read_only"
    async_pattern: str = ""
    id_fields: list[str] = field(default_factory=list)
    recommended_discovery_flow: list[str] = field(default_factory=list)
    replaces_tools: list[str] = field(default_factory=list)
    deprecated: bool = False
    stability: str = "stable"
    family: str = ""
    accepted_handle_fields: list[str] = field(default_factory=list)
    produced_handle_fields: list[str] = field(default_factory=list)
    supports_terminal_write: bool = False
    supports_discovery: bool = False
    supports_polling: bool = False
    async_like: bool = False

    def to_openai_tool_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (self.description or self.title or self.name)[:900],
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }

    def contract_signature(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "title": self.title,
            "inputSchema": self.input_schema,
        }


@dataclass(frozen=True)
class McpCatalogSnapshot:
    server_name: str
    endpoint_url: str
    schema_hash: str
    fetched_at: str
    tools: list[McpToolSpec] = field(default_factory=list)
    bootstrap_manifest: dict[str, Any] = field(default_factory=dict)
    capability_manifest: dict[str, Any] = field(default_factory=dict)
    manifest_source: str = ""

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def tool_name_set(self) -> set[str]:
        return {tool.name for tool in self.tools}

    def tool_map(self) -> dict[str, McpToolSpec]:
        return {tool.name: tool for tool in self.tools}

    def get_tool(self, tool_name: str) -> McpToolSpec | None:
        return self.tool_map().get(str(tool_name or "").strip())

    def to_prompt_catalog(self) -> list[dict[str, Any]]:
        return [tool.contract_signature() for tool in self.tools]

    def has_tool_name(self, tool_name: str) -> bool:
        return self.get_tool(tool_name) is not None


@dataclass(frozen=True)
class McpCatalogDiff:
    previous_hash: str = ""
    current_hash: str = ""
    added_tools: list[str] = field(default_factory=list)
    removed_tools: list[str] = field(default_factory=list)
    changed_tools: list[str] = field(default_factory=list)
    unchanged: bool = False
    summary: str = ""


__all__ = [
    "McpCatalogDiff",
    "McpCatalogSnapshot",
    "McpFieldSpec",
    "McpToolSpec",
]
