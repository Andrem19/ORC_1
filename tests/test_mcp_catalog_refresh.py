from __future__ import annotations

import asyncio
import json

from app.services.direct_execution.mcp_client import DirectMcpConfig
from app.services.mcp_catalog.refresh import (
    McpCatalogRefreshService,
    _parse_manifest_payload,
)
from app.services.mcp_catalog.store import McpCatalogStore


class _FakeClient:
    def __init__(self, config, *, tools, bootstrap_text="") -> None:
        self.config = config
        self._tools = tools
        self._bootstrap_text = bootstrap_text

    async def list_tools(self):
        return list(self._tools)

    async def call_tool(self, tool_name, arguments):
        del arguments
        if tool_name == "system_bootstrap" and self._bootstrap_text:
            return {"content": [{"type": "text", "text": self._bootstrap_text}]}
        raise RuntimeError(f"unexpected tool call: {tool_name}")

    async def close(self):
        return None


def _service(tmp_path, monkeypatch, *, tools, bootstrap_text=""):
    def _factory(config):
        return _FakeClient(config, tools=tools, bootstrap_text=bootstrap_text)

    monkeypatch.setattr("app.services.mcp_catalog.refresh.DirectMcpClient", _factory)
    return McpCatalogRefreshService(
        mcp_config=DirectMcpConfig(endpoint_url="http://127.0.0.1:8766/mcp"),
        store=McpCatalogStore(tmp_path / "state", run_id="run_1"),
        artifact_root=str(tmp_path / "artifacts"),
    )


def test_refresh_service_writes_stable_snapshot_and_reuses_hash(tmp_path, monkeypatch) -> None:
    tools = [
        {
            "name": "research_search",
            "description": "Search one research project.",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        }
    ]
    service = _service(tmp_path, monkeypatch, tools=tools)

    first = asyncio.run(service.refresh())
    second = asyncio.run(service.refresh())

    assert first.snapshot.schema_hash == second.snapshot.schema_hash
    assert first.saved_paths["latest"].exists()
    assert first.saved_paths["history"].exists()
    latest_payload = json.loads(first.saved_paths["latest"].read_text(encoding="utf-8"))
    assert latest_payload["schema_hash"] == first.snapshot.schema_hash
    assert (tmp_path / "artifacts" / "mcp_catalog" / "snapshot.json").exists()


def test_refresh_service_diff_detects_added_and_changed_tools(tmp_path, monkeypatch) -> None:
    first_tools = [
        {"name": "research_search", "description": "Search.", "inputSchema": {"type": "object", "properties": {}}},
    ]
    second_tools = [
        {"name": "research_search", "description": "Search updated.", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "research_map", "description": "Inspect atlas.", "inputSchema": {"type": "object", "properties": {}}},
    ]
    service = _service(tmp_path, monkeypatch, tools=first_tools)
    first = asyncio.run(service.refresh())
    service = _service(tmp_path, monkeypatch, tools=second_tools)
    second = asyncio.run(service.refresh())

    assert first.snapshot.schema_hash != second.snapshot.schema_hash
    assert second.diff.added_tools == ["research_map"]
    assert second.diff.changed_tools == ["research_search"]
    assert second.diff.removed_tools == []


def test_parse_bootstrap_manifest_from_text_wrapped_json() -> None:
    payload = {
        "content": [
            {
                "type": "text",
                "text": "system bootstrap\n```json\n{\"tools\": [{\"name\": \"research_map\", \"hint\": \"Use inspect first.\"}]}\n```",
            }
        ]
    }

    parsed = _parse_manifest_payload(payload)

    assert parsed["tools"][0]["name"] == "research_map"
    assert parsed["tools"][0]["hint"] == "Use inspect first."
