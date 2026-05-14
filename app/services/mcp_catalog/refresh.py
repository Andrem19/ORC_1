"""
Startup refresh for the live MCP catalog snapshot.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig, DirectMcpError, _to_jsonable
from app.services.mcp_catalog.diff import diff_catalog_snapshots
from app.services.mcp_catalog.models import McpCatalogDiff, McpCatalogSnapshot
from app.services.mcp_catalog.normalizer import build_catalog_snapshot
from app.services.mcp_catalog.store import McpCatalogStore

logger = logging.getLogger("orchestrator.mcp_catalog")


class McpCatalogUnavailableError(RuntimeError):
    """Raised when the live MCP catalog cannot be fetched at startup."""


@dataclass(frozen=True)
class McpCatalogRefreshResult:
    snapshot: McpCatalogSnapshot
    diff: McpCatalogDiff
    saved_paths: dict[str, Any]


class McpCatalogRefreshService:
    def __init__(
        self,
        *,
        mcp_config: DirectMcpConfig,
        store: McpCatalogStore,
        artifact_root: str = "",
    ) -> None:
        self.mcp_config = mcp_config
        self.store = store
        self.artifact_root = artifact_root

    async def refresh(self) -> McpCatalogRefreshResult:
        previous = self.store.load_latest()
        client = DirectMcpClient(self.mcp_config)
        try:
            tools = await client.list_tools()
        except Exception as exc:
            try:
                await client.close()
            except Exception:
                pass
            raise McpCatalogUnavailableError(f"mcp_catalog_unavailable:{exc}") from exc
        capability_manifest, capability_source = await _best_effort_manifest(client, tool_names={tool.get("name", "") for tool in tools})
        bootstrap_manifest = {}
        bootstrap_source = ""
        if not capability_manifest:
            bootstrap_manifest, bootstrap_source = await _best_effort_bootstrap_manifest(client, tool_names={tool.get("name", "") for tool in tools})
        snapshot = build_catalog_snapshot(
            tools=tools,
            endpoint_url=self.mcp_config.endpoint_url,
            bootstrap_manifest=bootstrap_manifest,
            capability_manifest=capability_manifest,
            manifest_source=capability_source or bootstrap_source,
        )
        diff = diff_catalog_snapshots(previous, snapshot)
        saved_paths = self.store.save_snapshot(snapshot, diff=diff, artifact_root=self.artifact_root or None)
        try:
            await client.close()
        except Exception:
            pass
        return McpCatalogRefreshResult(snapshot=snapshot, diff=diff, saved_paths=saved_paths)

    async def capture_remote_incident(self, *, summary: str, metadata: dict[str, Any]) -> bool:
        client = DirectMcpClient(self.mcp_config)
        try:
            payload = await client.call_tool(
                "incidents",
                {
                    "action": "capture",
                    "summary": summary[:240],
                    "severity": "high",
                    "service": "orchestrator",
                    "affected_tool": "dev_space1",
                    "metadata": metadata,
                },
            )
            del payload
            return True
        except Exception as exc:
            logger.warning("Remote MCP incident capture failed: %s", exc)
            return False
        finally:
            try:
                await client.close()
            except Exception:
                pass


async def _best_effort_manifest(client: DirectMcpClient, *, tool_names: set[str]) -> tuple[dict[str, Any], str]:
    if "system_capabilities" not in tool_names:
        return {}, ""
    try:
        payload = _to_jsonable(await client.call_tool("system_capabilities", {"view": "raw"}))
        parsed = _parse_manifest_payload(payload)
        return (parsed if isinstance(parsed, dict) else {}), "system_capabilities"
    except Exception as exc:
        logger.warning("system_capabilities(view='raw') unavailable: %s", exc)
        return {}, ""


async def _best_effort_bootstrap_manifest(client: DirectMcpClient, *, tool_names: set[str]) -> tuple[dict[str, Any], str]:
    if "system_bootstrap" not in tool_names:
        return {}, ""
    try:
        payload = _to_jsonable(await client.call_tool("system_bootstrap", {"view": "raw"}))
        parsed = _parse_manifest_payload(payload)
        return (parsed if isinstance(parsed, dict) else {}), "system_bootstrap"
    except Exception as exc:
        logger.warning("system_bootstrap(view='raw') unavailable: %s", exc)
        return {}, ""


def _parse_manifest_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        structured = payload.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        for content_item in payload.get("content", []) or []:
            if not isinstance(content_item, dict):
                continue
            text = str(content_item.get("text", "") or "").strip()
            parsed = _extract_json_object(text)
            if isinstance(parsed, dict):
                return parsed
    if isinstance(payload, str):
        parsed = _extract_json_object(payload)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    direct = _try_json_loads(stripped)
    if isinstance(direct, dict):
        return direct
    fenced = stripped.replace("```json", "```")
    if "```" in fenced:
        parts = fenced.split("```")
        for part in parts:
            parsed = _try_json_loads(part.strip())
            if isinstance(parsed, dict):
                return parsed
    first = stripped.find("{")
    last = stripped.rfind("}")
    if first >= 0 and last > first:
        parsed = _try_json_loads(stripped[first:last + 1])
        if isinstance(parsed, dict):
            return parsed
    return None


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


__all__ = [
    "McpCatalogRefreshResult",
    "McpCatalogRefreshService",
    "McpCatalogUnavailableError",
]
