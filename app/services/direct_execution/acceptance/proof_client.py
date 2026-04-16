"""Read-only MCP proof client used by acceptance verifiers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig, DirectMcpError, _to_jsonable


class AcceptanceProofInfraError(RuntimeError):
    """Raised when MCP proof collection fails for infrastructure reasons."""


@dataclass
class ProofClient:
    config: DirectMcpConfig
    client: DirectMcpClient | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def __aenter__(self) -> "ProofClient":
        if self.client is None:
            self.client = DirectMcpClient(self.config)
        await self.client.open()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.client is not None:
            await self.client.close()

    async def call_proof(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            assert self.client is not None
            raw = _to_jsonable(await self.client.call_tool(tool_name, arguments))
        except DirectMcpError as exc:
            raise AcceptanceProofInfraError(str(exc)) from exc
        except Exception as exc:
            raise AcceptanceProofInfraError(f"acceptance_mcp_call_failed:{exc}") from exc
        payload = _extract_structured_payload(raw)
        self.calls.append({"tool": tool_name, "arguments": dict(arguments), "payload": payload})
        return payload


def build_proof_client_config(direct_config: Any) -> DirectMcpConfig:
    return DirectMcpConfig(
        endpoint_url=str(getattr(direct_config, "mcp_endpoint_url", "")),
        auth_mode=str(getattr(direct_config, "mcp_auth_mode", "none")),
        token_env_var=str(getattr(direct_config, "mcp_token_env_var", "DEV_SPACE1_MCP_BEARER_TOKEN")),
        connect_timeout_seconds=float(getattr(direct_config, "connect_timeout_seconds", 10.0) or 10.0),
        read_timeout_seconds=float(getattr(direct_config, "read_timeout_seconds", 60.0) or 60.0),
        retry_budget=int(getattr(direct_config, "retry_budget", 1) or 0),
    )


def _extract_structured_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        # Claude MCP protocol: structuredContent at top level
        if isinstance(raw.get("structuredContent"), dict):
            return dict(raw["structuredContent"])
        content = raw.get("content")
        if isinstance(content, list):
            # Claude MCP protocol: nested structuredContent
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("structuredContent"), dict):
                    return dict(item["structuredContent"])
            # Standard MCP SDK: content[].text contains JSON string
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text and text.startswith("{"):
                        try:
                            parsed = json.loads(text)
                            if isinstance(parsed, dict):
                                return parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
        if isinstance(raw.get("payload"), dict):
            return _extract_structured_payload(raw["payload"])
        return raw
    return {"raw": raw}


__all__ = ["AcceptanceProofInfraError", "ProofClient", "build_proof_client_config"]
