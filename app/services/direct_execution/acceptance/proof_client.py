"""Read-only MCP proof client used by acceptance verifiers."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig, DirectMcpError, _to_jsonable

logger = logging.getLogger("orchestrator.direct.acceptance.proof")

_PROOF_CONNECT_RETRIES = 3
_PROOF_CONNECT_BASE_DELAY = 0.5


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
        await self._open_with_retry()
        return self

    async def _open_with_retry(self) -> None:
        """Open the MCP connection with retries for transient failures.

        The acceptance verifier often runs immediately after the tool loop
        closes its own MCP connection.  Rapid open/close cycles to the same
        MCP endpoint can produce transient ``Cancelled via cancel scope``
        errors from the anyio-backed MCP SDK.  Retrying with a short delay
        avoids falling back to a full slice re-execution for what is purely
        a momentary connection hiccup.
        """
        assert self.client is not None
        last_error: Exception | None = None
        for attempt in range(_PROOF_CONNECT_RETRIES):
            try:
                await self.client.open()
                if attempt > 0:
                    logger.info(
                        "Proof MCP connection succeeded on attempt %d/%d",
                        attempt + 1, _PROOF_CONNECT_RETRIES,
                    )
                return
            except DirectMcpError as exc:
                last_error = exc
                if attempt < _PROOF_CONNECT_RETRIES - 1:
                    delay = min(_PROOF_CONNECT_BASE_DELAY * (2 ** attempt), 4.0)
                    logger.warning(
                        "Proof MCP connection failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, _PROOF_CONNECT_RETRIES, delay, exc,
                    )
                    await asyncio.sleep(delay)
                    # Reset client for a clean reconnection attempt
                    await self.client.close()
                    self.client = DirectMcpClient(self.config)
        raise AcceptanceProofInfraError(
            f"acceptance_mcp_open_failed_after_{_PROOF_CONNECT_RETRIES}_retries: {last_error}"
        ) from last_error

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
