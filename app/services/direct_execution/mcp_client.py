"""
Minimal direct MCP client for dev_space1 tool access.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

import httpx

logger = logging.getLogger("orchestrator.direct.mcp")


@dataclass
class DirectMcpConfig:
    endpoint_url: str
    auth_mode: str = "none"
    token_env_var: str = "DEV_SPACE1_MCP_BEARER_TOKEN"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    retry_budget: int = 1


class DirectMcpError(RuntimeError):
    """Raised when direct MCP transport fails."""


class DirectMcpClient:
    def __init__(self, config: DirectMcpConfig) -> None:
        self.config = config
        self._http_client: httpx.AsyncClient | None = None
        self._stream_context: Any = None
        self._session: Any = None
        self._session_id_getter: Callable[[], str] | None = None
        self._initialized = False
        self._request_lock = asyncio.Lock()

    def validate_runtime_requirements(self) -> None:
        _load_mcp_client_session()
        _load_streamable_http_client()

    async def open(self) -> None:
        if self._initialized:
            return
        self.validate_runtime_requirements()
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.read_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )
        headers: dict[str, str] = {}
        token = os.environ.get(self.config.token_env_var, "").strip()
        if self.config.auth_mode == "bearer":
            if not token:
                raise DirectMcpError(f"direct_mcp_auth_token_missing:{self.config.token_env_var}")
            headers["Authorization"] = f"Bearer {token}"
        http_client = httpx.AsyncClient(timeout=timeout, headers=headers or None)
        try:
            stream_context = _load_streamable_http_client()(self.config.endpoint_url, http_client=http_client)
            read_stream, write_stream, get_session_id = await stream_context.__aenter__()
            session = _load_mcp_client_session()(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
        except BaseException as exc:
            await _close_quietly(http_client=http_client, stream_context=locals().get("stream_context"), session=locals().get("session"))
            raise DirectMcpError(f"direct_mcp_open_failed:{exc}") from exc
        self._http_client = http_client
        self._stream_context = stream_context
        self._session = session
        self._session_id_getter = get_session_id
        self._initialized = True
        logger.info("Direct MCP connected: %s", self.config.endpoint_url)

    async def close(self) -> None:
        http_client = self._http_client
        stream_context = self._stream_context
        session = self._session
        self._http_client = None
        self._stream_context = None
        self._session = None
        self._session_id_getter = None
        self._initialized = False
        await _close_quietly(http_client=http_client, stream_context=stream_context, session=session)

    async def list_tools(self) -> list[dict[str, Any]]:
        async with self._request_lock:
            payload = await self._run(lambda: self._session.list_tools())
        tools = getattr(payload, "tools", payload)
        result: list[dict[str, Any]] = []
        for item in tools or []:
            data = _to_jsonable(item)
            name = str(data.get("name", "") or "").strip()
            if not name:
                continue
            result.append(data)
        return result

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        async with self._request_lock:
            return await self._run(lambda: self._session.call_tool(tool_name, arguments=arguments or {}))

    def session_id(self) -> str:
        if self._session_id_getter is None:
            return ""
        try:
            return str(self._session_id_getter() or "")
        except Exception:
            return ""

    async def _run(self, operation: Callable[[], Any]) -> Any:
        if not self._initialized:
            await self.open()
        last_error: Exception | None = None
        for attempt in range(max(0, int(self.config.retry_budget or 0)) + 1):
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                logger.warning("Direct MCP operation failed on attempt %d: %s", attempt + 1, exc)
                if attempt >= int(self.config.retry_budget or 0):
                    break
                await self.close()
                await asyncio.sleep(min(0.25 * (2 ** attempt), 1.0))
                await self.open()
        assert last_error is not None
        raise DirectMcpError(f"direct_mcp_operation_failed:{last_error}") from last_error


async def _close_quietly(*, http_client: httpx.AsyncClient | None, stream_context: Any, session: Any) -> None:
    try:
        if session is not None:
            await session.__aexit__(None, None, None)
    except BaseException:
        pass
    try:
        if stream_context is not None:
            await stream_context.__aexit__(None, None, None)
    except BaseException:
        pass
    try:
        if http_client is not None:
            await http_client.aclose()
    except Exception:
        pass


def _load_mcp_client_session() -> Any:
    try:
        from mcp import ClientSession
    except Exception as exc:
        raise DirectMcpError("Official MCP Python SDK is required for direct MCP execution.") from exc
    return ClientSession


def _load_streamable_http_client() -> Any:
    try:
        from mcp.client.streamable_http import streamable_http_client
    except Exception as exc:
        raise DirectMcpError("Official MCP Python SDK streamable HTTP client is required for direct MCP execution.") from exc
    return streamable_http_client


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return {key: _to_jsonable(item) for key, item in vars(value).items() if not key.startswith("_")}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


__all__ = ["DirectMcpClient", "DirectMcpConfig", "DirectMcpError", "_to_jsonable"]
