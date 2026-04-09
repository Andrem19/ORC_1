"""
Async MCP broker transport compatible with Python 3.10.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import httpx

logger = logging.getLogger("orchestrator.broker.transport")


@dataclass
class BrokerTransportConfig:
    endpoint_url: str = "http://127.0.0.1:8766/mcp"
    auth_mode: str = "none"  # none | bearer
    token_env_var: str = "DEV_SPACE1_MCP_BEARER_TOKEN"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    retry_budget: int = 2


class AsyncBrokerTransport(Protocol):
    def validate_runtime_requirements(self) -> None:
        ...

    async def open(self) -> None:
        ...

    async def close(self) -> None:
        ...

    async def initialize(self) -> Any:
        ...

    async def list_tools(self) -> Any:
        ...

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        ...

    def session_id(self) -> str:
        ...


class BrokerTransportError(RuntimeError):
    """Raised when the MCP transport cannot be opened or safely used."""


class McpBrokerTransport:
    def __init__(self, config: BrokerTransportConfig) -> None:
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
                raise RuntimeError(f"broker_auth_token_missing:{self.config.token_env_var}")
            headers["Authorization"] = f"Bearer {token}"
        http_client = httpx.AsyncClient(timeout=timeout, headers=headers or None)
        stream_context = None
        session = None
        try:
            streamable_http_client = _load_streamable_http_client()
            stream_context = streamable_http_client(self.config.endpoint_url, http_client=http_client)
            read_stream, write_stream, get_session_id = await stream_context.__aenter__()
            session_cls = _load_mcp_client_session()
            session = session_cls(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
        except BaseException as exc:
            await self._close_components_quietly(
                http_client=http_client,
                stream_context=stream_context,
                session=session,
                context="open_failed",
            )
            self._http_client = None
            self._stream_context = None
            self._session = None
            self._session_id_getter = None
            self._initialized = False
            raise BrokerTransportError(
                f"broker_transport_open_failed:{exc.__class__.__name__}: {exc}"
            ) from exc
        self._http_client = http_client
        self._stream_context = stream_context
        self._session = session
        self._session_id_getter = get_session_id
        self._initialized = True
        logger.info("Broker transport connected: %s", self.config.endpoint_url)

    async def close(self) -> None:
        http_client = self._http_client
        stream_context = self._stream_context
        session = self._session
        self._http_client = None
        self._stream_context = None
        self._session = None
        self._session_id_getter = None
        self._initialized = False
        await self._close_components_quietly(
            http_client=http_client,
            stream_context=stream_context,
            session=session,
            context="close",
        )

    async def initialize(self) -> Any:
        async with self._request_lock:
            return await self._run_with_retries(lambda: self._session.initialize())

    async def list_tools(self) -> Any:
        async with self._request_lock:
            return await self._run_with_retries(lambda: self._session.list_tools())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Any:
        async with self._request_lock:
            return await self._run_with_retries(lambda: self._session.call_tool(tool_name, arguments=arguments or {}))

    def session_id(self) -> str:
        if self._session_id_getter is None:
            return ""
        try:
            return str(self._session_id_getter() or "")
        except Exception:
            return ""

    async def _run_with_retries(self, operation: Callable[[], Awaitable[Any]]) -> Any:
        if not self._initialized:
            await self.open()
        last_error: Exception | None = None
        for attempt in range(self.config.retry_budget + 1):
            try:
                return await operation()
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Broker transport operation failed on attempt %d/%d: %s",
                    attempt + 1,
                    self.config.retry_budget + 1,
                    exc,
                )
                if attempt >= self.config.retry_budget:
                    break
                await self.close()
                await asyncio.sleep(min(0.25 * (2**attempt), 1.0))
                await self.open()
        assert last_error is not None
        raise last_error

    async def _close_components_quietly(
        self,
        *,
        http_client: httpx.AsyncClient | None,
        stream_context: Any,
        session: Any,
        context: str,
    ) -> None:
        try:
            if session is not None:
                await session.__aexit__(None, None, None)
        except BaseException as exc:
            logger.debug("Broker transport session cleanup failed during %s: %s", context, exc)
        try:
            if stream_context is not None:
                await stream_context.__aexit__(None, None, None)
        except BaseException as exc:
            logger.debug("Broker transport stream cleanup failed during %s: %s", context, exc)
        try:
            if http_client is not None:
                await http_client.aclose()
        except BaseException as exc:
            logger.debug("Broker transport http client cleanup failed during %s: %s", context, exc)


def _load_mcp_client_session() -> Any:
    try:
        from mcp import ClientSession
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Official MCP Python SDK is required for broker transport. Install package `mcp` in the active runtime."
        ) from exc
    return ClientSession


def _load_streamable_http_client() -> Any:
    try:
        from mcp.client.streamable_http import streamable_http_client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Official MCP Python SDK is required for broker transport. Install package `mcp` in the active runtime."
        ) from exc
    return streamable_http_client


__all__ = ["AsyncBrokerTransport", "BrokerTransportConfig", "BrokerTransportError", "McpBrokerTransport"]
