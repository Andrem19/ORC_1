from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

from app.broker.transport import BrokerTransportConfig, BrokerTransportError, McpBrokerTransport


def test_transport_run_with_retries_reopens_after_failure(monkeypatch) -> None:
    transport = McpBrokerTransport(BrokerTransportConfig(retry_budget=1))
    transport._initialized = True  # type: ignore[attr-defined]
    reopen_calls: list[str] = []
    attempts = {"count": 0}

    async def fake_close() -> None:
        reopen_calls.append("close")
        transport._initialized = False  # type: ignore[attr-defined]

    async def fake_open() -> None:
        reopen_calls.append("open")
        transport._initialized = True  # type: ignore[attr-defined]

    async def no_sleep(*_args, **_kwargs) -> None:
        return None

    async def operation():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("boom")
        return "ok"

    monkeypatch.setattr(transport, "close", fake_close)
    monkeypatch.setattr(transport, "open", fake_open)
    monkeypatch.setattr("app.broker.transport.asyncio.sleep", no_sleep)

    result = asyncio.run(transport._run_with_retries(operation))

    assert result == "ok"
    assert reopen_calls == ["close", "open"]
    assert attempts["count"] == 2


def test_transport_run_with_retries_raises_after_budget_exhausted(monkeypatch) -> None:
    transport = McpBrokerTransport(BrokerTransportConfig(retry_budget=1))
    transport._initialized = True  # type: ignore[attr-defined]
    attempts = {"count": 0}

    async def fake_close() -> None:
        transport._initialized = False  # type: ignore[attr-defined]

    async def fake_open() -> None:
        transport._initialized = True  # type: ignore[attr-defined]

    async def no_sleep(*_args, **_kwargs) -> None:
        return None

    async def operation():
        attempts["count"] += 1
        raise RuntimeError("still broken")

    monkeypatch.setattr(transport, "close", fake_close)
    monkeypatch.setattr(transport, "open", fake_open)
    monkeypatch.setattr("app.broker.transport.asyncio.sleep", no_sleep)

    with pytest.raises(RuntimeError, match="still broken"):
        asyncio.run(transport._run_with_retries(operation))

    assert attempts["count"] == 2


def test_transport_open_wraps_initialize_failure_and_cleans_stack(monkeypatch) -> None:
    transport = McpBrokerTransport(BrokerTransportConfig())
    cleanup_events: list[str] = []

    class _FailingSession:
        def __init__(self, *_args) -> None:
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            cleanup_events.append("session_exit")
            return False

        async def initialize(self):
            raise asyncio.CancelledError("connect failed")

    @asynccontextmanager
    async def _fake_streamable_http_client(*_args, **_kwargs):
        cleanup_events.append("stream_enter")
        try:
            yield ("read_stream", "write_stream", lambda: "session_test")
        finally:
            cleanup_events.append("stream_exit")

    monkeypatch.setattr("app.broker.transport._load_mcp_client_session", lambda: _FailingSession)
    monkeypatch.setattr("app.broker.transport._load_streamable_http_client", lambda: _fake_streamable_http_client)

    with pytest.raises(BrokerTransportError, match="broker_transport_open_failed:CancelledError"):
        asyncio.run(transport.open())

    assert cleanup_events == ["stream_enter", "session_exit", "stream_exit"]
    assert transport._initialized is False  # type: ignore[attr-defined]
