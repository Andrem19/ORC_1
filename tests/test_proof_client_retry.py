"""Tests for ProofClient MCP connection retry logic.

Covers the fix where ProofClient.__aenter__ retries transient MCP connection
failures instead of immediately giving up, which was causing successful worker
results to be rejected when the acceptance verifier's MCP connection failed
(e.g., ``Cancelled via cancel scope`` errors from rapid open/close cycles).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.services.direct_execution.acceptance.proof_client import (
    AcceptanceProofInfraError,
    ProofClient,
    _PROOF_CONNECT_RETRIES,
)
from app.services.direct_execution.mcp_client import DirectMcpConfig, DirectMcpError


def _config() -> DirectMcpConfig:
    return DirectMcpConfig(
        endpoint_url="http://localhost:9999/mcp",
        connect_timeout_seconds=1.0,
        read_timeout_seconds=5.0,
        retry_budget=0,
    )


def _run(coro):
    return asyncio.run(coro)


# -- Unit tests -----------------------------------------------------------------


class TestProofClientRetryOpen:
    """ProofClient.__aenter__ retries transient connection failures."""

    def test_succeeds_immediately(self) -> None:
        """When open() succeeds on the first attempt, no retry occurs."""
        config = _config()
        client = ProofClient(config)
        mock_open = AsyncMock()

        with patch.object(ProofClient, "_open_with_retry", mock_open):
            async def _enter_and_exit():
                async with client as pc:
                    pass

            _run(_enter_and_exit())
            assert mock_open.call_count == 1

    def test_retries_on_transient_failure(self) -> None:
        """When open() fails once then succeeds, the connection is established."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:Cancelled via cancel scope abc123")

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=[direct_error, None])
            instance.close = AsyncMock()

            async def _go():
                async with client as pc:
                    pass

            _run(_go())
            assert instance.open.call_count == 2

    def test_raises_after_exhausting_retries(self) -> None:
        """After all retries are exhausted, AcceptanceProofInfraError is raised."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:Cancelled via cancel scope xyz")

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=direct_error)
            instance.close = AsyncMock()

            async def _go():
                with pytest.raises(AcceptanceProofInfraError, match="acceptance_mcp_open_failed"):
                    async with client as pc:
                        pass

            _run(_go())
            assert instance.open.call_count == _PROOF_CONNECT_RETRIES

    def test_succeeds_on_final_retry(self) -> None:
        """When all-but-one retries fail, the last attempt succeeds."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:Connection reset")
        open_side_effects = [direct_error] * (_PROOF_CONNECT_RETRIES - 1) + [None]

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=open_side_effects)
            instance.close = AsyncMock()

            async def _go():
                async with client as pc:
                    pass

            _run(_go())
            assert instance.open.call_count == _PROOF_CONNECT_RETRIES

    def test_creates_new_client_on_retry(self) -> None:
        """On each retry, a fresh DirectMcpClient is created for clean state."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:Cancelled via cancel scope")

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=[direct_error, None])
            instance.close = AsyncMock()

            async def _go():
                async with client:
                    pass

            _run(_go())
            # DirectMcpClient should be constructed once for init + once for retry
            assert MockClient.call_count >= 2

    def test_call_proof_still_works_after_retry_open(self) -> None:
        """After a successful retry, call_proof dispatches to the MCP client."""
        config = _config()
        client = ProofClient(config)

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock()  # succeeds immediately
            instance.close = AsyncMock()
            instance.call_tool = AsyncMock(
                return_value={"content": [{"type": "text", "text": '{"status": "pass"}'}]}
            )

            async def _go():
                async with client as pc:
                    result = await pc.call_proof("research_memory", {"action": "prove"})
                    assert isinstance(result, dict)

            _run(_go())

    def test_exit_closes_client_even_after_retry(self) -> None:
        """__aexit__ closes the MCP client even when retries occurred."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:timeout")

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=[direct_error, None])
            instance.close = AsyncMock()

            async def _go():
                async with client:
                    pass

            _run(_go())
            # The final client should be closed on exit
            instance.close.assert_called()

    def test_log_on_successful_retry(self, capsys) -> None:
        """Successful retry after initial failure logs an info message."""
        config = _config()
        client = ProofClient(config)
        direct_error = DirectMcpError("direct_mcp_open_failed:Cancelled via cancel scope")

        with patch(
            "app.services.direct_execution.acceptance.proof_client.DirectMcpClient"
        ) as MockClient:
            instance = MockClient.return_value
            instance.open = AsyncMock(side_effect=[direct_error, None])
            instance.close = AsyncMock()

            async def _go():
                async with client:
                    pass

            _run(_go())
            # No exception = success
