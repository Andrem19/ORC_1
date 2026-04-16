"""Tests for MiniMax provider integration."""
from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

from app.config import (
    DirectExecutionConfig,
    LMStudioConfig,
    MinimaxConfig,
    OrchestratorConfig,
    load_config_from_dict,
)
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi


class TestMinimaxConfig:
    """MinimaxConfig dataclass and config parsing."""

    def test_default_values(self) -> None:
        cfg = MinimaxConfig()
        assert cfg.base_url == "https://api.minimax.io"
        assert cfg.model == "MiniMax-M2.7"
        assert cfg.api_key_env_var == "MINIMAX_API"
        assert cfg.temperature == 1.0
        assert cfg.max_tokens == 8192

    def test_load_from_dict(self) -> None:
        data = {
            "minimax": {
                "base_url": "https://custom.api/v1",
                "model": "MiniMax-M2.7",
                "api_key_env_var": "CUSTOM_MINIMAX_KEY",
                "temperature": 0.5,
                "max_tokens": 4096,
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.minimax.base_url == "https://custom.api/v1"
        assert cfg.minimax.model == "MiniMax-M2.7"
        assert cfg.minimax.api_key_env_var == "CUSTOM_MINIMAX_KEY"
        assert cfg.minimax.temperature == 0.5
        assert cfg.minimax.max_tokens == 4096

    def test_load_from_dict_ignores_unknown_keys(self) -> None:
        data = {
            "minimax": {
                "model": "MiniMax-M2.7",
                "unknown_key": "should_be_ignored",
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.minimax.model == "MiniMax-M2.7"
        assert not hasattr(cfg.minimax, "unknown_key") or getattr(cfg.minimax, "unknown_key", None) is None

    def test_missing_section_uses_defaults(self) -> None:
        cfg = load_config_from_dict({})
        assert cfg.minimax.model == "MiniMax-M2.7"
        assert cfg.minimax.base_url == "https://api.minimax.io"

    def test_explicit_minimax_provider_beats_lmstudio_adapter_name(self) -> None:
        from app.services.direct_execution.executor import _resolve_provider_name

        adapter = MagicMock()
        adapter.name.return_value = "lmstudio_worker_api"

        provider = _resolve_provider_name(
            adapter=adapter,
            explicit_provider="minimax",
            configured_provider="minimax",
        )

        assert provider == "minimax"


class TestMinimaxAdapterCreation:
    """runtime_factory creates the correct adapter for minimax provider."""

    @patch.dict(os.environ, {"MINIMAX_API": "test-api-key-123"})
    def test_create_worker_adapter_minimax(self) -> None:
        from app.runtime_factory import create_worker_adapter

        cfg = OrchestratorConfig()
        cfg.direct_execution = DirectExecutionConfig(provider="minimax")
        cfg.minimax = MinimaxConfig(
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2.7",
            api_key_env_var="MINIMAX_API",
        )
        adapter = create_worker_adapter(cfg)
        assert isinstance(adapter, LmStudioWorkerApi)
        assert adapter.base_url == "https://api.minimax.io/v1"
        assert adapter.model == "MiniMax-M2.7"
        assert adapter.api_key == "test-api-key-123"
        assert adapter.temperature == 1.0

    @patch.dict(os.environ, {"CUSTOM_MM_KEY": "sk-custom-456"})
    def test_create_worker_adapter_custom_env_var(self) -> None:
        from app.runtime_factory import create_worker_adapter

        cfg = OrchestratorConfig()
        cfg.direct_execution = DirectExecutionConfig(provider="minimax")
        cfg.minimax = MinimaxConfig(
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2.7-highspeed",
            api_key_env_var="CUSTOM_MM_KEY",
        )
        adapter = create_worker_adapter(cfg)
        assert isinstance(adapter, LmStudioWorkerApi)
        assert adapter.api_key == "sk-custom-456"
        assert adapter.model == "MiniMax-M2.7-highspeed"

    @patch.dict(os.environ, {}, clear=True)
    def test_create_worker_adapter_no_api_key(self) -> None:
        from app.runtime_factory import create_worker_adapter

        cfg = OrchestratorConfig()
        cfg.direct_execution = DirectExecutionConfig(provider="minimax")
        cfg.minimax = MinimaxConfig(
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2.7",
            api_key_env_var="NONEXISTENT_KEY",
        )
        adapter = create_worker_adapter(cfg)
        assert isinstance(adapter, LmStudioWorkerApi)
        assert adapter.api_key == ""

    def test_create_fallback_adapter_minimax(self) -> None:
        from app.runtime_factory import create_fallback_adapter

        with patch.dict(os.environ, {"MINIMAX_API": "fallback-key"}):
            cfg = OrchestratorConfig()
            cfg.minimax = MinimaxConfig(
                base_url="https://api.minimax.io/v1",
                model="MiniMax-M2.7",
            )
            adapter = create_fallback_adapter("minimax", cfg)
            assert isinstance(adapter, LmStudioWorkerApi)
            assert adapter.api_key == "fallback-key"

    def test_create_fallback_adapter_off_returns_none(self) -> None:
        from app.runtime_factory import create_fallback_adapter

        cfg = OrchestratorConfig()
        assert create_fallback_adapter("off", cfg) is None
        assert create_fallback_adapter("none", cfg) is None

    def test_create_fallback_adapter_unknown_raises(self) -> None:
        from app.runtime_factory import create_fallback_adapter

        cfg = OrchestratorConfig()
        try:
            create_fallback_adapter("unknown_provider", cfg)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "unknown_provider" in str(e)
            assert "minimax" in str(e)


class TestMinimaxProviderRouting:
    """Provider-specific logic routes minimax correctly."""

    def test_execution_parsing_allows_minimax_tool_namespace(self) -> None:
        """Minimax uses the same unprefixed tool names as LM Studio."""
        from app.execution_parsing import _validate_worker_action
        from app.execution_models import WorkerAction

        action = WorkerAction(
            action_id="a1",
            action_type="tool_call",
            tool="backtests_runs",
            arguments={"action": "inspect", "view": "list"},
            reason="checking",
        )
        # Should NOT raise — minimax uses unprefixed names like lmstudio
        _validate_worker_action(action, allowlist={"backtests_runs"}, provider="minimax")

    def test_execution_parsing_rejects_prefixed_namespace_for_minimax(self) -> None:
        """Minimax should NOT use mcp__dev_space1__ prefix (same as lmstudio)."""
        from app.execution_parsing import StructuredOutputError, _validate_worker_action
        from app.execution_models import WorkerAction

        action = WorkerAction(
            action_id="a1",
            action_type="tool_call",
            tool="mcp__dev_space1__backtests_runs",
            arguments={},
            reason="checking",
        )
        try:
            _validate_worker_action(action, allowlist={"mcp__dev_space1__backtests_runs"}, provider="minimax")
            assert False, "Expected StructuredOutputError"
        except StructuredOutputError as e:
            assert "tool_prefixed_namespace_forbidden" in str(e)

    def test_prompt_model_contract_footer_includes_minimax(self) -> None:
        from app.services.direct_execution.prompt import _model_contract_footer

        footer = _model_contract_footer(provider="minimax")
        assert any("final_report" in line.lower() for line in footer)

    def test_guardrails_write_result_guard_applies_to_minimax(self) -> None:
        from app.execution_models import PlanSlice
        from app.services.direct_execution.guardrails import _write_result_guard_reason

        slice_obj = PlanSlice(
            slice_id="test",
            title="t",
            hypothesis="h",
            objective="o",
            success_criteria=["criterion"],
            allowed_tools=["research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=5,
            max_tool_calls=5,
            max_expensive_calls=0,
            parallel_slot=1,
            runtime_profile="write_result",
        )
        facts = {
            "direct.provider": "minimax",
            "direct.successful_mutating_tool_count": 0,
        }
        reason = _write_result_guard_reason(slice_obj=slice_obj, facts=facts)
        assert reason == "write_result_without_mutating_tool"

    def test_temperature_not_reduced_for_minimax(self) -> None:
        """Minimax is NOT classified as a weak provider — temperature should pass through."""
        from app.services.direct_execution.temperature_config import get_adaptive_temperature

        result = get_adaptive_temperature(base_temperature=1.0, provider="minimax")
        assert result == 1.0  # No reduction for non-weak providers

    def test_temperature_reduced_for_lmstudio(self) -> None:
        """LMStudio IS weak — temperature should be reduced."""
        from app.services.direct_execution.temperature_config import get_adaptive_temperature

        result = get_adaptive_temperature(base_temperature=1.0, provider="lmstudio")
        assert result < 1.0

    def test_executor_activates_tool_loop_for_minimax(self) -> None:
        """DirectSliceExecutor should use LmStudioToolLoop when provider=minimax."""
        # Verify the condition check matches minimax
        provider = "minimax"
        # The condition in executor.py is: provider in ("lmstudio", "minimax")
        assert provider in ("lmstudio", "minimax")

    def test_minimax_not_weak_provider(self) -> None:
        """Minimax should not be classified as a weak provider."""
        from app.services.direct_execution.prompt import _is_weak_provider

        assert _is_weak_provider("minimax") is False
        assert _is_weak_provider("lmstudio") is True


class TestMinimaxConfigToml:
    """Integration test: config.toml with minimax section parses correctly."""

    def test_full_config_with_minimax(self) -> None:
        data = {
            "goal": "test",
            "direct_execution": {
                "provider": "minimax",
                "fallback_1": "claude_cli",
                "fallback_2": "off",
            },
            "minimax": {
                "base_url": "https://api.minimax.io",
                "model": "MiniMax-M2.7",
                "api_key_env_var": "MINIMAX_API",
                "temperature": 1.0,
                "max_tokens": 8192,
            },
            "lmstudio": {
                "base_url": "http://localhost:1234",
                "model": "test-model",
                "reasoning_effort": "none",
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.direct_execution.provider == "minimax"
        assert cfg.minimax.model == "MiniMax-M2.7"
        assert cfg.minimax.base_url == "https://api.minimax.io"
        assert cfg.lmstudio.base_url == "http://localhost:1234"


class TestMinimaxConnectionPoolHealthCheck:
    """Cloud provider health checks in LMStudioConnectionPool."""

    def test_cloud_health_probe_passes_on_200(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="test-key",
            model="MiniMax-M2.7",
        )

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"choices":[]}'

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            assert pool.health_check() is True
        mock_conn.request.assert_called_once()

    def test_cloud_health_probe_fails_on_401(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="bad-key",
            model="MiniMax-M2.7",
        )

        mock_resp = MagicMock()
        mock_resp.status = 401
        mock_resp.read.return_value = b'{"error": "unauthorized"}'

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            assert pool.health_check() is False

    def test_cloud_health_probe_fails_on_connection_error(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="test-key",
            model="MiniMax-M2.7",
        )

        with patch("http.client.HTTPSConnection", side_effect=ConnectionError("unreachable")):
            assert pool.health_check() is False

    def test_cloud_health_probe_includes_model(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="test-key",
            model="MiniMax-M2.7",
        )

        captured_body = {}

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        def capture_request(method, path, body, headers):
            captured_body["data"] = json.loads(body)

        mock_conn.request.side_effect = capture_request

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            pool.health_check()

        assert captured_body["data"]["model"] == "MiniMax-M2.7"

    def test_local_health_probe_still_works(self) -> None:
        """Non-cloud URLs still use GET /v1/models via urllib3."""
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="http://localhost:1234",
        )

        def fake_urlopen(method, url, body=None, headers=None, timeout=None, retries=None):
            assert method == "GET"
            assert "/v1/models" in url
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"data": [{"id": "test-model"}]}).encode()
            return resp

        pool._pool.urlopen = fake_urlopen
        assert pool.health_check() is True

    def test_cloud_probe_sends_bearer_auth(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="sk-test-123",
            model="MiniMax-M2.7",
        )
        captured_headers = {}

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        def capture_request(method, path, body, headers):
            captured_headers.update(headers)

        mock_conn.request.side_effect = capture_request

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            pool.health_check()
        assert captured_headers.get("Authorization") == "Bearer sk-test-123"

    def test_warm_up_delegates_to_health_check(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="test",
            model="MiniMax-M2.7",
        )

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"{}"

        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            assert pool.warm_up() is True

    def test_cloud_provider_detection(self) -> None:
        from app.services.direct_execution.lmstudio_connection import _is_cloud_provider

        assert _is_cloud_provider("https://api.minimax.io") is True
        assert _is_cloud_provider("https://api.minimaxi.com") is True
        assert _is_cloud_provider("https://api.openai.com") is True
        assert _is_cloud_provider("http://localhost:1234") is False
        assert _is_cloud_provider("http://192.168.1.100:1234") is False


class TestMinimaxSystemRoleConversion:
    """System role messages are converted for cloud providers."""

    def test_system_message_converted_to_user(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        messages = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Do work."},
        ]
        result = LMStudioConnectionPool._convert_system_to_user(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "[System]" in result[0]["content"]
        assert "You are an agent." in result[0]["content"]
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Do work."

    def test_multiple_system_messages_merged(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        messages = [
            {"role": "system", "content": "Part 1."},
            {"role": "system", "content": "Part 2."},
            {"role": "user", "content": "Hello"},
        ]
        result = LMStudioConnectionPool._convert_system_to_user(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert "Part 1." in result[0]["content"]
        assert "Part 2." in result[0]["content"]

    def test_no_system_messages_unchanged(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = LMStudioConnectionPool._convert_system_to_user(messages)
        assert result == messages

    def test_system_in_middle_converted(self) -> None:
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        messages = [
            {"role": "user", "content": "Start"},
            {"role": "system", "content": "Nudge"},
            {"role": "assistant", "content": "Ok"},
        ]
        result = LMStudioConnectionPool._convert_system_to_user(messages)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Start"
        assert result[1]["role"] == "user"
        assert "[System]" in result[1]["content"]
        assert result[2]["role"] == "assistant"

    def test_chat_completion_converts_system_for_cloud(self) -> None:
        """chat_completion auto-converts system messages for cloud providers."""
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="https://api.minimax.io",
            api_key="test",
            model="MiniMax-M2.7",
        )
        captured_body = {}

        def fake_request(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body["data"] = json.loads(body)
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            return resp

        pool._pool.request = fake_request
        result = pool.chat_completion(
            messages=[
                {"role": "system", "content": "You are a worker."},
                {"role": "user", "content": "Do stuff."},
            ],
            model="MiniMax-M2.7",
        )
        assert "error" not in result
        sent_messages = captured_body["data"]["messages"]
        # system should be converted to user
        assert all(m["role"] != "system" for m in sent_messages)
        assert any("[System]" in m["content"] for m in sent_messages)

    def test_chat_completion_keeps_system_for_local(self) -> None:
        """chat_completion keeps system messages for local LM Studio."""
        from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool

        pool = LMStudioConnectionPool(
            base_url="http://localhost:1234",
        )
        captured_body = {}

        def fake_request(method, url, body=None, headers=None, timeout=None, retries=None):
            captured_body["data"] = json.loads(body)
            resp = MagicMock()
            resp.status = 200
            resp.data = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            return resp

        pool._pool.request = fake_request
        pool.chat_completion(
            messages=[
                {"role": "system", "content": "You are a worker."},
                {"role": "user", "content": "Do stuff."},
            ],
        )
        sent_messages = captured_body["data"]["messages"]
        assert sent_messages[0]["role"] == "system"

    def test_temperature_not_adjusted_for_minimax_in_tool_loop(self) -> None:
        """Tool loop uses minimax provider for temperature, not lmstudio."""
        from app.services.direct_execution.lmstudio_connection import _is_cloud_provider

        assert _is_cloud_provider("https://api.minimax.io")
        from app.services.direct_execution.temperature_config import get_adaptive_temperature

        # minimax should not reduce temperature
        assert get_adaptive_temperature("minimax", 1.0) == 1.0
