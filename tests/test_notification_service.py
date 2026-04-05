"""Tests for the notification service."""

from __future__ import annotations

import json
import os
import time
from unittest.mock import patch

import pytest

from app.config import NotificationConfig, OrchestratorConfig, load_config_from_dict
from app.models import (
    PlannerDecision,
    PlannerOutput,
    TaskResult,
    OrchestratorState,
)
from app.services.notification_service import NotificationService


# ---------------------------------------------------------------------------
# Tests: config
# ---------------------------------------------------------------------------


class TestNotificationConfig:
    def test_defaults(self):
        cfg = NotificationConfig()
        assert cfg.enabled is False
        assert cfg.min_interval_seconds == 30

    def test_from_dict(self):
        data = {"notifications": {"enabled": True, "min_interval_seconds": 60}}
        cfg = load_config_from_dict(data)
        assert cfg.notifications.enabled is True
        assert cfg.notifications.min_interval_seconds == 60

    def test_missing_notifications_key(self):
        cfg = load_config_from_dict({})
        assert cfg.notifications.enabled is False


# ---------------------------------------------------------------------------
# Tests: service basics
# ---------------------------------------------------------------------------


class TestNotificationServiceBasics:
    def test_disabled_by_default(self):
        svc = NotificationService()
        assert svc._enabled is False

    def test_enabled_with_credentials(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok123", "CHAT_ID": "123"}):
            cfg = NotificationConfig(enabled=True)
            svc = NotificationService(cfg)
            assert svc._enabled is True

    def test_disabled_without_token(self):
        with patch.dict(os.environ, {"CHAT_ID": "123"}, clear=False):
            env = {k: v for k, v in os.environ.items() if k != "ALGO_BOT"}
            with patch.dict(os.environ, env, clear=True):
                cfg = NotificationConfig(enabled=True)
                svc = NotificationService(cfg)
                assert svc._enabled is False

    def test_disabled_without_chat_id(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok123"}, clear=False):
            env = {k: v for k, v in os.environ.items() if k != "CHAT_ID"}
            with patch.dict(os.environ, env, clear=True):
                cfg = NotificationConfig(enabled=True)
                svc = NotificationService(cfg)
                assert svc._enabled is False

    def test_no_op_when_disabled(self):
        svc = NotificationService()
        result = svc.send_lifecycle("started")
        assert result is False

    def test_is_configured(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "99"}):
            svc = NotificationService()
            assert svc.is_configured() is True


# ---------------------------------------------------------------------------
# Tests: rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_skips_message(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=100)
            svc = NotificationService(cfg)
            svc._last_send_time = time.monotonic()  # just sent
            result = svc._send("test")
            assert result is False

    def test_message_sent_after_interval(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            svc._last_send_time = 0  # long ago
            with patch.object(svc, "_send", return_value=True) as mock_send:
                svc.send_lifecycle("started", "testing")
                mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: message formatting (via mock server)
# ---------------------------------------------------------------------------


class TestMessageSending:
    def test_send_worker_result_success(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            result = TaskResult(
                task_id="abc123",
                worker_id="qwen-1",
                status="success",
                summary="Created cf_test feature",
                confidence=0.92,
            )
            # Use send method directly with mocked connection
            with patch("app.services.notification_service.HTTPSConnection") as mock_conn:
                resp = type("R", (), {"status": 200, "read": lambda self: b'{"ok":true}'})()
                conn_instance = mock_conn.return_value
                conn_instance.getresponse.return_value = resp
                sent = svc.send_worker_result(result, cycle=5)
                assert sent is True
                # Verify the request was made
                conn_instance.request.assert_called_once()
                call_args = conn_instance.request.call_args
                body = json.loads(call_args[0][2])
                assert "abc123" in body["text"]
                assert "qwen-1" in body["text"]

    def test_send_planner_decision(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            output = PlannerOutput(
                decision=PlannerDecision.LAUNCH_WORKER,
                target_worker_id="qwen-1",
                task_instruction="Run backtest on cf_new_feature",
                reason="Testing new feature impact",
            )
            with patch.object(svc, "_send", return_value=True) as mock:
                svc.send_planner_decision(output, cycle=10)
                text = mock.call_args[0][0]
                assert "launch_worker" in text
                assert "qwen-1" in text

    def test_send_error(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            with patch.object(svc, "_send", return_value=True) as mock:
                svc.send_error("Connection timeout", context="worker")
                text = mock.call_args[0][0]
                assert "Connection timeout" in text
                assert "worker" in text


# ---------------------------------------------------------------------------
# Tests: integration with OrchestratorConfig
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    def test_full_config_with_notifications(self):
        data = {
            "goal": "test",
            "notifications": {
                "enabled": True,
                "min_interval_seconds": 60,
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.enabled is True
        assert cfg.notifications.min_interval_seconds == 60

    def test_config_serialization(self):
        cfg = OrchestratorConfig()
        d = cfg.to_dict()
        assert "notifications" in d
        assert d["notifications"]["enabled"] is False
