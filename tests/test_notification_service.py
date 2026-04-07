"""Tests for the notification service."""

from __future__ import annotations

import json
import os
import re
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
            result = svc._try_send("test", parse_mode=None)
            assert result is False

    def test_message_sent_after_interval(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            svc._last_send_time = 0  # long ago
            with patch.object(svc, "_try_send", return_value=True) as mock_send:
                svc.send_lifecycle("started", "testing")
                mock_send.assert_called()


# ---------------------------------------------------------------------------
# Tests: HTML formatting in sent messages
# ---------------------------------------------------------------------------


class TestHtmlFormatting:
    def _make_svc(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0, batch_enabled=False)
            return NotificationService(cfg)

    def _mock_conn(self):
        """Return a mocked HTTPSConnection that returns 200."""
        resp = type("R", (), {"status": 200, "read": lambda self: b'{"ok":true}'})()
        mock_conn_cls = patch(
            "app.services.notification_service.HTTPSConnection"
        )
        return mock_conn_cls, resp

    def test_send_worker_result_has_html_parse_mode(self):
        svc = self._make_svc()
        result = TaskResult(
            task_id="abc123",
            worker_id="qwen-1",
            status="success",
            summary="Created cf_test feature",
            confidence=0.92,
        )
        with self._mock_conn()[0] as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            conn_instance.getresponse.return_value = type(
                "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
            )()
            sent = svc.send_worker_result(result, cycle=5)
            assert sent is True
            call_args = conn_instance.request.call_args
            body = json.loads(call_args[0][2])
            assert body.get("parse_mode") == "HTML"
            assert "<b>" in body["text"]
            assert "abc123" in body["text"]

    def test_send_worker_result_failure(self):
        svc = self._make_svc()
        result = TaskResult(
            task_id="fail-1",
            worker_id="qwen-2",
            status="error",
            error="Connection timeout",
        )
        with self._mock_conn()[0] as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            conn_instance.getresponse.return_value = type(
                "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
            )()
            sent = svc.send_worker_result(result, cycle=3)
            assert sent is True
            body = json.loads(conn_instance.request.call_args[0][2])
            assert "fail-1" in body["text"]
            assert "Connection timeout" in body["text"]

    def test_send_error_has_html(self):
        svc = self._make_svc()
        with self._mock_conn()[0] as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            conn_instance.getresponse.return_value = type(
                "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
            )()
            sent = svc.send_error("Something went wrong", context="worker")
            assert sent is True
            body = json.loads(conn_instance.request.call_args[0][2])
            assert body["parse_mode"] == "HTML"
            assert "Something went wrong" in body["text"]

    def test_send_lifecycle_html(self):
        svc = self._make_svc()
        with self._mock_conn()[0] as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            conn_instance.getresponse.return_value = type(
                "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
            )()
            sent = svc.send_lifecycle("started", "Goal: test something")
            assert sent is True
            body = json.loads(conn_instance.request.call_args[0][2])
            assert body["parse_mode"] == "HTML"
            assert "started" in body["text"]

    def test_special_chars_escaped_in_html(self):
        svc = self._make_svc()
        result = TaskResult(
            task_id="x<y&z>",
            worker_id="qwen-1",
            status="error",
            error="a < b & c > d",
        )
        with self._mock_conn()[0] as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            conn_instance.getresponse.return_value = type(
                "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
            )()
            sent = svc.send_worker_result(result, cycle=1)
            assert sent is True
            body = json.loads(conn_instance.request.call_args[0][2])
            html = body["text"]
            # HTML entities should be present
            assert "&lt;" in html
            assert "&amp;" in html
            assert "&gt;" in html

    def test_fallback_to_plain_text_on_html_error(self):
        svc = self._make_svc()
        result = TaskResult(
            task_id="abc",
            worker_id="qwen-1",
            status="success",
        )
        with patch(
            "app.services.notification_service.HTTPSConnection"
        ) as mock_conn_cls:
            conn_instance = mock_conn_cls.return_value
            # First call (HTML) fails, second call (plain) succeeds
            conn_instance.getresponse.side_effect = [
                type("R", (), {
                    "status": 400,
                    "read": lambda self: b'{"description":"Bad Request: can\'t parse entities"}',
                })(),
                type("R", (), {"status": 200, "read": lambda self: b'{"ok":true}'})(),
            ]
            sent = svc.send_worker_result(result, cycle=1)
            assert sent is True
            # Two calls: first with HTML, second without
            assert conn_instance.request.call_count == 2
            # Second call should not have parse_mode
            second_body = json.loads(conn_instance.request.call_args_list[1][0][2])
            assert "parse_mode" not in second_body


# ---------------------------------------------------------------------------
# Tests: config integration
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


# ---------------------------------------------------------------------------
# Tests: translation integration
# ---------------------------------------------------------------------------


class TestTranslationIntegration:
    def test_translator_created_in_init(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True)
            svc = NotificationService(cfg)
            assert svc._translator is not None

    def test_translate_to_russian_passed_to_translator(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, translate_to_russian=True)
            svc = NotificationService(cfg)
            assert svc._translator.is_enabled is True

    def test_translate_disabled_by_default(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True)
            svc = NotificationService(cfg)
            assert svc._translator.is_enabled is False

    def test_send_calls_translate(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            with patch.object(
                svc._translator, "translate", return_value="translated"
            ) as mock_tr:
                with patch(
                    "app.services.notification_service.HTTPSConnection"
                ) as mock_conn:
                    conn_instance = mock_conn.return_value
                    conn_instance.getresponse.return_value = type(
                        "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
                    )()
                    svc.send_lifecycle("started")
                    mock_tr.assert_called_once()

    def test_init_translation_delegates(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, translate_to_russian=True)
            svc = NotificationService(cfg)
            with patch.object(svc._translator, "load_model") as mock_load:
                svc.init_translation()
                mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: batch notifications
# ---------------------------------------------------------------------------


def _make_batch_svc(**overrides):
    """Create an enabled NotificationService with batch config."""
    with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
        defaults = dict(enabled=True, min_interval_seconds=0, batch_enabled=True)
        defaults.update(overrides)
        cfg = NotificationConfig(**defaults)
        return NotificationService(cfg)


def _mock_telegram_ok():
    """Patch HTTPSConnection to return 200."""
    return patch(
        "app.services.notification_service.HTTPSConnection",
        return_value=type(
            "Conn", (),
            {
                "getresponse": lambda self: type(
                    "R", (), {"status": 200, "read": lambda self: b'{"ok":true}'}
                )(),
                "request": lambda self, *a, **kw: None,
                "close": lambda self: None,
            },
        )(),
    )


class TestBatchNotifications:
    def test_batch_queues_worker_result(self):
        svc = _make_batch_svc()
        result = TaskResult(task_id="t1", worker_id="w1", status="success", summary="ok")
        with _mock_telegram_ok():
            sent = svc.send_worker_result(result, cycle=1)
        assert sent is True
        assert len(svc._batch_queue) == 1
        assert svc._batch_queue[0].result.task_id == "t1"

    def test_flush_sends_batch_summary(self):
        svc = _make_batch_svc(batch_debounce_seconds=0)
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success", summary="feature A")
        r2 = TaskResult(task_id="t2", worker_id="w2", status="error", error="timeout")
        r3 = TaskResult(task_id="t3", worker_id="w3", status="success", summary="feature B")
        with _mock_telegram_ok():
            svc.send_worker_result(r1, cycle=5)
            svc.send_worker_result(r2, cycle=5)
            svc.send_worker_result(r3, cycle=5)
            svc._flush_batch()
        # Queue should be empty after flush
        assert len(svc._batch_queue) == 0

    def test_single_item_sends_individually(self):
        svc = _make_batch_svc()
        result = TaskResult(task_id="t1", worker_id="w1", status="success")
        with _mock_telegram_ok():
            svc._queue_for_batch(result, cycle=1)
            svc._flush_batch()
        assert len(svc._batch_queue) == 0

    def test_urgent_notifications_not_batched(self):
        svc = _make_batch_svc()
        with _mock_telegram_ok() as mock:
            svc.send_error("critical error", context="worker")
            svc.send_lifecycle("finished", "done")
        # These go directly through _send_structured, not batch queue
        assert len(svc._batch_queue) == 0

    def test_batch_disabled_sends_immediately(self):
        svc = _make_batch_svc(batch_enabled=False)
        result = TaskResult(task_id="t1", worker_id="w1", status="success")
        with _mock_telegram_ok():
            sent = svc.send_worker_result(result, cycle=1)
        assert sent is True
        assert len(svc._batch_queue) == 0

    def test_flush_on_shutdown(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success")
        svc._queue_for_batch(r1, cycle=1)
        assert len(svc._batch_queue) == 1
        with _mock_telegram_ok():
            svc.flush()
        assert len(svc._batch_queue) == 0

    def test_config_batch_defaults(self):
        cfg = NotificationConfig()
        assert cfg.batch_enabled is True
        assert cfg.batch_debounce_seconds == 5.0


class TestDiagnosticDigest:
    def test_send_diagnostic_digest(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            with patch.object(svc, "_send_structured", return_value=True) as mock:
                result = svc.send_diagnostic_digest("Errors found: MCP timeout", cycle=50)
            assert result is True
            sections = mock.call_args[0][0]
            body_text = "".join(s.value for s in sections if s.kind == "body")
            assert "MCP timeout" in body_text

    def test_digest_disabled_returns_false(self):
        cfg = NotificationConfig(enabled=False)
        svc = NotificationService(cfg)
        assert svc.send_diagnostic_digest("test", cycle=1) is False


class TestExecutionPrediction:
    def test_send_execution_prediction(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            svc = NotificationService(cfg)
            with patch.object(svc, "_send_structured", return_value=True) as mock:
                result = svc.send_execution_prediction(
                    "Stage 1: ~8 min, Stage 2: ~15 min", plan_version=3,
                )
            assert result is True
            sections = mock.call_args[0][0]
            body_text = "".join(s.value for s in sections if s.kind == "body")
            assert "Stage 1" in body_text

    def test_prediction_disabled_returns_false(self):
        cfg = NotificationConfig(enabled=False)
        svc = NotificationService(cfg)
        assert svc.send_execution_prediction("test", plan_version=1) is False
