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


# ---------------------------------------------------------------------------
# Tests: enriched worker result messages
# ---------------------------------------------------------------------------


def _make_svc_no_batch():
    with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
        cfg = NotificationConfig(enabled=True, min_interval_seconds=0, batch_enabled=False)
        return NotificationService(cfg)


class TestEnrichedWorkerResult:
    def test_title_used_instead_of_task_id(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="plan_abc:slice_xyz",
            worker_id="qwen-1",
            status="success",
            title="Validate OHLCV quality",
            summary="Data quality confirmed.",
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=3)
        sections = mock.call_args[0][0]
        header_text = "".join(s.value for s in sections if s.kind == "header")
        assert "Validate OHLCV quality" in header_text
        assert "slice_xyz" not in header_text

    def test_task_id_used_when_no_title(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="plan_abc:slice_xyz",
            worker_id="qwen-1",
            status="success",
            title="",
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        header_text = "".join(s.value for s in sections if s.kind == "header")
        assert "slice_xyz" in header_text

    def test_verdict_icon_in_header(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="t1", worker_id="w1", status="success",
            verdict="PROMOTE", title="Test slice",
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        header_text = "".join(s.value for s in sections if s.kind == "header")
        assert "\U0001f3c6" in header_text  # 🏆

    def test_findings_appear_as_bullets(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="t1", worker_id="w1", status="success",
            findings=["Strong momentum edge confirmed", "RSI divergence pattern found"],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Strong momentum edge confirmed" in body_text
        assert "RSI divergence pattern found" in body_text
        assert "\u2022" in body_text  # bullet

    def test_key_metrics_in_code_block(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="t1", worker_id="w1", status="success",
            key_metrics={"return": 2.45, "sharpe": 1.3},
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        code_text = " ".join(s.value for s in sections if s.kind == "code")
        assert "return=2.45" in code_text
        assert "sharpe=1.30" in code_text

    def test_next_actions_appear(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="t1", worker_id="w1", status="success",
            next_actions=["Extend backtest to 2020", "Test on ETH/USDT"],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Extend backtest to 2020" in body_text
        assert "\u2192" in body_text  # →

    def test_no_extra_sections_when_fields_empty(self):
        svc = _make_svc_no_batch()
        result = TaskResult(task_id="t1", worker_id="w1", status="success")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=1)
        sections = mock.call_args[0][0]
        # Should not crash and should produce at least header + separator
        assert any(s.kind == "header" for s in sections)
        assert any(s.kind == "separator" for s in sections)

    def test_sequence_label_in_header(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="plan_abc:slice_xyz",
            worker_id="qwen-1",
            status="success",
            title="Validate data",
            sequence_label="v4 B2",
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=5)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "v4 B2" in header_text
        assert "Cycle #5" in header_text
        assert "Validate data" in header_text

    def test_sequence_label_absent_when_empty(self):
        svc = _make_svc_no_batch()
        result = TaskResult(
            task_id="t1", worker_id="w1", status="success", title="Test",
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_worker_result(result, cycle=3)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "Cycle #3" in header_text
        assert "Test" in header_text
        # No extra pipe separator when no sequence label
        assert header_text.count("|") == 1  # Cycle | Test


# ---------------------------------------------------------------------------
# Tests: enriched batch summary with verdict distribution
# ---------------------------------------------------------------------------


class TestEnrichedBatchSummary:
    def test_sequence_label_in_batch_header(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success",
                        title="Slice A", sequence_label="v4 B2")
        r2 = TaskResult(task_id="t2", worker_id="w2", status="success",
                        title="Slice B", sequence_label="v4 B2")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc._send_batch_summary([
                _QueuedNotification(r1, 5, 0.0),
                _QueuedNotification(r2, 5, 0.0),
            ])
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "v4 B2" in header_text
        assert "Cycle #5" in header_text

    def test_verdict_distribution_line_present(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success",
                        verdict="PROMOTE", title="Slice A")
        r2 = TaskResult(task_id="t2", worker_id="w2", status="success",
                        verdict="WATCHLIST", title="Slice B")
        r3 = TaskResult(task_id="t3", worker_id="w3", status="error",
                        verdict="FAILED", title="Slice C")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc._send_batch_summary([
                _QueuedNotification(r1, 5, 0.0),
                _QueuedNotification(r2, 5, 0.0),
                _QueuedNotification(r3, 5, 0.0),
            ])
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "1 promoted" in body_text or "\U0001f3c6 1" in body_text
        assert "watchlist" in body_text.lower() or "\U0001f441 1" in body_text

    def test_per_worker_line_uses_title(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="plan:slice_a", worker_id="w1", status="success",
                        title="OHLCV validation", verdict="PROMOTE",
                        summary="All good")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc._send_batch_summary([_QueuedNotification(r1, 3, 0.0)])
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "OHLCV validation" in body_text

    def test_verdict_appended_to_line(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success",
                        verdict="WATCHLIST", summary="Weak signal")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc._send_batch_summary([_QueuedNotification(r1, 1, 0.0)])
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "WATCHLIST" in body_text

    def test_no_verdict_line_when_no_verdicts_set(self):
        svc = _make_batch_svc()
        r1 = TaskResult(task_id="t1", worker_id="w1", status="success", verdict="")
        r2 = TaskResult(task_id="t2", worker_id="w2", status="error", verdict="")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc._send_batch_summary([
                _QueuedNotification(r1, 1, 0.0),
                _QueuedNotification(r2, 1, 0.0),
            ])
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        # No verdict-specific words
        assert "promoted" not in body_text


# ---------------------------------------------------------------------------
# Tests: send_run_complete
# ---------------------------------------------------------------------------


from app.services.notification_service import _ms_to_human, _short_task_label, _verdict_icon, _stop_icon, _QueuedNotification


class TestMsToHuman:
    def test_seconds_only(self):
        assert _ms_to_human(45_000) == "45s"

    def test_minutes_and_seconds(self):
        assert _ms_to_human(272_000) == "4m 32s"

    def test_hours(self):
        assert _ms_to_human(3_661_000) == "1h 01m"

    def test_zero(self):
        assert _ms_to_human(0) == "—"


class TestShortTaskLabel:
    def test_title_preferred(self):
        r = TaskResult(task_id="plan:slice", worker_id="w", status="success", title="My Task")
        assert _short_task_label(r) == "My Task"

    def test_slice_id_fallback(self):
        r = TaskResult(task_id="plan_abc:slice_xyz", worker_id="w", status="success")
        assert _short_task_label(r) == "slice_xyz"

    def test_full_task_id_when_no_colon(self):
        r = TaskResult(task_id="standalone", worker_id="w", status="success")
        assert _short_task_label(r) == "standalone"


class TestVerdictIcon:
    def test_promote(self):
        assert _verdict_icon("PROMOTE") == "\U0001f3c6"

    def test_watchlist(self):
        assert _verdict_icon("WATCHLIST") == "\U0001f441"

    def test_failed(self):
        assert _verdict_icon("FAILED") == "\u274c"

    def test_empty(self):
        assert _verdict_icon("") == "\u2139\ufe0f"


class TestSendRunComplete:
    def _make_report(self, **overrides):
        from types import SimpleNamespace
        from app.reporting.models import DirectExecutionMetrics, NarrativeSectionsRu
        narrative = NarrativeSectionsRu(
            executive_summary_ru="Исследование завершено успешно.",
            recommended_next_actions_ru=["Расширить бэктест", "Протестировать на ETH"],
        )
        defaults = dict(
            stop_reason="goal_reached",
            goal="Research crypto momentum strategy",
            duration_ms=272_000,
            completed_sequences=3,
            failed_sequences=0,
            partial_sequences=1,
            best_outcomes=["btc_momentum_v2 — strong edge confirmed", "eth_rsi_cross — moderate signal"],
            unresolved_blockers=["MCP dataset sync timeout"],
            executive_summary_ru="Исследование завершено успешно.",
            narrative_sections_ru=narrative,
            direct_metrics=DirectExecutionMetrics(direct_tool_calls_observed=142, direct_failed=3),
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_enabled_svc(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            return NotificationService(cfg)

    def test_run_complete_disabled_returns_false(self):
        svc = NotificationService()
        assert svc.send_run_complete(object()) is False

    def test_run_complete_includes_stop_reason(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "goal_reached" in header_text

    def test_run_complete_includes_goal(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        all_text = " ".join(s.value for s in sections)
        assert "Research crypto momentum strategy" in all_text

    def test_run_complete_includes_duration(self):
        svc = self._make_enabled_svc()
        report = self._make_report(duration_ms=272_000)
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        all_text = " ".join(s.value for s in sections)
        assert "4m 32s" in all_text

    def test_run_complete_includes_executive_summary(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Исследование завершено успешно" in body_text

    def test_run_complete_includes_best_outcomes(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "btc_momentum_v2" in body_text

    def test_run_complete_includes_blockers(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "MCP dataset sync timeout" in body_text

    def test_run_complete_includes_tool_usage(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        all_text = " ".join(s.value for s in sections)
        assert "142" in all_text

    def test_run_complete_includes_next_actions(self):
        svc = self._make_enabled_svc()
        report = self._make_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Расширить бэктест" in body_text

    def test_run_complete_stop_reason_icon_goal_reached(self):
        assert _stop_icon("goal_reached") == "\u2705"

    def test_run_complete_stop_reason_icon_max_errors(self):
        assert _stop_icon("max_errors") == "\U0001f525"

    def test_run_complete_skips_sections_when_empty(self):
        svc = self._make_enabled_svc()
        report = self._make_report(
            best_outcomes=[],
            unresolved_blockers=[],
            executive_summary_ru="",
        )
        # Override narrative to have no next actions
        from types import SimpleNamespace
        from app.reporting.models import NarrativeSectionsRu
        report.narrative_sections_ru = NarrativeSectionsRu()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_run_complete(report)
        sections = mock.call_args[0][0]
        header_values = [s.value for s in sections if s.kind == "header"]
        # Should only have "Run complete —..." header, not Best outcomes etc.
        assert not any("Best outcomes" in h for h in header_values)
        assert not any("Blockers" in h for h in header_values)


# ---------------------------------------------------------------------------
# Tests: send_sequence_complete
# ---------------------------------------------------------------------------


class TestSendSequenceComplete:
    def _make_seq_report(self, **overrides):
        from types import SimpleNamespace
        from app.reporting.models import NarrativeSectionsRu
        narrative = NarrativeSectionsRu()
        defaults = dict(
            sequence_id="compiled_plan_v1",
            sequence_status="completed",
            duration_ms=600_000,
            batch_results=[
                {"plan_id": "compiled_plan_v1_batch_1", "status": "completed", "final_verdict": "PROMOTE"},
                {"plan_id": "compiled_plan_v1_batch_2", "status": "completed", "final_verdict": "WATCHLIST"},
            ],
            slice_verdict_rollup={"PROMOTE": 3, "WATCHLIST": 2, "REJECT": 1},
            confirmed_findings=["Strong momentum edge", "RSI divergence confirmed"],
            failed_branches=[],
            blockers=[],
            executive_summary_ru="Sequence plan_v1 завершена успешно. 4 batch-планов выполнено.",
            recommended_next_actions=["Расширить бэктест", "Протестировать ETH"],
            compiled_plan_count=4,
            narrative_sections_ru=narrative,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _make_enabled_svc(self):
        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(enabled=True, min_interval_seconds=0)
            return NotificationService(cfg)

    def test_sequence_complete_disabled_returns_false(self):
        svc = NotificationService()
        assert svc.send_sequence_complete(object()) is False

    def test_sequence_complete_includes_sequence_id(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "compiled_plan_v1" in header_text

    def test_sequence_complete_includes_status(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "completed" in header_text

    def test_sequence_complete_includes_duration(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        all_text = " ".join(s.value for s in sections)
        assert "10m" in all_text

    def test_sequence_complete_includes_batch_results(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        all_text = " ".join(s.value for s in sections)
        assert "compiled_plan_v1_batch_1" in all_text
        assert "compiled_plan_v1_batch_2" in all_text

    def test_sequence_complete_includes_verdict_rollup(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "PROMOTE: 3" in body_text
        assert "WATCHLIST: 2" in body_text

    def test_sequence_complete_includes_findings(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Strong momentum edge" in body_text

    def test_sequence_complete_includes_summary(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "завершена успешно" in body_text

    def test_sequence_complete_includes_next_actions(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report()
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Расширить бэктест" in body_text

    def test_sequence_complete_includes_blockers(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report(blockers=["MCP timeout", "Missing feature"])
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "MCP timeout" in body_text

    def test_sequence_complete_includes_failed_branches(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report(
            failed_branches=["stage_5: confidence too low (minimax, tools=2)"],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "stage_5" in body_text

    def test_sequence_complete_failed_status_icon(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report(sequence_status="failed")
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        header_text = " ".join(s.value for s in sections if s.kind == "header")
        assert "failed" in header_text

    def test_sequence_complete_skips_empty_sections(self):
        svc = self._make_enabled_svc()
        report = self._make_seq_report(
            confirmed_findings=[],
            failed_branches=[],
            blockers=[],
            executive_summary_ru="",
            recommended_next_actions=[],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        header_values = [s.value for s in sections if s.kind == "header"]
        assert not any("Findings" in h for h in header_values)
        assert not any("Failed" in h for h in header_values)
        assert not any("Blockers" in h for h in header_values)

    def test_sequence_complete_renders_narrative_key_findings(self):
        from app.reporting.models import NarrativeSectionsRu
        narrative = NarrativeSectionsRu(
            executive_summary_ru="Последовательность выполнена.",
            key_findings_ru=["Feature cf_volatility улучшил Sharpe до 1.2"],
            important_failures_ru=[],
            recommended_next_actions_ru=["Протестировать на walk-forward"],
            operator_notes_ru=["Рекомендация: проверить стационарность"],
        )
        svc = self._make_enabled_svc()
        report = self._make_seq_report(narrative_sections_ru=narrative)
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "cf_volatility" in body_text

    def test_sequence_complete_renders_narrative_next_actions_over_deterministic(self):
        from app.reporting.models import NarrativeSectionsRu
        narrative = NarrativeSectionsRu(
            executive_summary_ru="Тестирование завершено успешно.",
            key_findings_ru=[],
            important_failures_ru=[],
            recommended_next_actions_ru=["LLM-действие: запустить walk-forward"],
            operator_notes_ru=[],
        )
        svc = self._make_enabled_svc()
        report = self._make_seq_report(
            narrative_sections_ru=narrative,
            recommended_next_actions=["Детерминированное действие"],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "LLM-действие" in body_text
        assert "Детерминированное" not in body_text

    def test_sequence_complete_renders_narrative_notes(self):
        from app.reporting.models import NarrativeSectionsRu
        narrative = NarrativeSectionsRu(
            executive_summary_ru="Выполнено.",
            key_findings_ru=[],
            important_failures_ru=[],
            recommended_next_actions_ru=[],
            operator_notes_ru=["Проверить стационарность feature cf_alpha"],
        )
        svc = self._make_enabled_svc()
        report = self._make_seq_report(narrative_sections_ru=narrative)
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "Проверить стационарность" in body_text

    def test_sequence_complete_renders_narrative_failures(self):
        from app.reporting.models import NarrativeSectionsRu
        narrative = NarrativeSectionsRu(
            executive_summary_ru="Частичный успех.",
            key_findings_ru=[],
            important_failures_ru=["Batch 3 упал из-за таймаута LMStudio"],
            recommended_next_actions_ru=[],
            operator_notes_ru=[],
        )
        svc = self._make_enabled_svc()
        report = self._make_seq_report(
            narrative_sections_ru=narrative,
            failed_branches=[],
        )
        with patch.object(svc, "_send_structured", return_value=True) as mock:
            svc.send_sequence_complete(report)
        sections = mock.call_args[0][0]
        body_text = " ".join(s.value for s in sections if s.kind == "body")
        assert "таймаута LMStudio" in body_text
