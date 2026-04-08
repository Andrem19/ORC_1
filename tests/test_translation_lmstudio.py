"""Tests for LM Studio translation backend."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from app.config import NotificationConfig, load_config_from_dict
from app.services.translation_service import LMStudioTranslator, TranslationService


# ---------------------------------------------------------------------------
# Tests: config
# ---------------------------------------------------------------------------


class TestLMStudioConfig:
    def test_backend_defaults_to_opus(self):
        cfg = NotificationConfig()
        assert cfg.translation_backend == "opus"

    def test_lmstudio_translation_via_shared_config(self):
        """LM Studio translation params come from [lmstudio.translation] sub-config."""
        data = {
            "notifications": {
                "enabled": True,
                "translate_to_russian": True,
                "translation_backend": "lmstudio",
            },
            "lmstudio": {
                "enabled": True,
                "base_url": "http://192.168.1.100:1234",
                "model": "qwen/qwen3.5-9b",
                "translation": {
                    "max_tokens": 2048,
                    "timeout_seconds": 60,
                },
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translation_backend == "lmstudio"
        assert cfg.lmstudio.base_url == "http://192.168.1.100:1234"
        assert cfg.lmstudio.model == "qwen/qwen3.5-9b"
        assert cfg.lmstudio.translation.max_tokens == 2048
        assert cfg.lmstudio.translation.timeout_seconds == 60

    def test_from_dict_opus_backend(self):
        data = {
            "notifications": {
                "enabled": True,
                "translate_to_russian": True,
                "translation_backend": "opus",
            }
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translation_backend == "opus"


# ---------------------------------------------------------------------------
# Tests: LMStudioTranslator
# ---------------------------------------------------------------------------


class TestLMStudioTranslator:
    def _make_translator(self, **kwargs) -> LMStudioTranslator:
        defaults = dict(
            base_url="http://localhost:1234",
            model="",
            max_tokens=1024,
            timeout_seconds=30,
        )
        defaults.update(kwargs)
        return LMStudioTranslator(**defaults)

    def test_empty_text_returns_same(self):
        t = self._make_translator()
        assert t.translate("") == ""
        assert t.translate("   ") == "   "

    def test_cache_hit(self):
        t = self._make_translator()
        t._cache["hello"] = "привет"
        assert t.translate("hello") == "привет"

    def test_translate_success(self):
        t = self._make_translator()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Привет, мир!"}}],
        }).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            result = t.translate("Hello, world!")

        assert result == "Привет, мир!"
        assert "Hello, world!" in t._cache

    def test_translate_with_model_name(self):
        t = self._make_translator(model="qwen3.5-9b")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Тест"}}],
        }).encode()

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            t.translate("Test")

        # Verify model was included in request body
        call_args = mock_conn.request.call_args
        body = json.loads(call_args[0][2])
        assert body["model"] == "qwen3.5-9b"

    def test_translate_http_error_returns_original(self):
        t = self._make_translator()
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.read.return_value = b"Internal Server Error"

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            result = t.translate("Hello")

        assert result == "Hello"

    def test_translate_connection_error_returns_original(self):
        t = self._make_translator()

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError("refused")
            result = t.translate("Hello")

        assert result == "Hello"

    def test_translate_empty_response_returns_original(self):
        t = self._make_translator()
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "  "}}],
        }).encode()

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            result = t.translate("Hello")

        assert result == "Hello"

    def test_cache_eviction(self):
        t = self._make_translator()
        for i in range(205):
            t._cache[f"key_{i}"] = f"val_{i}"
            if len(t._cache) > 200:
                t._cache.popitem(last=False)
        assert len(t._cache) == 200
        assert "key_0" not in t._cache


# ---------------------------------------------------------------------------
# Tests: check_available
# ---------------------------------------------------------------------------


class TestLMStudioCheckAvailable:
    def test_available_on_200(self):
        t = LMStudioTranslator(base_url="http://localhost:1234")
        mock_resp = MagicMock()
        mock_resp.status = 200

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            assert t.check_available() is True
        assert t.is_available is True

    def test_unavailable_on_500(self):
        t = LMStudioTranslator(base_url="http://localhost:1234")
        mock_resp = MagicMock()
        mock_resp.status = 500

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            assert t.check_available() is False

    def test_unavailable_on_connection_error(self):
        t = LMStudioTranslator(base_url="http://localhost:1234")

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError()
            assert t.check_available() is False


# ---------------------------------------------------------------------------
# Tests: TranslationService with LMStudio backend
# ---------------------------------------------------------------------------


class TestTranslationServiceLMStudio:
    def test_creates_lmstudio_translator(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
            lmstudio_base_url="http://localhost:9999",
            lmstudio_model="qwen3.5-9b",
        )
        assert svc._lmstudio_translator is not None
        assert svc._lmstudio_translator._base_url == "http://localhost:9999"
        assert svc._lmstudio_translator._model == "qwen3.5-9b"

    def test_no_lmstudio_when_disabled(self):
        svc = TranslationService(
            translate_to_russian=False,
            backend="lmstudio",
        )
        assert svc._lmstudio_translator is None

    def test_no_lmstudio_for_opus_backend(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="opus",
        )
        assert svc._lmstudio_translator is None

    def test_load_model_lmstudio_checks_availability(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )
        mock_resp = MagicMock()
        mock_resp.status = 200

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            svc.load_model()

        assert svc._model_loaded is True
        assert svc.is_ready is True

    def test_load_model_lmstudio_unavailable_raises(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError()
            with pytest.raises(RuntimeError, match="not reachable"):
                svc.load_model()

    def test_translate_uses_lmstudio(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )
        svc._model_loaded = True

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Оркестратор запущен"}}],
        }).encode()

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            result = svc.translate("Orchestrator started")

        assert result == "Оркестратор запущен"

    def test_translate_lmstudio_preserves_tech_terms(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )
        svc._model_loaded = True

        # Simulate LM Studio returning text with preserved placeholders
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "choices": [{"message": {"content": "Запуск __TK0__ на __TK1__"}}],
        }).encode()

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            result = svc.translate("Running qwen-1 on BTCUSDT")

        assert "qwen-1" in result
        assert "BTCUSDT" in result

    def test_translate_lmstudio_fallback_on_error(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )
        svc._model_loaded = True

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError("fail")
            result = svc.translate("Hello world")

        assert result == "Hello world"

    def test_is_ready_lmstudio(self):
        svc = TranslationService(
            translate_to_russian=True,
            backend="lmstudio",
        )
        assert svc.is_ready is False
        svc._model_loaded = True
        assert svc.is_ready is True


# ---------------------------------------------------------------------------
# Tests: NotificationService integration with LMStudio config
# ---------------------------------------------------------------------------


class TestNotificationLMStudioIntegration:
    def test_notification_service_passes_lmstudio_config(self):
        from app.services.notification_service import NotificationService
        from app.config import LMStudioConfig, LMStudioTranslationConfig

        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(
                enabled=True,
                translate_to_russian=True,
                translation_backend="lmstudio",
            )
            lm_cfg = LMStudioConfig(
                enabled=True,
                base_url="http://localhost:9999",
                model="qwen/qwen3.5-9b",
            )
            svc = NotificationService(cfg, lmstudio_config=lm_cfg)
            assert svc._translator._backend == "lmstudio"
            assert svc._translator._lmstudio_translator is not None
            assert svc._translator._lmstudio_translator._base_url == "http://localhost:9999"
            assert svc._translator._lmstudio_translator._model == "qwen/qwen3.5-9b"

    def test_notification_service_opus_config(self):
        from app.services.notification_service import NotificationService

        with patch.dict(os.environ, {"ALGO_BOT": "tok", "CHAT_ID": "1"}):
            cfg = NotificationConfig(
                enabled=True,
                translate_to_russian=True,
                translation_backend="opus",
            )
            svc = NotificationService(cfg)
            assert svc._translator._backend == "opus"
            assert svc._translator._lmstudio_translator is None
