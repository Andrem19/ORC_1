"""Tests for LM Studio translation backend and CliTranslator."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.config import NotificationConfig, load_config_from_dict
from app.services.translation_service import CliTranslator, LMStudioTranslator, TranslationService


# ---------------------------------------------------------------------------
# Tests: config
# ---------------------------------------------------------------------------


class TestTranslationProviderConfig:
    def test_provider_defaults_to_lmstudio(self):
        cfg = NotificationConfig()
        assert cfg.translation_provider == "lmstudio"

    def test_lmstudio_provider_from_dict(self):
        data = {
            "notifications": {
                "enabled": True,
                "translate_to_russian": True,
                "translation_provider": "lmstudio",
                "translation_fallback_1": "qwen_cli",
            },
            "lmstudio": {
                "base_url": "http://192.168.1.100:1234",
                "model": "qwen/qwen3.5-9b",
            },
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translation_provider == "lmstudio"
        assert cfg.notifications.translation_fallback_1 == "qwen_cli"
        assert cfg.lmstudio.base_url == "http://192.168.1.100:1234"
        assert cfg.lmstudio.model == "qwen/qwen3.5-9b"


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
        mock_resp.read.return_value = json.dumps({"data": [{"id": "qwen/qwen3.5-9b"}]}).encode()

        with patch(
            "app.lmstudio_api.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            assert t.check_available() is True
        assert t.is_available is True

    def test_unavailable_on_500(self):
        t = LMStudioTranslator(base_url="http://localhost:1234")
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.read.return_value = b'{"error":"down"}'

        with patch(
            "app.lmstudio_api.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            assert t.check_available() is False

    def test_unavailable_on_connection_error(self):
        t = LMStudioTranslator(base_url="http://localhost:1234")

        with patch(
            "app.lmstudio_api.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError()
            assert t.check_available() is False

    def test_unavailable_when_model_missing(self):
        t = LMStudioTranslator(base_url="http://localhost:1234", model="missing-model")
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({
            "data": [{"id": "qwen/qwen3.5-9b"}],
        }).encode()

        with patch("app.lmstudio_api.HTTPConnection") as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            assert t.check_available() is False


# ---------------------------------------------------------------------------
# Tests: CliTranslator
# ---------------------------------------------------------------------------


class TestCliTranslator:
    def test_check_available_with_mock(self):
        with patch("shutil.which", return_value="/usr/bin/qwen"):
            t = CliTranslator(cli_path="qwen")
            assert t.check_available() is True

    def test_check_available_missing(self):
        with patch("shutil.which", return_value=None):
            t = CliTranslator(cli_path="nonexistent-cli-xyz")
            # Force re-resolve by not caching
            t._resolved = None
            assert t.check_available() is False

    def test_translate_empty_returns_same(self):
        t = CliTranslator(cli_path="qwen")
        assert t.translate("") == ""
        assert t.translate("   ") == "   "

    def test_translate_subprocess_success(self):
        t = CliTranslator(cli_path="qwen", timeout=10)
        t._resolved = "/usr/bin/qwen"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="Привет мир", returncode=0)
            result = t.translate("Hello world")
        assert result == "Привет мир"

    def test_translate_subprocess_timeout_returns_original(self):
        import subprocess
        t = CliTranslator(cli_path="qwen", timeout=1)
        t._resolved = "/usr/bin/qwen"
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("qwen", 1)):
            result = t.translate("Hello world")
        assert result == "Hello world"

    def test_translate_no_resolved_returns_original(self):
        t = CliTranslator(cli_path="nonexistent-cli-xyz")
        t._resolved = None
        with patch("shutil.which", return_value=None):
            result = t.translate("Hello world")
        assert result == "Hello world"


# ---------------------------------------------------------------------------
# Tests: TranslationService with providers
# ---------------------------------------------------------------------------


class TestTranslationServiceProviders:
    def test_creates_lmstudio_translator(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
            lmstudio_base_url="http://localhost:9999",
            lmstudio_model="qwen3.5-9b",
        )
        assert "lmstudio" in svc._translators

    def test_creates_cli_translators_for_fallbacks(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
            fallback_1="qwen_cli",
            fallback_2="claude_cli",
        )
        assert "qwen_cli" in svc._translators
        assert "claude_cli" in svc._translators

    def test_no_translators_when_disabled(self):
        svc = TranslationService(translate=False)
        assert len(svc._translators) == 0

    def test_load_model_lmstudio_checks_availability(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
        )
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"data": []}).encode()

        with patch(
            "app.lmstudio_api.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn = mock_conn_cls.return_value
            mock_conn.getresponse.return_value = mock_resp
            svc.load_model()

        assert svc._model_loaded is True
        assert svc.is_ready is True

    def test_load_model_lmstudio_unavailable_raises(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
        )

        with patch(
            "app.lmstudio_api.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError()
            with pytest.raises(RuntimeError, match="not available"):
                svc.load_model()

    def test_translate_uses_lmstudio(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
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

    def test_translate_preserves_tech_terms(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
        )
        svc._model_loaded = True

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

    def test_translate_fallback_on_error(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
        )
        svc._model_loaded = True

        with patch(
            "app.services.translation_service.HTTPConnection"
        ) as mock_conn_cls:
            mock_conn_cls.return_value.request.side_effect = ConnectionError("fail")
            result = svc.translate("Hello world")

        assert result == "Hello world"

    def test_is_ready_after_load(self):
        svc = TranslationService(
            translate=True,
            provider="lmstudio",
        )
        assert svc.is_ready is False
        svc._model_loaded = True
        assert svc.is_ready is True
