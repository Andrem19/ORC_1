"""Tests for the translation service."""

from __future__ import annotations

import pytest

from app.config import NotificationConfig, load_config_from_dict
from app.services.translation_service import TranslationService


# ---------------------------------------------------------------------------
# Tests: config integration
# ---------------------------------------------------------------------------


class TestTranslationConfig:
    def test_translate_defaults_to_false(self):
        cfg = NotificationConfig()
        assert cfg.translate_to_russian is False

    def test_provider_default(self):
        cfg = NotificationConfig()
        assert cfg.translation_provider == "lmstudio"

    def test_fallback_defaults_empty(self):
        cfg = NotificationConfig()
        assert cfg.translation_fallback_1 == ""
        assert cfg.translation_fallback_2 == ""

    def test_from_dict(self):
        data = {
            "notifications": {
                "enabled": True,
                "translate_to_russian": True,
                "translation_provider": "lmstudio",
                "translation_fallback_1": "qwen_cli",
                "translation_fallback_2": "claude_cli",
            }
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translate_to_russian is True
        assert cfg.notifications.translation_provider == "lmstudio"
        assert cfg.notifications.translation_fallback_1 == "qwen_cli"
        assert cfg.notifications.translation_fallback_2 == "claude_cli"

    def test_missing_translate_key_defaults_false(self):
        data = {"notifications": {"enabled": True}}
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translate_to_russian is False

    def test_provider_off_disables(self):
        cfg = NotificationConfig(translation_provider="off")
        svc = TranslationService(translate=True, provider="off")
        assert svc.is_enabled is True
        # load_model should not raise — just logs and returns
        svc.load_model()
        assert svc.is_ready is False


# ---------------------------------------------------------------------------
# Tests: term protection
# ---------------------------------------------------------------------------


class TestTermProtection:
    def _make_service(self) -> TranslationService:
        return TranslationService(translate=True)

    def test_feature_names_protected(self):
        svc = self._make_service()
        text = "Created cf_alpha_1 feature and cl_1h classifier"
        protected, placeholders = svc._protect_terms(text)
        assert "cf_alpha_1" not in protected
        assert "cl_1h" not in protected
        assert len(placeholders) == 2
        restored = svc._restore_terms(protected, placeholders)
        assert restored == text

    def test_snapshot_ids_protected(self):
        svc = self._make_service()
        text = "Running backtest on active-signal-v1 snapshot"
        protected, placeholders = svc._protect_terms(text)
        assert "active-signal-v1" not in protected
        assert len(placeholders) == 1
        restored = svc._restore_terms(protected, placeholders)
        assert restored == text

    def test_trading_symbols_protected(self):
        svc = self._make_service()
        text = "Testing on BTCUSDT and ETHUSDT pairs"
        protected, placeholders = svc._protect_terms(text)
        assert "BTCUSDT" not in protected
        assert "ETHUSDT" not in protected
        assert len(placeholders) == 2
        restored = svc._restore_terms(protected, placeholders)
        assert restored == text

    def test_worker_ids_protected(self):
        svc = self._make_service()
        text = "Worker qwen-1 completed task"
        protected, placeholders = svc._protect_terms(text)
        assert "qwen-1" not in protected
        assert len(placeholders) == 1
        restored = svc._restore_terms(protected, placeholders)
        assert restored == text

    def test_plan_markers_protected(self):
        svc = self._make_service()
        text = "Plan v2 completed ETAP 3 successfully"
        protected, placeholders = svc._protect_terms(text)
        assert "Plan v2" not in protected
        assert "ETAP 3" not in protected
        assert len(placeholders) == 2

    def test_urls_protected(self):
        svc = self._make_service()
        text = "See https://api.telegram.org/bot123/send for details"
        protected, placeholders = svc._protect_terms(text)
        assert "https://api.telegram.org/bot123/send" not in protected
        assert len(placeholders) == 1

    def test_dataset_paths_protected(self):
        svc = self._make_service()
        text = "Loaded binance/um/BTCUSDT/1h dataset"
        protected, placeholders = svc._protect_terms(text)
        assert "binance/um/BTCUSDT/1h" not in protected
        assert len(placeholders) == 1

    def test_mixed_text_protection(self):
        svc = self._make_service()
        text = "Testing cf_alpha_1 on BTCUSDT with active-signal-v1"
        protected, placeholders = svc._protect_terms(text)
        assert len(placeholders) == 3
        restored = svc._restore_terms(protected, placeholders)
        assert restored == text

    def test_no_technical_terms(self):
        svc = self._make_service()
        text = "Hello world, this is a simple message"
        protected, placeholders = svc._protect_terms(text)
        assert protected == text
        assert len(placeholders) == 0

    def test_uuid_protected(self):
        svc = self._make_service()
        text = "Task a1b2c3d4-e5f6-7890-abcd-ef1234567890 failed"
        protected, placeholders = svc._protect_terms(text)
        assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" not in protected
        assert len(placeholders) == 1


# ---------------------------------------------------------------------------
# Tests: translate method behavior
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_disabled_returns_original(self):
        svc = TranslationService(translate=False)
        text = "Hello, how are you?"
        assert svc.translate(text) == text

    def test_lazy_load_failure_returns_original(self):
        svc = TranslationService(translate=True)
        text = "Hello, how are you?"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(svc, "load_model", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
            assert svc.translate(text) == text

    def test_empty_text_returns_empty(self):
        svc = TranslationService(translate=False)
        assert svc.translate("") == ""
        assert svc.translate("   ") == "   "

    def test_translate_with_mock_provider(self):
        svc = TranslationService(translate=True, provider="lmstudio")
        svc._model_loaded = True
        # Mock the translator to return a translation
        from unittest.mock import MagicMock
        mock_t = MagicMock()
        mock_t.translate.return_value = "translated"
        svc._translators["lmstudio"] = mock_t
        assert svc.translate("Hello") == "translated"


# ---------------------------------------------------------------------------
# Tests: fallback chain
# ---------------------------------------------------------------------------


class TestFallbackChain:
    def test_primary_succeeds_no_fallback(self):
        from unittest.mock import MagicMock
        svc = TranslationService(
            translate=True, provider="lmstudio",
            fallback_1="qwen_cli",
        )
        svc._model_loaded = True
        mock_lm = MagicMock()
        mock_lm.translate.return_value = "primary result"
        svc._translators["lmstudio"] = mock_lm
        svc._translators["qwen_cli"] = MagicMock()

        result = svc.translate("Hello")
        assert result == "primary result"
        svc._translators["qwen_cli"].translate.assert_not_called()

    def test_fallback_used_when_primary_fails(self):
        from unittest.mock import MagicMock
        svc = TranslationService(
            translate=True, provider="lmstudio",
            fallback_1="qwen_cli",
        )
        svc._model_loaded = True

        mock_lm = MagicMock()
        mock_lm.translate.side_effect = Exception("LM down")
        mock_qwen = MagicMock()
        mock_qwen.translate.return_value = "fallback result"
        svc._translators["lmstudio"] = mock_lm
        svc._translators["qwen_cli"] = mock_qwen

        result = svc.translate("Hello")
        assert result == "fallback result"

    def test_second_fallback_used(self):
        from unittest.mock import MagicMock
        svc = TranslationService(
            translate=True, provider="lmstudio",
            fallback_1="qwen_cli",
            fallback_2="claude_cli",
        )
        svc._model_loaded = True

        mock_lm = MagicMock()
        mock_lm.translate.side_effect = Exception("LM down")
        mock_qwen = MagicMock()
        mock_qwen.translate.side_effect = Exception("qwen down")
        mock_claude = MagicMock()
        mock_claude.translate.return_value = "claude result"
        svc._translators["lmstudio"] = mock_lm
        svc._translators["qwen_cli"] = mock_qwen
        svc._translators["claude_cli"] = mock_claude

        result = svc.translate("Hello")
        assert result == "claude result"

    def test_all_fail_returns_original(self):
        from unittest.mock import MagicMock
        svc = TranslationService(
            translate=True, provider="lmstudio",
            fallback_1="qwen_cli",
        )
        svc._model_loaded = True

        mock_lm = MagicMock()
        mock_lm.translate.side_effect = Exception("fail")
        mock_qwen = MagicMock()
        mock_qwen.translate.side_effect = Exception("fail")
        svc._translators["lmstudio"] = mock_lm
        svc._translators["qwen_cli"] = mock_qwen

        result = svc.translate("Hello world")
        assert result == "Hello world"

    def test_off_provider_skipped(self):
        svc = TranslationService(
            translate=True, provider="lmstudio",
            fallback_1="off",
        )
        svc._model_loaded = True
        # "off" should not be in translators
        assert "off" not in svc._translators


# ---------------------------------------------------------------------------
# Tests: load_model graceful degradation
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_disabled_does_not_load(self):
        svc = TranslationService(translate=False)
        svc.load_model()
        assert not svc.is_ready

    def test_enabled_but_not_loaded_returns_original(self):
        svc = TranslationService(translate=True)
        text = "This should pass through unchanged"
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(svc, "load_model", lambda: None)
            assert svc.translate(text) == text


# ---------------------------------------------------------------------------
# Tests: is_enabled / is_ready properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_enabled_false(self):
        svc = TranslationService(translate=False)
        assert svc.is_enabled is False

    def test_is_enabled_true(self):
        svc = TranslationService(translate=True)
        assert svc.is_enabled is True

    def test_is_ready_false_before_load(self):
        svc = TranslationService(translate=True)
        assert svc.is_ready is False


# ---------------------------------------------------------------------------
# Tests: translation cache
# ---------------------------------------------------------------------------


class TestTranslationCache:
    def test_cache_stores_result(self):
        svc = TranslationService(translate=True)
        svc._cache["test key"] = "test value"
        assert "test key" in svc._cache
        assert svc._cache["test key"] == "test value"

    def test_cache_max_size(self):
        from app.services.translation_service import _CACHE_MAX
        assert _CACHE_MAX == 200

    def test_cache_eviction(self):
        from collections import OrderedDict
        svc = TranslationService(translate=True)
        svc._cache = OrderedDict()
        for i in range(205):
            svc._cache[f"key_{i}"] = f"val_{i}"
            if len(svc._cache) > 200:
                svc._cache.popitem(last=False)
        assert len(svc._cache) == 200
        assert "key_0" not in svc._cache
        assert "key_4" not in svc._cache
        assert "key_5" in svc._cache
        assert "key_204" in svc._cache


# ---------------------------------------------------------------------------
# Tests: provider config
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_default_provider(self):
        svc = TranslationService(translate=True)
        assert svc._provider == "lmstudio"

    def test_custom_provider(self):
        svc = TranslationService(translate=True, provider="qwen_cli")
        assert svc._provider == "qwen_cli"
