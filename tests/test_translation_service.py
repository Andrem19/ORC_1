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

    def test_model_dir_default(self):
        cfg = NotificationConfig()
        assert cfg.translation_model_dir == "models/opus-mt-en-ru"

    def test_from_dict(self):
        data = {
            "notifications": {
                "enabled": True,
                "translate_to_russian": True,
                "translation_model_dir": "custom/path",
            }
        }
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translate_to_russian is True
        assert cfg.notifications.translation_model_dir == "custom/path"

    def test_missing_translate_key_defaults_false(self):
        data = {"notifications": {"enabled": True}}
        cfg = load_config_from_dict(data)
        assert cfg.notifications.translate_to_russian is False


# ---------------------------------------------------------------------------
# Tests: term protection
# ---------------------------------------------------------------------------


class TestTermProtection:
    def _make_service(self) -> TranslationService:
        return TranslationService(translate_to_russian=True)

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

    def test_decision_values_protected(self):
        svc = self._make_service()
        text = "launch_worker was the decision"
        protected, placeholders = svc._protect_terms(text)
        assert "launch_worker" not in protected
        assert len(placeholders) == 1

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
        # All three technical terms should be replaced
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
        svc = TranslationService(translate_to_russian=False)
        text = "Hello, how are you?"
        assert svc.translate(text) == text

    def test_model_not_loaded_returns_original(self):
        svc = TranslationService(translate_to_russian=True)
        # Model not loaded — should return original text
        text = "Hello, how are you?"
        assert svc.translate(text) == text

    def test_empty_text_returns_empty(self):
        svc = TranslationService(translate_to_russian=False)
        assert svc.translate("") == ""
        assert svc.translate("   ") == "   "


# ---------------------------------------------------------------------------
# Tests: long text splitting
# ---------------------------------------------------------------------------


class TestSplitLongText:
    def _make_service(self) -> TranslationService:
        return TranslationService(translate_to_russian=True)

    def test_short_text_not_split(self):
        svc = self._make_service()
        text = "This is a short text. It has few words."
        chunks = svc._split_long_text(text, max_words=80)
        assert chunks == [text]

    def test_long_text_split_at_sentences(self):
        svc = self._make_service()
        # Create text with many sentences
        sentences = [f"Sentence number {i} here." for i in range(20)]
        text = " ".join(sentences)
        chunks = svc._split_long_text(text, max_words=10)
        assert len(chunks) > 1
        # Each chunk should have words <= max_words (approximately)
        for chunk in chunks:
            assert len(chunk.split()) <= 20  # Allow some slack for boundary splitting


# ---------------------------------------------------------------------------
# Tests: load_model graceful degradation
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_disabled_does_not_load(self):
        svc = TranslationService(translate_to_russian=False)
        svc.load_model()
        assert svc._model is None
        assert not svc.is_ready

    def test_enabled_without_transformers_degrades(self):
        svc = TranslationService(translate_to_russian=True)
        # transformers may not be installed in test env
        svc.load_model()
        # Should not crash — either loads or degrades gracefully
        assert True  # If we get here, no exception was raised

    def test_enabled_but_not_loaded_returns_original(self):
        svc = TranslationService(translate_to_russian=True)
        # Don't call load_model
        text = "This should pass through unchanged"
        assert svc.translate(text) == text


# ---------------------------------------------------------------------------
# Tests: is_enabled / is_ready properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_is_enabled_false(self):
        svc = TranslationService(translate_to_russian=False)
        assert svc.is_enabled is False

    def test_is_enabled_true(self):
        svc = TranslationService(translate_to_russian=True)
        assert svc.is_enabled is True

    def test_is_ready_false_before_load(self):
        svc = TranslationService(translate_to_russian=True)
        assert svc.is_ready is False
