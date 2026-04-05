"""
Translation service — English to Russian via Helsinki-NLP/opus-mt-en-ru.

Protects technical terms (snapshot IDs, feature names, symbols, etc.)
from translation using a placeholder approach:
  1. Replace technical terms with __TK0__, __TK1__, ... placeholders
  2. Translate the protected text
  3. Restore original terms in place of placeholders
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger("orchestrator.translation")

# Compiled regex for technical terms to protect from translation.
# Ordered by specificity: most specific patterns first.
_TECH_PATTERNS = [
    # URLs
    r"https?://\S+",
    # Dataset paths: binance/um/BTCUSDT/1h
    r"binance/[a-z]+/[A-Z]+/\d+[mh]",
    # Feature names: cf_*, cl_*, rsi_*, ema_*, sma_*, macd_*, bb_*, atr_*, vol_*, obv_*, adx_*
    r"\b(?:cf|cl|rsi|ema|sma|macd|bb|atr|vol|obv|adx)_[a-zA-Z0-9_]+",
    # Trading symbols: BTCUSDT, ETHUSDT, etc.
    r"\b[A-Z]{3,5}USDT\b",
    # Snapshot IDs: active-signal-v1, snap_xxx, any-word-vN
    r"\bactive-signal[\w-]*",
    r"\bsnap_[\w-]+",
    r"\b[\w]+-v\d+\b",
    # Worker IDs: qwen-1, qwen-2
    r"\bqwen-\d+\b",
    # Decision enum values
    r"\b(?:launch_worker|wait|finish|retry_worker|stop_worker|reassign_task)\b",
    # UUIDs
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
]

_TECH_RE = re.compile("|".join(f"(?:{p})" for p in _TECH_PATTERNS))


class TranslationService:
    """Translates English text to Russian, preserving technical terms."""

    def __init__(
        self,
        translate_to_russian: bool = False,
        model_dir: str = "models/opus-mt-en-ru",
    ) -> None:
        self._translate = translate_to_russian
        self._model_dir = Path(model_dir)
        self._tokenizer = None
        self._model = None
        self._model_loaded = False

    @property
    def is_enabled(self) -> bool:
        return self._translate

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._model is not None

    def load_model(self) -> None:
        """Download (if needed) and load the translation model.

        On first call, downloads Helsinki-NLP/opus-mt-en-ru from HuggingFace
        and saves it to model_dir. On subsequent calls, loads from disk.
        Called at startup from main.py when translate_to_russian is True.
        """
        if not self._translate:
            return

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            logger.warning(
                "transformers/torch not installed. "
                "Notifications will be sent in English. "
                "Install: pip install transformers torch sentencepiece"
            )
            return

        try:
            config_path = self._model_dir / "config.json"
            if config_path.exists():
                logger.info("Loading translation model from %s", self._model_dir)
                self._tokenizer = MarianTokenizer.from_pretrained(self._model_dir)
                self._model = MarianMTModel.from_pretrained(self._model_dir)
            else:
                logger.info(
                    "Downloading Helsinki-NLP/opus-mt-en-ru to %s ...",
                    self._model_dir,
                )
                self._model_dir.mkdir(parents=True, exist_ok=True)
                self._tokenizer = MarianTokenizer.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-ru"
                )
                self._model = MarianMTModel.from_pretrained(
                    "Helsinki-NLP/opus-mt-en-ru"
                )
                self._tokenizer.save_pretrained(self._model_dir)
                self._model.save_pretrained(self._model_dir)
                logger.info("Model saved to %s", self._model_dir)

            self._model_loaded = True
            logger.info("Translation model ready")
        except Exception as e:
            logger.error("Failed to load translation model: %s", e)
            self._model = None
            self._tokenizer = None

    def translate(self, text: str) -> str:
        """Translate English text to Russian, preserving technical terms.

        Returns original text if translation is disabled or model is not loaded.
        """
        if not self._translate or not self._model_loaded or self._model is None:
            return text

        if not text.strip():
            return text

        try:
            protected, placeholders = self._protect_terms(text)
            translated = self._do_translate(protected)
            restored = self._restore_terms(translated, placeholders)
            return restored
        except Exception as e:
            logger.error("Translation failed, sending English: %s", e)
            return text

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _protect_terms(self, text: str) -> tuple[str, dict[str, str]]:
        """Replace technical terms with numbered placeholders."""
        placeholder_map: dict[str, str] = {}
        counter = [0]

        def _replace(match: re.Match) -> str:
            key = f"__TK{counter[0]}__"
            placeholder_map[key] = match.group(0)
            counter[0] += 1
            return key

        protected = _TECH_RE.sub(_replace, text)
        return protected, placeholder_map

    def _restore_terms(self, text: str, placeholder_map: dict[str, str]) -> str:
        """Replace placeholders back with original technical terms."""
        for placeholder, original in placeholder_map.items():
            text = text.replace(placeholder, original)
        return text

    def _do_translate(self, text: str) -> str:
        """Translate protected text using MarianMT model.

        Splits into paragraph-level chunks to stay within the model's
        512-token limit. Each chunk is translated independently.
        """
        import torch

        paragraphs = text.split("\n")
        translated_paragraphs = []

        for para in paragraphs:
            if not para.strip():
                translated_paragraphs.append(para)
                continue

            # Further split very long paragraphs at sentence boundaries
            chunks = self._split_long_text(para, max_words=80)
            translated_chunks = []

            for chunk in chunks:
                if not chunk.strip():
                    translated_chunks.append(chunk)
                    continue

                inputs = self._tokenizer(
                    chunk, return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():
                    outputs = self._model.generate(**inputs)
                decoded = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
                translated_chunks.append(decoded)

            translated_paragraphs.append(" ".join(translated_chunks))

        return "\n".join(translated_paragraphs)

    def _split_long_text(self, text: str, max_words: int = 80) -> list[str]:
        """Split text into chunks at sentence boundaries if it exceeds max_words."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

        # Split at sentence boundaries (., !, ? followed by space)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_words = sentence.split()
            if current_len + len(sent_words) > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(sentence)
            current_len += len(sent_words)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
