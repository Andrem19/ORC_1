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
from collections import OrderedDict
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

# Maximum cache entries for translation results.
_CACHE_MAX = 200

# Max tokens to send to the model per chunk (leaves headroom from 512 limit).
_MAX_TOKENS = 400


class TranslationService:
    """Translates English text to Russian, preserving technical terms."""

    def __init__(
        self,
        translate_to_russian: bool = False,
        model_dir: str = "models/opus-mt-en-ru",
        model_name: str = "Helsinki-NLP/opus-mt-en-ru",
    ) -> None:
        self._translate = translate_to_russian
        self._model_dir = Path(model_dir)
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._model_loaded = False
        self._cache: OrderedDict[str, str] = OrderedDict()

    @property
    def is_enabled(self) -> bool:
        return self._translate

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._model is not None

    def load_model(self) -> None:
        """Download (if needed) and load the translation model.

        On first call, downloads the model from HuggingFace
        and saves it to model_dir. On subsequent calls, loads from disk.
        Called at startup from main.py when translate_to_russian is True.

        Raises RuntimeError if required packages are missing or model fails to load.
        """
        if not self._translate:
            return

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            raise RuntimeError(
                "translate_to_russian is enabled but required packages are missing. "
                "Install them in the project conda environment: "
                "conda run -n env6 pip install transformers torch sentencepiece"
            )

        try:
            config_path = self._model_dir / "config.json"
            if config_path.exists():
                logger.info("Loading translation model from %s", self._model_dir)
                self._tokenizer = MarianTokenizer.from_pretrained(self._model_dir)
                self._model = MarianMTModel.from_pretrained(self._model_dir)
            else:
                logger.info(
                    "Downloading %s to %s ...",
                    self._model_name,
                    self._model_dir,
                )
                self._model_dir.mkdir(parents=True, exist_ok=True)
                self._tokenizer = MarianTokenizer.from_pretrained(
                    self._model_name
                )
                self._model = MarianMTModel.from_pretrained(
                    self._model_name
                )
                self._tokenizer.save_pretrained(self._model_dir)
                self._model.save_pretrained(self._model_dir)
                logger.info("Model saved to %s", self._model_dir)

            self._model_loaded = True
            logger.info("Translation model ready")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load translation model: {e}"
            ) from e

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

            # Check cache
            cache_key = protected
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return self._restore_terms(self._cache[cache_key], placeholders)

            translated = self._do_translate(protected)
            self._restore_placeholders(translated, placeholders)

            # Store in cache
            self._cache[cache_key] = translated
            if len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)

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
        """Replace placeholders back with original technical terms.

        The MT model may mangle placeholder format (e.g. strip underscores),
        so we use fuzzy regex matching as a fallback.
        """
        for placeholder, original in placeholder_map.items():
            if placeholder in text:
                text = text.replace(placeholder, original)
                continue
            # Fuzzy fallback: MT model may modify surrounding underscores
            idx_match = re.search(r'\d+', placeholder)
            if idx_match:
                idx = idx_match.group()
                fuzzy = re.compile(r'_*TK' + idx + r'_*')
                text = fuzzy.sub(original, text)
        return text

    def _restore_placeholders(self, text: str, placeholder_map: dict[str, str]) -> str:
        """In-place restore for cached translations (modifies text string is immutable,
        but we update the cache entry). This is a no-op helper for clarity."""
        pass

    def _do_translate(self, text: str) -> str:
        """Translate protected text using MarianMT model.

        Splits into paragraph-level chunks and batch-translates them
        for better performance.
        """
        import torch

        paragraphs = text.split("\n")
        translated_paragraphs: list[str] = []

        # Collect all translatable chunks across paragraphs, preserving structure.
        all_chunks: list[str] = []
        chunk_indices: list[tuple[int, int]] = []  # (para_idx, chunk_idx)

        for p_idx, para in enumerate(paragraphs):
            if not para.strip():
                continue
            chunks = self._split_long_text(para)
            for c_idx, chunk in enumerate(chunks):
                if chunk.strip():
                    all_chunks.append(chunk)
                    chunk_indices.append((p_idx, c_idx))

        # Batch translate all chunks at once
        if all_chunks:
            inputs = self._tokenizer(
                all_chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=_MAX_TOKENS,
            )
            with torch.no_grad():
                outputs = self._model.generate(**inputs)
            decoded = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)
        else:
            decoded = []

        # Map results back to paragraph structure
        chunk_results: dict[tuple[int, int], str] = {}
        for i, (p_idx, c_idx) in enumerate(chunk_indices):
            chunk_results[(p_idx, c_idx)] = decoded[i]

        for p_idx, para in enumerate(paragraphs):
            if not para.strip():
                translated_paragraphs.append(para)
                continue
            chunks = self._split_long_text(para)
            translated_chunks = []
            for c_idx, chunk in enumerate(chunks):
                key = (p_idx, c_idx)
                if key in chunk_results:
                    translated_chunks.append(chunk_results[key])
                elif chunk.strip():
                    translated_chunks.append(chunk)
            translated_paragraphs.append(" ".join(translated_chunks))

        return "\n".join(translated_paragraphs)

    def _split_long_text(self, text: str, max_tokens: int = _MAX_TOKENS) -> list[str]:
        """Split text into chunks that fit within *max_tokens*.

        Uses the loaded tokenizer for accurate token counting.
        Falls back to word-based splitting if tokenizer is unavailable.
        """
        if self._tokenizer is not None:
            return self._split_by_tokens(text, max_tokens)
        return self._split_by_words(text)

    def _split_by_tokens(self, text: str, max_tokens: int) -> list[str]:
        """Split text at sentence boundaries respecting token limits."""
        tokens = self._tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_tokens:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            sent_tokens = len(self._tokenizer.encode(sentence, add_special_tokens=False))
            if current and current_len + sent_tokens > max_tokens:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(sentence)
            current_len += sent_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks if chunks else [text]

    def _split_by_words(self, text: str, max_words: int = 80) -> list[str]:
        """Fallback word-based splitting when tokenizer is unavailable."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

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
