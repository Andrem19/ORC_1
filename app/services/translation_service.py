"""
Translation service — English to Russian.

Supports two backends:
  - "opus": Helsinki-NLP/opus-mt-en-ru (local MarianMT model)
  - "lmstudio": LM Studio chat completions API (e.g. qwen models)

Protects technical terms (snapshot IDs, feature names, symbols, etc.)
from translation using a placeholder approach:
  1. Replace technical terms with __TK0__, __TK1__, ... placeholders
  2. Translate the protected text
  3. Restore original terms in place of placeholders
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

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

# System prompt for LM Studio translation.
_LMSTUDIO_SYSTEM_PROMPT = """You are a professional English-to-Russian translator for software monitoring notifications. Translate the user message from English to Russian.

Rules:
- Translate naturally, preserving the meaning accurately
- Keep all placeholders like __TK0__, __TK1__ exactly as-is — they are technical identifiers
- Keep numbers, percentages, and metric values unchanged
- Keep the same line structure — do not merge or split lines
- Be concise — these are short status notifications, not literary text
- Do not add any explanations, comments, or extra text — output ONLY the translation"""


class LMStudioTranslator:
    """Translates text via LM Studio /v1/chat/completions API."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234",
        model: str = "",
        max_tokens: int = 1024,
        timeout_seconds: int = 30,
        reasoning_effort: str = "",
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout_seconds
        self._reasoning_effort = reasoning_effort
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._available_checked: bool = False
        self._available: bool = False

    @property
    def is_available(self) -> bool:
        return self._available

    def check_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        try:
            parsed = urlparse(self._base_url)
            conn = HTTPConnection(
                parsed.hostname, parsed.port or 1234, timeout=5,
            )
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            conn.close()
            ok = resp.status == 200
            if ok and not self._available_checked:
                logger.info(
                    "LM Studio translator available at %s", self._base_url,
                )
            self._available = ok
            self._available_checked = True
            return ok
        except Exception:
            self._available = False
            self._available_checked = True
            return False

    def translate(self, text: str) -> str:
        """Translate text via LM Studio chat completions."""
        if not text.strip():
            return text

        # Check cache
        if text in self._cache:
            self._cache.move_to_end(text)
            return self._cache[text]

        body: dict = {
            "messages": [
                {"role": "system", "content": _LMSTUDIO_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
            "max_tokens": self._max_tokens,
        }
        if self._reasoning_effort:
            body["reasoning_effort"] = self._reasoning_effort
        if self._model:
            body["model"] = self._model

        try:
            parsed = urlparse(self._base_url)
            conn = HTTPConnection(
                parsed.hostname, parsed.port or 1234,
                timeout=self._timeout,
            )
            headers = {"Content-Type": "application/json"}
            conn.request(
                "POST", "/v1/chat/completions", json.dumps(body), headers,
            )
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            conn.close()

            if resp.status != 200:
                logger.warning(
                    "LM Studio translation returned HTTP %d: %s",
                    resp.status, resp_body[:200],
                )
                return text

            data = json.loads(resp_body)
            translated = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()

            if not translated:
                return text

            # Cache result
            self._cache[text] = translated
            if len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)

            return translated

        except Exception as e:
            logger.warning("LM Studio translation failed: %s", e)
            return text


class TranslationService:
    """Translates English text to Russian, preserving technical terms.

    Supports two backends:
      - "opus": local MarianMT model (Helsinki-NLP/opus-mt-en-ru)
      - "lmstudio": LM Studio chat completions API
    """

    def __init__(
        self,
        translate_to_russian: bool = False,
        model_dir: str = "models/opus-mt-en-ru",
        model_name: str = "Helsinki-NLP/opus-mt-en-ru",
        backend: str = "opus",
        lmstudio_base_url: str = "http://localhost:1234",
        lmstudio_model: str = "",
        lmstudio_max_tokens: int = 1024,
        lmstudio_timeout_seconds: int = 30,
        lmstudio_reasoning_effort: str = "",
    ) -> None:
        self._translate = translate_to_russian
        self._backend = backend
        self._model_dir = Path(model_dir)
        self._model_name = model_name
        self._tokenizer = None
        self._model = None
        self._model_loaded = False
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._lmstudio_translator: LMStudioTranslator | None = None
        if backend == "lmstudio" and translate_to_russian:
            self._lmstudio_translator = LMStudioTranslator(
                base_url=lmstudio_base_url,
                model=lmstudio_model,
                max_tokens=lmstudio_max_tokens,
                timeout_seconds=lmstudio_timeout_seconds,
                reasoning_effort=lmstudio_reasoning_effort,
            )

    @property
    def is_enabled(self) -> bool:
        return self._translate

    @property
    def is_ready(self) -> bool:
        if self._backend == "lmstudio":
            return self._model_loaded and self._lmstudio_translator is not None
        return self._model_loaded and self._model is not None

    def load_model(self) -> None:
        """Load the translation backend.

        For "opus" backend: downloads/loads the MarianMT model.
        For "lmstudio" backend: checks LM Studio server availability.

        Raises RuntimeError if required packages are missing or backend fails.
        """
        if not self._translate:
            return

        if self._backend == "lmstudio":
            self._load_lmstudio()
        else:
            self._load_opus()

    def _load_lmstudio(self) -> None:
        """Check LM Studio availability for translation."""
        if self._lmstudio_translator is None:
            raise RuntimeError(
                "LM Studio translator not configured. "
                "Set translation_backend='lmstudio' and translate_to_russian=true."
            )
        if not self._lmstudio_translator.check_available():
            raise RuntimeError(
                f"LM Studio server not reachable at "
                f"{self._lmstudio_translator._base_url}. "
                f"Start LM Studio with a loaded model first."
            )
        self._model_loaded = True
        logger.info(
            "LM Studio translation backend ready (%s)",
            self._lmstudio_translator._base_url,
        )

    def _load_opus(self) -> None:
        """Download (if needed) and load the MarianMT translation model."""

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

        Returns original text if translation is disabled or backend is not loaded.
        """
        if not self._translate or not self._model_loaded:
            return text

        if not text.strip():
            return text

        if self._backend == "lmstudio":
            return self._translate_lmstudio(text)
        return self._translate_opus(text)

    def _translate_lmstudio(self, text: str) -> str:
        """Translate via LM Studio with term protection."""
        if self._lmstudio_translator is None:
            return text
        try:
            protected, placeholders = self._protect_terms(text)

            # Check cache
            if protected in self._cache:
                self._cache.move_to_end(protected)
                return self._restore_terms(self._cache[protected], placeholders)

            translated = self._lmstudio_translator.translate(protected)

            # Store in cache
            self._cache[protected] = translated
            if len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)

            return self._restore_terms(translated, placeholders)
        except Exception as e:
            logger.error("LM Studio translation failed, sending English: %s", e)
            return text

    def _translate_opus(self, text: str) -> str:
        """Translate via MarianMT (opus) with term protection."""
        if self._model is None:
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
