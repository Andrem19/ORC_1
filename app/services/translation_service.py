"""
Translation service — English to Russian with fallback chain.

Supports providers:
  - "lmstudio": LM Studio chat completions API
  - "qwen_cli": qwen-code CLI subprocess
  - "claude_cli": claude CLI subprocess
  - "off": disabled

Falls back through: primary -> fallback_1 -> fallback_2.
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
import shutil
import subprocess
from collections import OrderedDict
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

from app.lmstudio_api import validate_lmstudio_endpoint

logger = logging.getLogger("orchestrator.translation")

# Compiled regex for technical terms to protect from translation.
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
    # Planner plan markers
    r"\bPlan v\d+\b",
    r"\bETAP \d+\b",
    # UUIDs
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
]

_TECH_RE = re.compile("|".join(f"(?:{p})" for p in _TECH_PATTERNS))

# Maximum cache entries for translation results.
_CACHE_MAX = 200

# System prompt for LM Studio translation.
_LMSTUDIO_SYSTEM_PROMPT = """You are a professional English-to-Russian translator for software monitoring notifications. Translate the user message from English to Russian.

Rules:
- Translate naturally, preserving the meaning accurately
- Keep all placeholders like __TK0__, __TK1__ exactly as-is — they are technical identifiers
- Keep numbers, percentages, and metric values unchanged
- Keep the same line structure — do not merge or split lines
- Be concise — these are short status notifications, not literary text
- Do not add any explanations, comments, or extra text — output ONLY the translation"""

_LMSTUDIO_PROBE_TEXT = "Orchestrator started"
_LMSTUDIO_PROBE_TIMEOUT_SECONDS = 5
_LMSTUDIO_PROBE_MAX_TOKENS = 48

# System prompt for CLI translators.
_CLI_TRANSLATION_PROMPT = (
    "Translate the following English text to Russian. "
    "Output ONLY the translation, nothing else. "
    "Keep all __TK__ placeholders, numbers, and technical identifiers unchanged. "
    "Be concise and natural.\n\n"
)


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

    def _normalized_reasoning_effort(self) -> str:
        effort = str(self._reasoning_effort or "").strip().lower()
        if effort == "off":
            return "none"
        return effort

    def _build_body(self, text: str, *, max_tokens: int | None = None) -> dict:
        body: dict = {
            "messages": [
                {"role": "system", "content": _LMSTUDIO_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
        }
        reasoning_effort = self._normalized_reasoning_effort()
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        if self._model:
            body["model"] = self._model
        return body

    def _request_translation(
        self,
        text: str,
        *,
        timeout: int | None = None,
        max_tokens: int | None = None,
        log_prefix: str = "LM Studio translation",
    ) -> str:
        body = self._build_body(text, max_tokens=max_tokens)
        parsed = urlparse(self._base_url)
        conn = HTTPConnection(
            parsed.hostname, parsed.port or 1234,
            timeout=timeout or self._timeout,
        )
        try:
            headers = {"Content-Type": "application/json"}
            conn.request(
                "POST", "/v1/chat/completions", json.dumps(body), headers,
            )
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            if resp.status != 200:
                logger.warning(
                    "%s returned HTTP %d: %s",
                    log_prefix,
                    resp.status,
                    resp_body[:200],
                )
                return ""

            data = json.loads(resp_body)
            translated = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
            if translated:
                return translated

            reasoning = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("reasoning_content", "")
            ).strip()
            logger.warning(
                "%s returned empty assistant content (reasoning_chars=%d)",
                log_prefix,
                len(reasoning),
            )
            return ""
        finally:
            conn.close()

    def _probe_generation(self) -> bool:
        try:
            translated = self._request_translation(
                _LMSTUDIO_PROBE_TEXT,
                timeout=min(self._timeout, _LMSTUDIO_PROBE_TIMEOUT_SECONDS),
                max_tokens=min(self._max_tokens, _LMSTUDIO_PROBE_MAX_TOKENS),
                log_prefix="LM Studio translation probe",
            )
        except Exception as exc:
            logger.warning("LM Studio translation probe failed: %s", exc)
            return False
        return bool(translated)

    def check_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        try:
            ok, models = validate_lmstudio_endpoint(
                base_url=self._base_url,
                model=self._model,
                timeout=5,
            )
            if not ok and self._model and models:
                logger.warning(
                    "LM Studio translator model '%s' not found at %s (available=%s)",
                    self._model,
                    self._base_url,
                    ", ".join(models[:10]),
                )
            elif ok and not self._probe_generation():
                logger.warning(
                    "LM Studio translator chat probe failed at %s; endpoint is reachable "
                    "but the model is not meeting translation latency/content requirements",
                    self._base_url,
                )
                ok = False
            self._available = ok
            self._available_checked = True
            if ok:
                logger.info(
                    "LM Studio translator available at %s", self._base_url,
                )
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

        try:
            translated = self._request_translation(
                text,
                log_prefix="LM Studio translation",
            )
            if not translated:
                self._available = False
                self._available_checked = True
                return text

            # Cache result
            self._cache[text] = translated
            if len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
            self._available = True
            self._available_checked = True

            return translated

        except Exception as e:
            logger.warning("LM Studio translation failed: %s", e)
            self._available = False
            self._available_checked = True
            return text


class CliTranslator:
    """Translates text via a CLI subprocess (qwen-code or claude)."""

    def __init__(self, cli_path: str, timeout: int = 60) -> None:
        self._cli_path = cli_path
        self._timeout = timeout
        self._resolved: str | None = None

    def check_available(self) -> bool:
        """Check if the CLI binary exists."""
        resolved = self._resolve()
        return bool(resolved)

    def _resolve(self) -> str:
        if self._resolved:
            return self._resolved
        direct = shutil.which(self._cli_path)
        if direct:
            self._resolved = direct
            return direct
        explicit = Path(self._cli_path).expanduser()
        if explicit.exists() and explicit.is_file():
            self._resolved = str(explicit)
            return str(explicit)
        return ""

    def _build_command(self, prompt: str, resolved: str) -> list[str]:
        cli_name = Path(resolved).name.lower()
        if cli_name == "claude":
            return [resolved, "--print", "--output-format", "text", prompt]
        return [resolved, "-p", prompt, "-e", "none", "-o", "text"]

    def translate(self, text: str) -> str:
        """Translate text via CLI subprocess."""
        if not text.strip():
            return text

        resolved = self._resolve()
        if not resolved:
            return text

        prompt = f"{_CLI_TRANSLATION_PROMPT}{text}"
        try:
            result = subprocess.run(
                self._build_command(prompt, resolved),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            if result.returncode != 0:
                logger.warning(
                    "CLI translation failed (%s) exit=%d: %s",
                    self._cli_path,
                    result.returncode,
                    (result.stderr or "").strip()[:200],
                )
                return text
            output = (result.stdout or "").strip()
            if not output:
                return text
            return output
        except subprocess.TimeoutExpired:
            logger.warning("CLI translation timed out (%s)", self._cli_path)
            return text
        except Exception as e:
            logger.warning("CLI translation failed (%s): %s", self._cli_path, e)
            return text


class TranslationService:
    """Translates English text to Russian with fallback chain.

    Providers: lmstudio, qwen_cli, claude_cli, or "off".
    Falls back through primary -> fallback_1 -> fallback_2.
    """

    def __init__(
        self,
        *,
        translate: bool = False,
        provider: str = "lmstudio",
        fallback_1: str = "",
        fallback_2: str = "",
        lmstudio_base_url: str = "http://localhost:1234",
        lmstudio_model: str = "",
        lmstudio_max_tokens: int = 1024,
        lmstudio_timeout_seconds: int = 30,
        lmstudio_reasoning_effort: str = "",
    ) -> None:
        self._translate = translate
        self._provider = provider.strip().lower() if provider else "off"
        self._fallback_1 = fallback_1.strip().lower() if fallback_1 else ""
        self._fallback_2 = fallback_2.strip().lower() if fallback_2 else ""
        self._active_provider = self._provider
        self._load_attempted = False
        self._model_loaded = False
        self._cache: OrderedDict[str, str] = OrderedDict()

        # Build translators lazily
        self._translators: dict[str, LMStudioTranslator | CliTranslator] = {}

        if translate and self._provider != "off":
            self._ensure_translator(
                self._provider,
                lmstudio_base_url, lmstudio_model,
                lmstudio_max_tokens, lmstudio_timeout_seconds,
                lmstudio_reasoning_effort,
            )
        if translate and self._fallback_1 and self._fallback_1 not in ("off", "none"):
            self._ensure_translator(self._fallback_1)
        if translate and self._fallback_2 and self._fallback_2 not in ("off", "none"):
            self._ensure_translator(self._fallback_2)

    def _ensure_translator(
        self,
        name: str,
        base_url: str = "",
        model: str = "",
        max_tokens: int = 1024,
        timeout: int = 30,
        reasoning_effort: str = "",
    ) -> None:
        if name in self._translators:
            return
        if name == "lmstudio":
            self._translators[name] = LMStudioTranslator(
                base_url=base_url or "http://localhost:1234",
                model=model,
                max_tokens=max_tokens,
                timeout_seconds=timeout,
                reasoning_effort=reasoning_effort,
            )
        elif name == "qwen_cli":
            self._translators[name] = CliTranslator(cli_path="qwen")
        elif name == "claude_cli":
            self._translators[name] = CliTranslator(cli_path="claude")

    @property
    def is_enabled(self) -> bool:
        return self._translate

    @property
    def is_ready(self) -> bool:
        return self._model_loaded

    def load_model(self) -> None:
        """Initialize the primary translation provider.

        For LM Studio: checks server availability.
        For CLI providers: checks binary availability.
        Raises RuntimeError if the primary provider is not available.
        """
        if not self._translate:
            return
        self._load_attempted = True

        if self._provider == "off" or not self._provider:
            logger.info("Translation disabled (provider=off)")
            return

        translator = self._translators.get(self._provider)
        if translator is None:
            raise RuntimeError(
                f"Unknown translation provider: {self._provider!r}. "
                f"Supported: lmstudio, qwen_cli, claude_cli, off"
            )

        available = translator.check_available()
        if not available:
            # Try fallbacks
            for fb_name in (self._fallback_1, self._fallback_2):
                if not fb_name or fb_name in ("off", "none", ""):
                    continue
                fb_translator = self._translators.get(fb_name)
                if fb_translator and fb_translator.check_available():
                    logger.warning(
                        "Primary translator %r unavailable, using fallback %r",
                        self._provider, fb_name,
                    )
                    self._active_provider = fb_name
                    self._model_loaded = True
                    return
            raise RuntimeError(
                f"Translation provider {self._provider!r} is not available "
                f"and no fallback is available."
            )

        self._active_provider = self._provider
        self._model_loaded = True
        logger.info("Translation provider %r ready", self._active_provider)

    def translate(self, text: str) -> str:
        """Translate English text to Russian, preserving technical terms.

        Returns original text if translation is disabled or all providers fail.
        """
        if not self._translate:
            return text

        if not self._model_loaded and not self._load_attempted:
            try:
                self.load_model()
            except Exception as exc:
                logger.warning("Translation unavailable, sending original text: %s", exc)

        if not self._model_loaded:
            return text

        if not text.strip():
            return text

        # Try primary, then fallbacks
        providers: list[str] = []
        for name in (
            self._active_provider,
            self._provider,
            self._fallback_1,
            self._fallback_2,
        ):
            if not name or name in ("off", "none") or name in providers:
                continue
            providers.append(name)

        protected, placeholders = self._protect_terms(text)

        # Check cache
        if protected in self._cache:
            self._cache.move_to_end(protected)
            return self._restore_terms(self._cache[protected], placeholders)

        for provider_name in providers:
            translator = self._translators.get(provider_name)
            if translator is None:
                continue
            if (
                isinstance(translator, LMStudioTranslator)
                and translator._available_checked
                and not translator.is_available
            ):
                continue
            try:
                translated = translator.translate(protected)
                if translated != protected and translated.strip():
                    # Cache and restore
                    self._cache[protected] = translated
                    if len(self._cache) > _CACHE_MAX:
                        self._cache.popitem(last=False)
                    self._active_provider = provider_name
                    return self._restore_terms(translated, placeholders)
            except Exception as e:
                logger.warning(
                    "Translation provider %r failed: %s", provider_name, e,
                )
                continue

        # All providers failed or returned unchanged text
        return text

    # ---------------------------------------------------------------
    # Internal: term protection
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
