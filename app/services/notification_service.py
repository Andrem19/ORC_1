"""
Notification service — sends Telegram messages via Bot API.

Uses http.client (stdlib) to call the Telegram Bot API directly.
Messages are formatted as HTML for rich display in Telegram.

Credentials are read from environment variables:
  - ALGO_BOT: Telegram bot token
  - CHAT_ID:  Default chat ID to send messages to
"""

from __future__ import annotations

import json
import logging
import os
import time
from http.client import HTTPSConnection

from app.config import NotificationConfig
from app.models import OrchestratorState, PlannerOutput, TaskResult
from app.services.tg_html import truncate_html
from app.services.tg_message_builder import (
    TelegramMessageBuilder,
    apply_translated_text,
    render_html,
    render_plain,
)
from app.services.translation_service import TranslationService

logger = logging.getLogger("orchestrator.notifications")

_RETRYABLE_ERRORS = (ConnectionError, OSError, TimeoutError)
_MAX_SEND_ATTEMPTS = 3
_RETRY_DELAY = 2.0


class NotificationService:
    """Sends Telegram notifications for orchestrator events."""

    def __init__(self, config: NotificationConfig | None = None) -> None:
        self.config = config or NotificationConfig()
        self._bot_token: str = os.environ.get("ALGO_BOT", "")
        self._chat_id: str = os.environ.get("CHAT_ID", "")
        self._last_send_time: float = 0.0
        self._enabled = (
            self.config.enabled
            and bool(self._bot_token)
            and bool(self._chat_id)
        )
        self._translator = TranslationService(
            translate_to_russian=self.config.translate_to_russian,
            model_dir=self.config.translation_model_dir,
            model_name=self.config.translation_model_name,
        )
        if self._enabled:
            logger.info(
                "Notifications enabled (chat_id=%s, min_interval=%ds)",
                self._chat_id,
                self.config.min_interval_seconds,
            )

    def is_configured(self) -> bool:
        """Check if Telegram credentials are present."""
        return bool(self._bot_token) and bool(self._chat_id)

    def init_translation(self) -> None:
        """Load the translation model at startup."""
        self._translator.load_model()

    # ---------------------------------------------------------------
    # Public notification methods
    # ---------------------------------------------------------------

    def send_lifecycle(self, event: str, detail: str = "") -> bool:
        """Send a lifecycle notification (start/stop/finish)."""
        if not self._enabled:
            return False
        builder = TelegramMessageBuilder()
        builder.add_header("", f"Orchestrator {event}")
        builder.add_separator()
        if detail:
            builder.add_body(detail)
        return self._send_structured(builder.build())

    def send_worker_result(self, result: TaskResult, cycle: int = 0) -> bool:
        """Send a worker result notification."""
        if not self._enabled:
            return False
        if result.status == "success":
            icon = "\u2705"
            label = "completed"
        else:
            icon = "\u274c"
            label = "failed"

        builder = TelegramMessageBuilder()
        builder.add_header(icon, f"Cycle #{cycle} | Worker {result.worker_id} {label}")
        builder.add_separator()
        builder.add_field("Task", result.task_id)
        if result.summary:
            builder.add_body(result.summary[:300])
        if result.error:
            builder.add_field("Error", result.error[:200])
        if result.confidence > 0:
            builder.add_field("Confidence", f"{result.confidence:.2f}")
        return self._send_structured(builder.build())

    def send_planner_decision(self, output: PlannerOutput, cycle: int = 0) -> bool:
        """Send a planner decision notification."""
        if not self._enabled:
            return False
        builder = TelegramMessageBuilder()
        builder.add_header(
            "\U0001f9d1\u200d\U0001f4bc",
            f"Cycle #{cycle} | Planner decision",
        )
        builder.add_separator()
        builder.add_field("Decision", output.decision.value)
        if output.reason:
            builder.add_body(output.reason[:200])
        if output.target_worker_id:
            builder.add_field("Worker", output.target_worker_id)
        if output.task_instruction:
            builder.add_field("Task", output.task_instruction[:150])
        if output.memory_update:
            builder.add_field("Memory", output.memory_update[:100])
        return self._send_structured(builder.build())

    def send_error(self, error: str, context: str = "") -> bool:
        """Send an error alert."""
        if not self._enabled:
            return False
        title = "Error"
        if context:
            title += f" ({context})"
        builder = TelegramMessageBuilder()
        builder.add_header("\u26a0\ufe0f", title)
        builder.add_separator()
        builder.add_body(error[:300])
        return self._send_structured(builder.build())

    def send_research_summary(self, state: OrchestratorState) -> bool:
        """Send a periodic research summary."""
        if not self._enabled:
            return False
        completed = len(state.completed_tasks())
        failed = len(state.failed_tasks())
        builder = TelegramMessageBuilder()
        builder.add_header("\U0001f4ca", "Research Summary")
        builder.add_separator()
        builder.add_field("Cycle", str(state.current_cycle))
        builder.add_field("Tasks", f"{completed} completed, {failed} failed")
        builder.add_field("Errors", str(state.total_errors))
        if state.last_planner_decision:
            builder.add_field("Last decision", state.last_planner_decision.value)
        return self._send_structured(builder.build())

    # ---------------------------------------------------------------
    # Internal — message pipeline
    # ---------------------------------------------------------------

    def _send_structured(self, sections: list) -> bool:
        """Translate plain text, render to HTML, send to Telegram."""
        plain = render_plain(sections)
        translated = self._translator.translate(plain)
        translated_sections = apply_translated_text(sections, translated)
        html = render_html(translated_sections)
        return self._send_html(html)

    def _send_html(self, html: str) -> bool:
        """Send HTML-formatted message to Telegram with plain-text fallback."""
        if not self._try_send(html, parse_mode="HTML"):
            # Fallback: strip tags and send as plain text
            import re
            plain = re.sub(r"<[^>]+>", "", html)
            plain = plain.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
            return self._try_send(plain[:4096], parse_mode=None)
        return True

    def _try_send(self, text: str, parse_mode: str | None) -> bool:
        """Send a message to Telegram with retry for transient network errors."""
        # Rate limiting
        now = time.monotonic()
        elapsed = now - self._last_send_time
        if elapsed < self.config.min_interval_seconds:
            logger.debug(
                "Notification rate-limited (%.1fs < %ds)",
                elapsed,
                self.config.min_interval_seconds,
            )
            return False

        for attempt in range(1, _MAX_SEND_ATTEMPTS + 1):
            try:
                conn = HTTPSConnection("api.telegram.org", timeout=10)
                path = f"/bot{self._bot_token}/sendMessage"
                payload: dict = {"chat_id": self._chat_id, "text": text}
                if parse_mode:
                    payload["parse_mode"] = parse_mode
                body = json.dumps(payload)
                headers = {"Content-Type": "application/json"}
                conn.request("POST", path, body, headers)
                resp = conn.getresponse()
                resp_body = resp.read().decode("utf-8")
                conn.close()

                if resp.status == 200:
                    self._last_send_time = now
                    logger.info("Telegram notification sent (%d chars)", len(text))
                    return True
                else:
                    logger.warning(
                        "Telegram API error %d: %s",
                        resp.status,
                        resp_body[:200],
                    )
                    return False

            except _RETRYABLE_ERRORS as e:
                if attempt < _MAX_SEND_ATTEMPTS:
                    logger.warning(
                        "Telegram send attempt %d/%d failed (network): %s — retrying in %.1fs",
                        attempt, _MAX_SEND_ATTEMPTS, e, _RETRY_DELAY,
                    )
                    time.sleep(_RETRY_DELAY)
                    continue
                logger.error(
                    "Telegram send failed after %d attempts: %s",
                    _MAX_SEND_ATTEMPTS, e,
                )
                return False
            except Exception as e:
                logger.error("Failed to send Telegram notification: %s", e)
                return False
        return False
