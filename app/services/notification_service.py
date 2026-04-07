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
import threading
import time
from dataclasses import dataclass
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


@dataclass
class _QueuedNotification:
    result: TaskResult
    cycle: int
    queued_at: float


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
            backend=self.config.translation_backend,
            lmstudio_base_url=self.config.translation_lmstudio_base_url,
            lmstudio_model=self.config.translation_lmstudio_model,
            lmstudio_max_tokens=self.config.translation_lmstudio_max_tokens,
            lmstudio_timeout_seconds=self.config.translation_lmstudio_timeout_seconds,
        )
        # Batch notification state
        self._batch_queue: list[_QueuedNotification] = []
        self._batch_timer: threading.Timer | None = None
        self._batch_lock: threading.Lock = threading.Lock()
        if self._enabled:
            logger.info(
                "Notifications enabled (chat_id=%s, min_interval=%ds, batch=%s)",
                self._chat_id,
                self.config.min_interval_seconds,
                self.config.batch_enabled,
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
        """Send a worker result notification (or queue for batch)."""
        if not self._enabled:
            return False
        if self.config.batch_enabled:
            return self._queue_for_batch(result, cycle)
        return self._send_single_worker_result(result, cycle)

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

    def send_diagnostic_digest(self, digest: str, cycle: int = 0) -> bool:
        """Send an LM Studio diagnostic digest notification."""
        if not self._enabled:
            return False
        title = "Diagnostic Digest"
        if cycle:
            title += f" (cycle #{cycle})"
        builder = TelegramMessageBuilder()
        builder.add_header("\U0001f4ca", title)
        builder.add_separator()
        builder.add_body(digest[:3000])
        return self._send_structured(builder.build())

    def send_execution_prediction(
        self, prediction_text: str, plan_version: int = 0,
    ) -> bool:
        """Send an execution time prediction notification."""
        if not self._enabled:
            return False
        title = "Execution Prediction"
        if plan_version:
            title += f" (plan v{plan_version})"
        builder = TelegramMessageBuilder()
        builder.add_header("\u23f1", title)
        builder.add_separator()
        builder.add_body(prediction_text[:2000])
        return self._send_structured(builder.build())

    # ---------------------------------------------------------------
    # Batch notification system
    # ---------------------------------------------------------------

    def _queue_for_batch(self, result: TaskResult, cycle: int) -> bool:
        """Queue a worker result for batch notification."""
        with self._batch_lock:
            self._batch_queue.append(_QueuedNotification(
                result=result, cycle=cycle, queued_at=time.monotonic(),
            ))
            # Reset debounce timer
            if self._batch_timer is not None:
                self._batch_timer.cancel()
            self._batch_timer = threading.Timer(
                self.config.batch_debounce_seconds,
                self._flush_batch,
            )
            self._batch_timer.daemon = True
            self._batch_timer.start()
        return True

    def _flush_batch(self) -> None:
        """Send all queued worker results as one batch notification."""
        with self._batch_lock:
            if not self._batch_queue:
                return
            items = list(self._batch_queue)
            self._batch_queue.clear()
            self._batch_timer = None

        if len(items) == 1:
            item = items[0]
            self._send_single_worker_result(item.result, item.cycle)
        else:
            self._send_batch_summary(items)

    def _send_batch_summary(self, items: list[_QueuedNotification]) -> bool:
        """Send a batched summary of multiple worker results."""
        success_count = sum(1 for i in items if i.result.status == "success")
        fail_count = len(items) - success_count
        cycle = items[0].cycle

        builder = TelegramMessageBuilder()
        builder.add_header(
            "\U0001f4cb",
            f"Cycle #{cycle} | {len(items)} workers completed: "
            f"{success_count} OK, {fail_count} failed",
        )
        builder.add_separator()

        for item in items:
            r = item.result
            icon = "\u2705" if r.status == "success" else "\u274c"
            line = f"{icon} {r.worker_id}: {r.task_id}"
            if r.summary:
                line += f" \u2014 {r.summary[:80]}"
            if r.error:
                line += f" [ERR: {r.error[:60]}]"
            builder.add_body(line)

        # Bypass rate limit for batch sends
        self._last_send_time = 0.0
        return self._send_structured(builder.build())

    def flush(self) -> None:
        """Flush any pending batched notifications. Call on shutdown."""
        with self._batch_lock:
            if self._batch_timer is not None:
                self._batch_timer.cancel()
        self._flush_batch()

    # ---------------------------------------------------------------
    # Single worker result (used when batch is disabled or single item)
    # ---------------------------------------------------------------

    def _send_single_worker_result(self, result: TaskResult, cycle: int) -> bool:
        """Send a single worker result notification."""
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
