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

from app.config import LMStudioConfig, NotificationConfig
from app.models import OrchestratorState, TaskResult
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

# ---------------------------------------------------------------------------
# Verdict / stop-reason icons
# ---------------------------------------------------------------------------

_VERDICT_ICON: dict[str, str] = {
    "PROMOTE": "\U0001f3c6",    # 🏆
    "WATCHLIST": "\U0001f441",  # 👁
    "REJECT": "\u274c",         # ❌
    "FAILED": "\u274c",         # ❌
    "PENDING": "\u23f3",        # ⏳
}

_STOP_REASON_ICON: dict[str, str] = {
    "goal_reached": "\u2705",        # ✅
    "graceful_stop": "\U0001f6d1",   # 🛑
    "no_progress": "\u26a0\ufe0f",   # ⚠️
    "max_errors": "\U0001f525",      # 🔥
    "goal_impossible": "\U0001f6ab", # 🚫
    "mcp_unhealthy": "\U0001f50c",   # 🔌
    "subprocess_error": "\U0001f4a5",# 💥
}


def _verdict_icon(verdict: str) -> str:
    return _VERDICT_ICON.get(verdict.upper() if verdict else "", "\u2139\ufe0f")  # ℹ️


def _stop_icon(reason: str) -> str:
    return _STOP_REASON_ICON.get(reason, "\u2139\ufe0f")  # ℹ️


def _ms_to_human(ms: int) -> str:
    """Convert milliseconds to human-readable duration string."""
    if ms <= 0:
        return "—"
    total_s = ms // 1000
    if total_s < 60:
        return f"{total_s}s"
    m, s = divmod(total_s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def _short_task_label(result: TaskResult) -> str:
    """Return human-readable task label: title if set, else last part of task_id."""
    if result.title:
        return result.title
    # task_id format: "plan_xxx:slice_yyy" — show slice part only
    parts = result.task_id.split(":")
    return parts[-1] if len(parts) > 1 else result.task_id


def _metrics_line(key_metrics: dict) -> str:
    """Compact one-line representation of key metrics."""
    if not key_metrics:
        return ""
    parts = []
    for k, v in list(key_metrics.items())[:5]:
        if isinstance(v, float):
            parts.append(f"{k}={v:.2f}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)


@dataclass
class _QueuedNotification:
    result: TaskResult
    cycle: int
    queued_at: float


class NotificationService:
    """Sends Telegram notifications for orchestrator events."""

    def __init__(
        self,
        config: NotificationConfig | None = None,
        lmstudio_config: LMStudioConfig | None = None,
    ) -> None:
        self.config = config or NotificationConfig()
        self._bot_token: str = os.environ.get("ALGO_BOT", "")
        self._chat_id: str = os.environ.get("CHAT_ID", "")
        self._last_send_time: float = 0.0
        self._enabled = (
            self.config.enabled
            and bool(self._bot_token)
            and bool(self._chat_id)
        )
        # Resolve LM Studio translation params from shared config
        lm = lmstudio_config
        self._translator = TranslationService(
            translate_to_russian=self.config.translate_to_russian,
            model_dir=self.config.translation_model_dir,
            model_name=self.config.translation_model_name,
            backend=self.config.translation_backend,
            lmstudio_base_url=lm.base_url if lm else "http://localhost:1234",
            lmstudio_model=lm.model if lm else "",
            lmstudio_max_tokens=lm.translation.max_tokens if lm else 1024,
            lmstudio_timeout_seconds=lm.translation.timeout_seconds if lm else 30,
            lmstudio_reasoning_effort=lm.reasoning_effort if lm else "",
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

    def send_run_complete(self, report: object) -> bool:
        """Send a rich structured run-complete message from a RunSummaryReport."""
        if not self._enabled:
            return False
        return self._send_run_complete_report(report)

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

        # Verdict distribution
        promote = sum(1 for i in items if i.result.verdict == "PROMOTE")
        watchlist = sum(1 for i in items if i.result.verdict == "WATCHLIST")
        reject = sum(1 for i in items if i.result.verdict in ("REJECT", "FAILED"))

        builder = TelegramMessageBuilder()
        # Header: "📋 Cycle #5 | v4 B2 | 3 workers completed: 2 OK, 1 failed"
        seq_label = items[0].result.sequence_label
        header_parts = [f"Cycle #{cycle}"]
        if seq_label:
            header_parts.append(seq_label)
        header_parts.append(f"{len(items)} workers: {success_count} OK, {fail_count} fail")
        builder.add_header(
            "\U0001f4cb",
            " | ".join(header_parts),
        )
        builder.add_separator()

        # Verdict distribution line (only if any verdicts set)
        if promote + watchlist + reject > 0:
            dist = f"\U0001f3c6 {promote} promoted  \U0001f441 {watchlist} watchlist  \u274c {reject} rejected"
            builder.add_body(dist)
            builder.add_separator()

        for item in items:
            r = item.result
            icon = "\u2705" if r.status == "success" else "\u274c"
            label = _short_task_label(r)
            line = f"{icon} {label}"
            if r.summary:
                line += f" \u2014 {r.summary[:80]}"
            if r.verdict:
                line += f" ({r.verdict})"
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

        task_label = _short_task_label(result)
        verdict_badge = _verdict_icon(result.verdict) if result.verdict else ""
        # Header: "Cycle #5 | v4 B2 | Validate OHLCV 🏆"
        parts = [f"Cycle #{cycle}"]
        if result.sequence_label:
            parts.append(result.sequence_label)
        parts.append(task_label)
        header = " | ".join(parts)
        if verdict_badge:
            header += f" {verdict_badge}"

        builder = TelegramMessageBuilder()
        builder.add_header(icon, header)
        builder.add_separator()

        if result.summary:
            builder.add_body(result.summary[:250])

        if result.confidence > 0:
            builder.add_field("Confidence", f"{result.confidence:.2f}")

        if result.error:
            builder.add_field("Error", result.error[:200])

        if result.findings:
            for finding in result.findings[:3]:
                builder.add_body(f"\u2022 {finding[:120]}")

        metrics_line = _metrics_line(result.key_metrics)
        if metrics_line:
            builder.add_code_block(metrics_line)

        if result.next_actions:
            for action in result.next_actions[:2]:
                builder.add_body(f"\u2192 {action[:100]}")

        return self._send_structured(builder.build())

    # ---------------------------------------------------------------
    # Rich run-complete message
    # ---------------------------------------------------------------

    def _send_run_complete_report(self, report: object) -> bool:
        """Build and send a rich run-complete Telegram message from RunSummaryReport."""
        # Use getattr so this works even if RunSummaryReport type is not directly imported
        stop_reason = str(getattr(report, "stop_reason", "") or "")
        goal = str(getattr(report, "goal", "") or "")
        duration_ms = int(getattr(report, "duration_ms", 0) or 0)
        completed_seq = int(getattr(report, "completed_sequences", 0))
        failed_seq = int(getattr(report, "failed_sequences", 0))
        partial_seq = int(getattr(report, "partial_sequences", 0))
        best_outcomes: list[str] = list(getattr(report, "best_outcomes", []) or [])
        unresolved_blockers: list[str] = list(getattr(report, "unresolved_blockers", []) or [])
        executive_summary = str(getattr(report, "executive_summary_ru", "") or "")
        narrative = getattr(report, "narrative_sections_ru", None)
        next_actions: list[str] = list(getattr(narrative, "recommended_next_actions_ru", []) or []) if narrative else []
        direct_metrics = getattr(report, "direct_metrics", None)
        tool_calls = int(getattr(direct_metrics, "direct_tool_calls_observed", 0) or 0) if direct_metrics else 0
        tool_failed = int(getattr(direct_metrics, "direct_failed", 0) or 0) if direct_metrics else 0

        icon = _stop_icon(stop_reason)
        builder = TelegramMessageBuilder()
        builder.add_header(icon, f"Run complete \u2014 {stop_reason}")
        builder.add_separator()

        if goal:
            builder.add_field("Goal", goal[:120])
        if duration_ms:
            builder.add_field("Duration", _ms_to_human(duration_ms))
        seq_parts = [f"{completed_seq} completed"]
        if failed_seq:
            seq_parts.append(f"{failed_seq} failed")
        if partial_seq:
            seq_parts.append(f"{partial_seq} partial")
        builder.add_field("Sequences", ", ".join(seq_parts))

        if executive_summary:
            builder.add_separator()
            builder.add_header("\U0001f4cb", "Executive summary")
            builder.add_body(executive_summary[:400])

        if best_outcomes:
            builder.add_separator()
            builder.add_header("\U0001f3c6", "Best outcomes")
            for outcome in best_outcomes[:4]:
                builder.add_body(f"\u2022 {outcome[:100]}")

        if next_actions:
            builder.add_separator()
            builder.add_header("\U0001f51c", "Next actions")
            for action in next_actions[:3]:
                builder.add_body(f"\u2022 {action[:100]}")

        if unresolved_blockers:
            builder.add_separator()
            builder.add_header("\u26a0\ufe0f", f"Blockers ({len(unresolved_blockers)})")
            for blocker in unresolved_blockers[:3]:
                builder.add_body(f"\u2022 {blocker[:100]}")

        if tool_calls > 0:
            fail_pct = f"{tool_failed / tool_calls * 100:.1f}%" if tool_calls else "0%"
            builder.add_separator()
            tool_line = f"\U0001f527 Direct tool usage: {tool_calls} calls, {tool_failed} failed ({fail_pct})"
            builder.add_body(tool_line)

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
