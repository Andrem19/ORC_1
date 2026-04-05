"""
Notification service — sends Telegram messages via Bot API.

Uses http.client (stdlib) to call the Telegram Bot API directly.
No external dependencies required.

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

logger = logging.getLogger("orchestrator.notifications")


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
        if self._enabled:
            logger.info(
                "Notifications enabled (chat_id=%s, min_interval=%ds)",
                self._chat_id,
                self.config.min_interval_seconds,
            )

    def is_configured(self) -> bool:
        """Check if Telegram credentials are present."""
        return bool(self._bot_token) and bool(self._chat_id)

    # ---------------------------------------------------------------
    # Public notification methods
    # ---------------------------------------------------------------

    def send_lifecycle(self, event: str, detail: str = "") -> bool:
        """Send a lifecycle notification (start/stop/finish)."""
        if not self._enabled:
            return False
        text = f"Orchestrator {event}"
        if detail:
            text += f"\n{detail}"
        return self._send(text)

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
        lines = [
            f"{icon} Cycle #{cycle} | Worker {result.worker_id} {label}",
            f"Task {result.task_id}",
        ]
        if result.summary:
            lines.append(result.summary[:300])
        if result.error:
            lines.append(f"Error: {result.error[:200]}")
        if result.confidence > 0:
            lines.append(f"Confidence: {result.confidence:.2f}")
        return self._send("\n".join(lines))

    def send_planner_decision(self, output: PlannerOutput, cycle: int = 0) -> bool:
        """Send a planner decision notification."""
        if not self._enabled:
            return False
        lines = [
            f"\U0001f9d1\u200d\U0001f4bc Cycle #{cycle} | Planner decision",
            f"{output.decision.value}",
        ]
        if output.reason:
            lines.append(output.reason[:200])
        if output.target_worker_id:
            lines.append(f"Worker: {output.target_worker_id}")
        if output.task_instruction:
            lines.append(f"Task: {output.task_instruction[:150]}")
        if output.memory_update:
            lines.append(f"Memory: {output.memory_update[:100]}")
        return self._send("\n".join(lines))

    def send_error(self, error: str, context: str = "") -> bool:
        """Send an error alert."""
        if not self._enabled:
            return False
        text = f"\u26a0\ufe0f Error"
        if context:
            text += f" ({context})"
        text += f"\n{error[:300]}"
        return self._send(text)

    def send_research_summary(self, state: OrchestratorState) -> bool:
        """Send a periodic research summary."""
        if not self._enabled:
            return False
        completed = len(state.completed_tasks())
        failed = len(state.failed_tasks())
        lines = [
            "\U0001f4ca Research Summary",
            f"Cycle: {state.current_cycle}",
            f"Tasks: {completed} completed, {failed} failed",
            f"Errors: {state.total_errors}",
        ]
        if state.last_planner_decision:
            lines.append(f"Last decision: {state.last_planner_decision.value}")
        return self._send("\n".join(lines))

    # ---------------------------------------------------------------
    # Internal — Telegram API call
    # ---------------------------------------------------------------

    def _send(self, text: str) -> bool:
        """Send a message to Telegram. Returns True if sent."""
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

        try:
            conn = HTTPSConnection("api.telegram.org", timeout=10)
            path = f"/bot{self._bot_token}/sendMessage"
            body = json.dumps({
                "chat_id": self._chat_id,
                "text": text[:4096],  # Telegram message limit
            })
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

        except Exception as e:
            logger.error("Failed to send Telegram notification: %s", e)
            return False
