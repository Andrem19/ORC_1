"""
Planner runtime telemetry and watchdog helpers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import time
from typing import Any


@dataclass
class PlannerRunSnapshot:
    """Telemetry for one planner subprocess request."""

    request_type: str = "create"
    request_version: int = 0
    attempt_number: int = 0
    prompt_length: int = 0
    output_mode: str = "text"
    started_at_monotonic: float = field(default_factory=time.monotonic)
    started_at_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    first_output_at_monotonic: float | None = None
    first_output_at_iso: str | None = None
    last_output_at_monotonic: float | None = None
    last_output_at_iso: str | None = None
    output_bytes: int = 0
    stderr_bytes: int = 0
    raw_stdout: str = ""
    raw_stderr: str = ""
    rendered_output: str = ""
    stream_event_count: int = 0
    exit_code: int | None = None
    completed: bool = False
    termination_reason: str = ""
    slow_active_warning_sent: bool = False
    stalled_warning_sent: bool = False
    no_first_byte_warning_sent: bool = False
    slow_notification_sent: bool = False
    stalled_notification_sent: bool = False
    timeout_retry_count: int = 0

    @property
    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at_monotonic)

    @property
    def has_any_output(self) -> bool:
        return (self.output_bytes + self.stderr_bytes) > 0

    @property
    def has_first_byte(self) -> bool:
        return self.first_output_at_monotonic is not None

    def record_output(self, *, stdout_fragment: str = "", stderr_fragment: str = "", rendered_fragment: str = "") -> None:
        now_mono = time.monotonic()
        now_iso = datetime.now(timezone.utc).isoformat()

        if stdout_fragment:
            self.raw_stdout += stdout_fragment
            self.output_bytes += len(stdout_fragment.encode("utf-8", errors="replace"))
        if stderr_fragment:
            self.raw_stderr += stderr_fragment
            self.stderr_bytes += len(stderr_fragment.encode("utf-8", errors="replace"))
        if rendered_fragment:
            self.rendered_output += rendered_fragment

        if stdout_fragment or stderr_fragment:
            if self.first_output_at_monotonic is None:
                self.first_output_at_monotonic = now_mono
                self.first_output_at_iso = now_iso
            self.last_output_at_monotonic = now_mono
            self.last_output_at_iso = now_iso

    def finish(self, *, exit_code: int | None, rendered_output: str | None = None, termination_reason: str = "") -> None:
        self.exit_code = exit_code
        self.completed = True
        if rendered_output is not None:
            self.rendered_output = rendered_output
        if termination_reason:
            self.termination_reason = termination_reason

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["elapsed_seconds"] = round(self.elapsed_seconds, 3)
        data["raw_stream_transcript"] = data.pop("raw_stdout", "")
        data["rendered_output_clean"] = data.get("rendered_output", "")
        data["stderr"] = data.pop("raw_stderr", "")
        data["timing_summary"] = {
            "started_at": self.started_at_iso,
            "first_output_at": self.first_output_at_iso,
            "last_output_at": self.last_output_at_iso,
            "elapsed_seconds": data["elapsed_seconds"],
        }
        return data
