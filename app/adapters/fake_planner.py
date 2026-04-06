"""
Fake planner adapter for testing and demo.

Simulates planner decisions without calling any real CLI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter, ProcessHandle

logger = logging.getLogger("orchestrator.adapter.fake_planner")


class FakePlanner(BaseAdapter):
    """Fake planner that returns scripted decisions for testing."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        delay: float = 0.01,
    ) -> None:
        self.responses = responses or []
        self._call_index = 0
        self._delay = delay
        self.call_log: list[str] = []  # record of all prompts received
        self._pending_response: str = ""

    def name(self) -> str:
        return "fake_planner"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 60, **kwargs: Any) -> AdapterResponse:
        self.call_log.append(prompt)
        logger.info("FakePlanner called (call #%d)", self._call_index + 1)
        time.sleep(self._delay)

        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
        else:
            # Default: finish
            resp = {
                "decision": "finish",
                "reason": "No more scripted responses",
                "should_finish": True,
                "final_summary": "Fake planner finished all scripted responses.",
            }

        output = json.dumps(resp)
        return AdapterResponse(
            success=True,
            raw_output=output,
            exit_code=0,
            duration_seconds=self._delay,
        )

    def start(self, prompt: str, **kwargs: Any) -> ProcessHandle:
        """Fake start: store response, return handle that finishes instantly."""
        self.call_log.append(prompt)
        logger.info("FakePlanner start (call #%d)", self._call_index + 1)

        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
        else:
            resp = {
                "decision": "finish",
                "reason": "No more scripted responses",
                "should_finish": True,
                "final_summary": "Fake planner finished all scripted responses.",
            }

        self._pending_response = json.dumps(resp)
        return ProcessHandle(
            process=None,
            task_id=kwargs.get("task_id", "planner"),
            worker_id="planner",
            started_at=time.monotonic(),
        )

    def check(self, handle: ProcessHandle) -> tuple[str, bool]:
        """Immediately return stored response (simulates instant completion)."""
        if self._pending_response:
            output = self._pending_response
            self._pending_response = ""
            handle.partial_output = output
            return output, True
        return "", True

    def reset(self) -> None:
        self._call_index = 0
        self.call_log.clear()
        self._pending_response = ""
