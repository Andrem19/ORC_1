"""
Fake planner adapter for testing and demo.

Simulates planner decisions without calling any real CLI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter

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

    def reset(self) -> None:
        self._call_index = 0
        self.call_log.clear()
