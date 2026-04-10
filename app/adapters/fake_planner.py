"""
Fake planner adapter for testing and demo.

Simulates markdown planner responses without calling any real CLI.
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
        responses: list[Any] | None = None,
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
            resp = {
                "plan_id": "plan_demo",
                "goal": "Default fake planner output",
                "baseline_ref": {
                    "snapshot_id": "active-signal-v1",
                    "version": 1,
                    "symbol": "BTCUSDT",
                    "anchor_timeframe": "1h",
                    "execution_timeframe": "5m",
                },
                "global_constraints": [
                    "keep baseline fixed",
                    "use cheap validation first",
                ],
                "slices": [
                    {
                        "slice_id": "slice_demo",
                        "title": "Demo validation",
                        "hypothesis": "default fake slice can be completed",
                        "objective": "produce one direct action flow",
                        "success_criteria": ["one final report emitted"],
                        "allowed_tools": ["system_health"],
                        "evidence_requirements": ["one structured conclusion"],
                        "policy_tags": ["cheap_first"],
                        "max_turns": 2,
                        "max_tool_calls": 1,
                        "max_expensive_calls": 0,
                        "parallel_slot": 1,
                    }
                ],
            }

        output = resp if isinstance(resp, str) else json.dumps(resp)
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
                "plan_id": "plan_demo",
                "goal": "Default fake planner output",
                "baseline_ref": {
                    "snapshot_id": "active-signal-v1",
                    "version": 1,
                    "symbol": "BTCUSDT",
                    "anchor_timeframe": "1h",
                    "execution_timeframe": "5m",
                },
                "global_constraints": [
                    "keep baseline fixed",
                    "use cheap validation first",
                ],
                "slices": [
                    {
                        "slice_id": "slice_demo",
                        "title": "Demo validation",
                        "hypothesis": "default fake slice can be completed",
                        "objective": "produce one direct action flow",
                        "success_criteria": ["one final report emitted"],
                        "allowed_tools": ["system_health"],
                        "evidence_requirements": ["one structured conclusion"],
                        "policy_tags": ["cheap_first"],
                        "max_turns": 2,
                        "max_tool_calls": 1,
                        "max_expensive_calls": 0,
                        "parallel_slot": 1,
                    }
                ],
            }

        self._pending_response = resp if isinstance(resp, str) else json.dumps(resp)
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
