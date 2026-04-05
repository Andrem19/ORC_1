"""
Fake worker adapter for testing and demo.

Simulates worker results without calling any real CLI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from app.adapters.base import AdapterResponse, BaseAdapter

logger = logging.getLogger("orchestrator.adapter.fake_worker")


class FakeWorker(BaseAdapter):
    """Fake worker that returns scripted results for testing."""

    def __init__(
        self,
        responses: list[dict[str, Any]] | None = None,
        delay: float = 0.01,
        worker_id: str = "fake-worker-1",
    ) -> None:
        self.responses = responses or []
        self._call_index = 0
        self._delay = delay
        self.worker_id = worker_id
        self.call_log: list[str] = []

    def name(self) -> str:
        return f"fake_worker({self.worker_id})"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 60, **kwargs: Any) -> AdapterResponse:
        self.call_log.append(prompt)
        logger.info("FakeWorker %s called (call #%d)", self.worker_id, self._call_index + 1)
        time.sleep(self._delay)

        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
        else:
            # Default: success with generic summary
            resp = {
                "status": "success",
                "summary": f"Fake worker {self.worker_id} completed task.",
                "artifacts": [],
                "next_hint": "",
                "confidence": 0.9,
                "error": "",
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
