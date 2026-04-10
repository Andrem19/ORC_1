from __future__ import annotations

import asyncio

from app.adapters.base import AdapterResponse, BaseAdapter
from app.services.direct_execution.invocation import invoke_adapter_with_retries


class _TimeoutThenSuccessAdapter(BaseAdapter):
    def __init__(self) -> None:
        self.timeouts: list[int] = []
        self.calls = 0

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):
        del prompt, kwargs
        self.calls += 1
        self.timeouts.append(timeout)
        if self.calls == 1:
            return AdapterResponse(success=False, raw_output="", error="Timed out after test", timed_out=True)
        return AdapterResponse(success=True, raw_output='{"ok": true}')

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "timeout-then-success"


def test_invoke_adapter_with_retries_reuses_timeout_budget() -> None:
    adapter = _TimeoutThenSuccessAdapter()

    response = asyncio.run(
        invoke_adapter_with_retries(
            adapter=adapter,
            prompt="test",
            timeout_seconds=17,
            max_attempts=3,
            base_backoff_seconds=0.001,
        )
    )

    assert response.success is True
    assert adapter.timeouts == [17, 17]

