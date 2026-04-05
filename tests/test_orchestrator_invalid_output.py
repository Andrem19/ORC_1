"""Integration test: handling invalid model output."""

import json
from pathlib import Path
import tempfile

from app.adapters.base import AdapterResponse, BaseAdapter
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.models import StopReason, PlannerDecision
from app.orchestrator import Orchestrator
from app.state_store import StateStore


class BrokenPlanner(BaseAdapter):
    """Returns garbage that can't be parsed."""

    def name(self) -> str:
        return "broken_planner"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 60, **kwargs) -> AdapterResponse:
        return AdapterResponse(
            success=True,
            raw_output="I think we should probably wait a bit more and then maybe do something.",
            exit_code=0,
        )


class EmptyOutputPlanner(BaseAdapter):
    """Returns empty output."""

    def name(self) -> str:
        return "empty_planner"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 60, **kwargs) -> AdapterResponse:
        return AdapterResponse(success=True, raw_output="", exit_code=0)


class FailingPlanner(BaseAdapter):
    """Returns failure response."""

    def name(self) -> str:
        return "failing_planner"

    def is_available(self) -> bool:
        return True

    def invoke(self, prompt: str, timeout: int = 60, **kwargs) -> AdapterResponse:
        return AdapterResponse(
            success=False,
            raw_output="",
            exit_code=1,
            error="CLI crashed with segfault",
        )


def _make_orch(planner: BaseAdapter) -> tuple[Orchestrator, Path]:
    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Test", poll_interval_seconds=0, max_empty_cycles=5)
    store = StateStore(tmp / "state.json")
    worker = FakeWorker(responses=[])
    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    return orch, tmp


def test_invalid_json_defaults_to_wait():
    """Non-JSON planner output defaults to 'wait' decision."""
    orch, tmp = _make_orch(BrokenPlanner())
    reason = orch.run()
    # Should hit max empty cycles since planner always says "wait" (default for unparseable)
    assert reason == StopReason.NO_PROGRESS
    assert orch.state.last_planner_decision == PlannerDecision.WAIT


def test_empty_output_handled():
    """Empty planner output is handled gracefully."""
    orch, tmp = _make_orch(EmptyOutputPlanner())
    reason = orch.run()
    assert reason == StopReason.NO_PROGRESS


def test_failing_planner_handled():
    """Planner adapter failure is handled without crash."""
    orch, tmp = _make_orch(FailingPlanner())
    reason = orch.run()
    assert reason == StopReason.NO_PROGRESS
    assert orch.state.total_errors > 0 or orch.state.empty_cycles >= 5
