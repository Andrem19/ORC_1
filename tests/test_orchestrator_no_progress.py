"""Integration test: orchestrator handles no-progress scenarios."""

import json
from pathlib import Path
import tempfile

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.models import StopReason
from app.orchestrator import Orchestrator
from app.state_store import StateStore


def test_stops_on_max_empty_cycles():
    """Orchestrator stops after too many empty cycles where planner just says wait."""
    # With poll=0 and max_empty_cycles=3, we need the planner to return "wait"
    # enough times. The scheduler calls planner every cycle because there are
    # no active/pending tasks, so empty_cycles increments each cycle.
    # With 3 max empty cycles, after 3 "wait" decisions it stops.
    planner = FakePlanner(responses=[
        {"decision": "wait", "reason": "not ready yet", "check_after_seconds": 0},
        {"decision": "wait", "reason": "still not ready", "check_after_seconds": 0},
        {"decision": "wait", "reason": "not ready", "check_after_seconds": 0},
        {"decision": "wait", "reason": "not ready", "check_after_seconds": 0},
        {"decision": "wait", "reason": "not ready", "check_after_seconds": 0},
    ])

    worker = FakeWorker(responses=[])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(
        goal="Test",
        poll_interval_seconds=0,
        max_empty_cycles=3,
    )
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(
        config=config,
        state_store=store,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    reason = orch.run()
    assert reason == StopReason.NO_PROGRESS


def test_empty_cycles_reset_on_new_task():
    """Empty cycles counter resets when a new task is launched."""
    planner = FakePlanner(responses=[
        {"decision": "wait", "reason": "waiting", "check_after_seconds": 0},
        {"decision": "wait", "reason": "still waiting", "check_after_seconds": 0},
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Do work", "reason": "now", "check_after_seconds": 0},
        {"decision": "finish", "should_finish": True, "final_summary": "Done", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "success", "summary": "Done", "confidence": 0.9},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Test", poll_interval_seconds=0, max_empty_cycles=5)
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()
    assert reason == StopReason.GOAL_REACHED
    # Empty cycles should have been reset
    assert orch.state.empty_cycles == 0
