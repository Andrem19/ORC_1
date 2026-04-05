"""Integration test: orchestrator happy path with fake adapters."""

import json
from pathlib import Path
import tempfile

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.models import StopReason, TaskStatus
from app.orchestrator import Orchestrator
from app.state_store import StateStore


def test_happy_path_single_task():
    """Plan -> launch worker -> get result -> finish."""
    planner = FakePlanner(responses=[
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Write hello.py",
            "reason": "Start",
            "check_after_seconds": 1,
        },
        {
            "decision": "finish",
            "should_finish": True,
            "final_summary": "All done",
            "reason": "Task completed",
        },
    ])
    worker = FakeWorker(responses=[
        {"status": "success", "summary": "Created hello.py", "confidence": 0.95},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(
        goal="Write hello.py",
        poll_interval_seconds=0,
        max_empty_cycles=10,
    )
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(
        config=config,
        state_store=store,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    reason = orch.run()
    assert reason == StopReason.GOAL_REACHED
    assert orch.state.status == "finished"
    assert len(orch.state.completed_tasks()) == 1


def test_happy_path_multiple_tasks():
    """Multiple tasks in sequence."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Task 1", "reason": "start", "check_after_seconds": 1},
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Task 2", "reason": "continue", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "All tasks done", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "success", "summary": "Task 1 done", "confidence": 0.9},
        {"status": "success", "summary": "Task 2 done", "confidence": 0.85},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Multi-task", poll_interval_seconds=0, max_empty_cycles=10)
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()

    assert reason == StopReason.GOAL_REACHED
    assert len(orch.state.completed_tasks()) == 2
    assert len(orch.state.results) == 2


def test_planner_receives_new_results():
    """Verify planner prompt contains new result information."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Task A", "reason": "start", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "Done", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "success", "summary": "Result A", "confidence": 0.9},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Test", poll_interval_seconds=0, max_empty_cycles=10)
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    orch.run()

    # The second planner call should contain result information
    assert len(planner.call_log) == 2
    second_prompt = planner.call_log[1]
    assert "Result A" in second_prompt or "New Results" in second_prompt
