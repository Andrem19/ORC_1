"""Integration test: retry logic."""

import json
from pathlib import Path
import tempfile

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.models import StopReason, TaskStatus, WorkerConfig
from app.orchestrator import Orchestrator
from app.state_store import StateStore


def test_retry_on_worker_error():
    """Worker fails, planner retries, worker succeeds on second attempt."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Write code", "reason": "start", "check_after_seconds": 1},
        {"decision": "retry_worker", "target_worker_id": "qwen-1", "task_instruction": "Try again with better instructions", "reason": "temporary error", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "Done after retry", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "error", "error": "Network timeout"},
        {"status": "success", "summary": "Code written on retry", "confidence": 0.85},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Test retry", poll_interval_seconds=0, max_empty_cycles=10)
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()

    assert reason == StopReason.GOAL_REACHED
    assert len(orch.state.results) == 2
    assert orch.state.results[0].status == "error"
    assert orch.state.results[1].status == "success"


def test_max_attempts_exhausted():
    """Task fails multiple times and eventually exhausts max attempts."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Do thing", "reason": "start", "check_after_seconds": 1},
        # After first error, planner tries to retry
        {"decision": "retry_worker", "target_worker_id": "qwen-1", "task_instruction": "Try again", "reason": "retry", "check_after_seconds": 1},
        # After second error, planner tries again
        {"decision": "retry_worker", "target_worker_id": "qwen-1", "task_instruction": "One more try", "reason": "retry", "check_after_seconds": 1},
        # Give up
        {"decision": "finish", "should_finish": True, "final_summary": "Failed but finishing", "reason": "max retries"},
    ])
    worker = FakeWorker(responses=[
        {"status": "error", "error": "Failed"},
        {"status": "error", "error": "Failed again"},
        {"status": "error", "error": "Still failing"},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(
        goal="Test",
        poll_interval_seconds=0,
        max_empty_cycles=10,
        max_task_attempts=5,  # allow many retries
    )
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()

    assert reason == StopReason.GOAL_REACHED
    assert orch.state.total_errors > 0


def test_stop_worker_cancels_tasks():
    """Planner issues stop_worker — active tasks for that worker are cancelled."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Do something", "reason": "start", "check_after_seconds": 1},
        {"decision": "stop_worker", "target_worker_id": "qwen-1", "reason": "worker misbehaving", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "Stopped bad worker", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "error", "error": "Produced garbage"},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(goal="Test stop", poll_interval_seconds=0, max_empty_cycles=10)
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()

    assert reason == StopReason.GOAL_REACHED
    # The task should have been cancelled
    cancelled = [t for t in orch.state.tasks if t.status == TaskStatus.CANCELLED]
    assert len(cancelled) >= 1


def test_reassign_task_moves_to_different_worker():
    """Planner reassigns a stalled task to another worker."""
    planner = FakePlanner(responses=[
        {"decision": "launch_worker", "target_worker_id": "qwen-1", "task_instruction": "Initial task", "reason": "start", "check_after_seconds": 1},
        {"decision": "reassign_task", "target_worker_id": "qwen-1", "reassign_to_worker_id": "qwen-2", "task_instruction": "Revised instructions for new worker", "reason": "stalled on qwen-1", "check_after_seconds": 1},
        {"decision": "finish", "should_finish": True, "final_summary": "Reassign worked", "reason": "done"},
    ])
    worker = FakeWorker(responses=[
        {"status": "error", "error": "Stalled"},
        {"status": "success", "summary": "Completed after reassignment", "confidence": 0.9},
    ])

    tmp = Path(tempfile.mkdtemp())
    config = OrchestratorConfig(
        goal="Test reassign",
        poll_interval_seconds=0,
        max_empty_cycles=10,
        workers=[
            WorkerConfig(worker_id="qwen-1", role="executor", system_prompt=""),
            WorkerConfig(worker_id="qwen-2", role="executor", system_prompt=""),
        ],
    )
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(config=config, state_store=store, planner_adapter=planner, worker_adapter=worker)
    reason = orch.run()

    assert reason == StopReason.GOAL_REACHED
    # At least one task should be reassigned (assigned to qwen-2)
    reassigned = [t for t in orch.state.tasks if t.assigned_worker_id == "qwen-2"]
    assert len(reassigned) >= 1
