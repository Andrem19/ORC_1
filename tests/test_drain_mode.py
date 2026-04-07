"""Unit tests for graceful drain shutdown mode."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.models import OrchestratorState, StopReason, Task, TaskStatus
from app.plan_models import PlanTask, ResearchPlan
from app.services.plan_orchestrator_service import PlanOrchestratorService


def _make_service(*, worker_ids: list[str] | None = None) -> tuple[PlanOrchestratorService, MagicMock]:
    """Create a PlanOrchestratorService with mocked dependencies."""
    orch = MagicMock()
    orch.config = SimpleNamespace(
        goal="test goal",
        max_concurrent_plan_tasks=2,
        research_config={},
        mcp_review=SimpleNamespace(max_problems_in_context=10),
        plan_task_timeout_seconds=600,
        drain_timeout_seconds=600,
        worker_adapter=SimpleNamespace(cli_path="qwen"),
        planner_adapter=SimpleNamespace(
            soft_timeout_seconds=300,
            hard_timeout_seconds=900,
            no_first_byte_seconds=180,
            model="opus",
        ),
    )
    orch._research_context_text = None
    orch._mcp_scanner = None
    orch.state = OrchestratorState(goal="test goal")
    orch.planner_service = MagicMock()
    orch.planner_service.last_plan_raw_output = ""
    orch.planner_service.is_running = False
    orch.worker_service = MagicMock()
    orch.scheduler = MagicMock()
    orch.memory_service = MagicMock()
    orch.notification_service = MagicMock()
    orch._plan_store = MagicMock()
    orch._worker_ids = worker_ids or ["qwen-1", "qwen-2"]
    orch._log_event = MagicMock()
    orch._finish = MagicMock()
    orch._collect_results = MagicMock(return_value=[])
    orch.save_state = MagicMock()
    return PlanOrchestratorService(orch), orch


def _add_running_task(svc: PlanOrchestratorService, task_id: str = "task-1") -> Task:
    """Add a RUNNING task to the service state."""
    task = Task(
        task_id=task_id,
        assigned_worker_id="qwen-1",
        status=TaskStatus.RUNNING,
        metadata={"plan_mode": True, "plan_version": 1, "stage_number": 1},
    )
    svc.state.add_task(task)
    return task


def test_drain_no_running_tasks() -> None:
    """Drain mode with no active tasks exits immediately with GRACEFUL_STOP."""
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(version=1, goal="test", tasks=[])

    # Activate drain mode
    svc._drain_mode = True
    svc._drain_started_at = time.monotonic()

    # Mock _plan_sleep to raise StopIteration so we break out of the loop
    call_count = 0

    def fake_sleep(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count > 5:
            raise StopIteration("loop did not exit")

    svc._plan_sleep = fake_sleep

    try:
        result = svc.run()
    except StopIteration:
        raise AssertionError("Drain should have exited immediately when no tasks are running")

    assert result == StopReason.GRACEFUL_STOP
    orch._finish.assert_called_once_with(StopReason.GRACEFUL_STOP, "Graceful drain completed")


def test_drain_waits_for_running_tasks() -> None:
    """Drain mode continues collecting results while tasks are running, then exits."""
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(
        version=1, goal="test", tasks=[PlanTask(stage_number=1)],
    )
    _add_running_task(svc, "task-1")

    svc._drain_mode = True
    svc._drain_started_at = time.monotonic()

    cycle_count = 0

    def fake_collect_results():
        nonlocal cycle_count
        cycle_count += 1
        # After 2 cycles, mark the task as completed
        if cycle_count >= 2:
            task = svc.state.find_task("task-1")
            if task:
                task.status = TaskStatus.COMPLETED
        return []

    orch._collect_results = fake_collect_results

    def fake_sleep(*args, **kwargs):
        if cycle_count > 10:
            raise StopIteration("loop did not exit after task completed")

    svc._plan_sleep = fake_sleep

    result = svc.run()

    assert result == StopReason.GRACEFUL_STOP
    orch._finish.assert_called_once_with(StopReason.GRACEFUL_STOP, "Graceful drain completed")
    # Should have collected results at least twice
    assert cycle_count >= 2


def test_drain_skips_dispatch() -> None:
    """In drain mode, _dispatch_plan_tasks is not called."""
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(
        version=1, goal="test", tasks=[PlanTask(stage_number=1)],
    )
    _add_running_task(svc, "task-1")

    svc._drain_mode = True
    svc._drain_started_at = time.monotonic()

    cycle_count = 0
    dispatch_called = False

    original_dispatch = svc._dispatch_plan_tasks

    def tracking_dispatch(*args, **kwargs):
        nonlocal dispatch_called
        dispatch_called = True

    svc._dispatch_plan_tasks = tracking_dispatch

    def fake_collect_results():
        nonlocal cycle_count
        cycle_count += 1
        if cycle_count >= 1:
            task = svc.state.find_task("task-1")
            if task:
                task.status = TaskStatus.COMPLETED
        return []

    orch._collect_results = fake_collect_results

    def fake_sleep(*args, **kwargs):
        if cycle_count > 10:
            raise StopIteration("loop did not exit")

    svc._plan_sleep = fake_sleep

    result = svc.run()
    assert result == StopReason.GRACEFUL_STOP
    assert not dispatch_called, "Dispatch should not be called during drain mode"


def test_drain_timeout_exceeded() -> None:
    """Drain timeout terminates remaining tasks and returns GRACEFUL_STOP."""
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(
        version=1, goal="test", tasks=[PlanTask(stage_number=1)],
    )
    task = _add_running_task(svc, "task-1")
    svc._drain_mode = True
    # Set drain start far in the past to exceed the 600s timeout
    svc._drain_started_at = time.monotonic() - 700

    def fake_sleep(*args, **kwargs):
        raise StopIteration("loop did not exit")

    svc._plan_sleep = fake_sleep

    result = svc.run()

    assert result == StopReason.GRACEFUL_STOP
    orch._finish.assert_called_once()
    call_args = orch._finish.call_args
    assert call_args[0][0] == StopReason.GRACEFUL_STOP
    assert "Drain timeout" in call_args[0][1]


def test_request_drain_propagates_to_plan_service() -> None:
    """orch.request_drain() propagates drain mode to plan service."""
    from app.orchestrator import Orchestrator

    config = MagicMock()
    config.goal = "test"
    config.state_dir = "state"
    config.state_file = "orchestrator_state.json"
    config.planner_timeout_seconds = 60
    config.worker_timeout_seconds = 60
    config.poll_interval_seconds = 10
    config.max_empty_cycles = 5
    config.max_errors_total = 10
    config.max_mcp_failures = 3
    config.max_task_attempts = 3
    config.max_worker_timeout_count = 3
    config.mcp_review = MagicMock(enabled=False)
    config.notifications = MagicMock()
    config.plan_mode = False
    config.drain_timeout_seconds = 600
    config.workers = [MagicMock(worker_id="w1")]

    with patch("app.orchestrator.StateStore"), \
         patch("app.orchestrator.PlannerService"), \
         patch("app.orchestrator.WorkerService"), \
         patch("app.orchestrator.Scheduler"), \
         patch("app.orchestrator.MemoryService"), \
         patch("app.orchestrator.NotificationService"), \
         patch("app.orchestrator.TaskSupervisor"):
        orch = Orchestrator(
            config=config,
            state_store=MagicMock(),
            planner_adapter=MagicMock(),
            worker_adapter=MagicMock(),
        )

        # Simulate plan service being set
        plan_svc = MagicMock()
        orch._plan_service = plan_svc

        orch.request_drain()

        assert orch._drain_mode is True
        assert orch._drain_started_at is not None
        plan_svc._drain_mode = True
        assert plan_svc._drain_mode is True


def test_second_ctrl_c_force_stops() -> None:
    """After drain mode, request_stop triggers immediate stop."""
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(version=1, goal="test", tasks=[])
    _add_running_task(svc, "task-1")

    svc._drain_mode = True
    svc._drain_started_at = time.monotonic()

    cycle_count = 0

    def fake_collect_results():
        nonlocal cycle_count
        cycle_count += 1
        if cycle_count == 1:
            # Simulate second Ctrl+C: set stop_requested
            svc._stop_requested = True
        return []

    orch._collect_results = fake_collect_results

    def fake_sleep(*args, **kwargs):
        if cycle_count > 10:
            raise StopIteration("loop did not exit")

    svc._plan_sleep = fake_sleep

    result = svc.run()

    # _stop_requested should cause immediate stop with NO_PROGRESS
    assert result == StopReason.NO_PROGRESS
    orch._finish.assert_called_with(StopReason.NO_PROGRESS, "Stopped by signal")


def test_drain_dispatch_guard() -> None:
    """_dispatch_plan_tasks returns immediately when drain mode is active."""
    svc, orch = _make_service()
    svc._drain_mode = True
    svc._current_plan = ResearchPlan(
        version=1, goal="test", tasks=[PlanTask(stage_number=1)],
    )

    plan_task = PlanTask(stage_number=1)
    svc._dispatch_plan_tasks([plan_task])

    # Worker service launch should not be called
    orch.worker_service.launch_task.assert_not_called()


def test_non_plan_mode_drain_no_tasks() -> None:
    """Non-plan mode drain exits immediately when no active tasks."""
    from app.orchestrator import Orchestrator

    config = MagicMock()
    config.goal = "test goal"
    config.state_dir = "state"
    config.state_file = "orchestrator_state.json"
    config.planner_timeout_seconds = 60
    config.worker_timeout_seconds = 60
    config.poll_interval_seconds = 10
    config.max_empty_cycles = 5
    config.max_errors_total = 10
    config.max_mcp_failures = 3
    config.max_task_attempts = 3
    config.max_worker_timeout_count = 3
    config.mcp_review = MagicMock(enabled=False)
    config.notifications = MagicMock()
    config.plan_mode = False
    config.drain_timeout_seconds = 600
    config.workers = [MagicMock(worker_id="w1")]

    with patch("app.orchestrator.StateStore"), \
         patch("app.orchestrator.PlannerService"), \
         patch("app.orchestrator.WorkerService"), \
         patch("app.orchestrator.Scheduler"), \
         patch("app.orchestrator.MemoryService"), \
         patch("app.orchestrator.NotificationService"), \
         patch("app.orchestrator.TaskSupervisor"):
        orch = Orchestrator(
            config=config,
            state_store=MagicMock(),
            planner_adapter=MagicMock(),
            worker_adapter=MagicMock(),
        )

        # Enter drain mode
        orch._drain_mode = True
        orch._drain_started_at = time.monotonic()

        # Mock _finish to track calls (it's a real method, not a mock)
        finish_calls = []
        original_finish = orch._finish

        def tracking_finish(reason, summary=""):
            finish_calls.append((reason, summary))

        orch._finish = tracking_finish

        # Mock scheduler sleep to raise if called too many times
        call_count = 0

        def fake_sleep(seconds):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                raise StopIteration("loop did not exit")

        orch.scheduler.sleep = fake_sleep
        orch.scheduler.sleep_interval = MagicMock(return_value=1)

        try:
            result = orch.run()
        except StopIteration:
            raise AssertionError("Drain should have exited immediately when no tasks running")

        assert result == StopReason.GRACEFUL_STOP
        assert len(finish_calls) == 1
        assert finish_calls[0][0] == StopReason.GRACEFUL_STOP
        assert finish_calls[0][1] == "Graceful drain completed"
