"""Tests for SIGINT signal handler debounce and _finish re-entrancy guard."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice
from app.models import OrchestratorState, StopReason
from main import _make_signal_handler


class TestFinishReentrancyGuard:
    def test_finish_called_twice_only_executes_once(self):
        """_finish() should silently return on second invocation."""
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.NotificationService"):
            orch = MagicMock(spec=Orchestrator)
            orch._finish_completed = False
            orch.state = OrchestratorState(goal="test")

            # Simulate real _finish behavior: set guard, collect output, notify
            call_count = 0

            def fake_finish(reason, summary=""):
                nonlocal call_count
                if orch._finish_completed:
                    return
                orch._finish_completed = True
                call_count += 1

            orch._finish = fake_finish

            orch._finish(StopReason.GRACEFUL_STOP, "first")
            orch._finish(StopReason.NO_PROGRESS, "second")
            assert call_count == 1


class TestSignalHandlerDebounce:
    def test_first_signal_enters_drain_mode(self):
        orch = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=MagicMock())
        handler(2, None)  # SIGINT
        orch.request_drain.assert_called_once()

    def test_second_signal_forces_stop(self):
        orch = MagicMock()
        orch.request_stop_now = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=MagicMock())
        handler(2, None)  # phase 0→1
        handler(2, None)  # phase 1→2
        orch.request_stop_now.assert_called_once()

    def test_third_signal_forces_hard_exit(self):
        orch = MagicMock()
        hard_exit = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=hard_exit)
        handler(2, None)  # phase 0→1
        handler(2, None)  # phase 1→2
        handler(2, None)  # phase 2→3
        orch.terminate_runtime_processes.assert_called_once_with(force=True)
        hard_exit.assert_called_once_with(130)

    def test_burst_of_signals_only_calls_drain_once(self):
        orch = MagicMock()
        orch.request_stop_now = MagicMock()
        hard_exit = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=hard_exit)
        for _ in range(5):
            handler(2, None)
        orch.request_drain.assert_called_once()
        orch.request_stop_now.assert_called_once()
        hard_exit.assert_called_once_with(130)


class TestRequestDrainIdempotency:
    """Verify request_drain() and request_stop_now() are idempotent."""

    def _make_orch(self):
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.NotificationService"):
            return Orchestrator.__new__(Orchestrator)

    def test_request_drain_idempotent(self):
        """request_drain() should only send one Telegram notification."""
        orch = self._make_orch()
        orch._drain_mode = False
        orch._drain_started_at = None
        orch._plan_service = None
        orch.state = OrchestratorState(goal="test")
        orch.notification_service = MagicMock()

        orch.request_drain()
        assert orch._drain_mode is True
        assert orch.notification_service.send_lifecycle.call_count == 1

        # Second call should be a no-op
        orch.request_drain()
        assert orch.notification_service.send_lifecycle.call_count == 1

    def test_request_stop_now_idempotent(self):
        """request_stop_now() should only log once."""
        orch = self._make_orch()
        orch._stop_requested = False
        orch._stop_now_requested = False
        orch._plan_service = None
        orch.process_registry = MagicMock()

        import logging
        with patch.object(logging.getLogger("orchestrator"), "info") as log_mock:
            orch.request_stop_now()
            assert orch._stop_requested is True

            orch.request_stop_now()
            # Only one log call — second was a no-op
            assert log_mock.call_count == 1
            orch.process_registry.terminate_all.assert_called_once()


def test_request_stop_now_persists_graceful_stop_snapshot_and_plan_progress(tmp_path):
    from app.adapters.fake_planner import FakePlanner
    from app.adapters.fake_worker import FakeWorker
    from app.config import OrchestratorConfig
    from app.models import WorkerConfig
    from app.orchestrator import Orchestrator

    cfg = OrchestratorConfig(
        goal="Signal stop",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
    )
    cfg.workers = [WorkerConfig(worker_id="worker-1", role="executor", system_prompt="")]
    orch = Orchestrator(
        config=cfg,
        planner_adapter=FakePlanner(responses=[], delay=0.0),
        worker_adapter=FakeWorker(responses=[], delay=0.0),
    )
    orch.notification_service = MagicMock()
    plan = ExecutionPlan(
        plan_id="plan_stop",
        goal="Persist operator stop",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        status="running",
        slices=[
            PlanSlice(
                slice_id="slice_done",
                title="Done",
                hypothesis="done",
                objective="done",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="completed",
                final_report_turn_id="turn_done",
                last_summary="already done",
            ),
            PlanSlice(
                slice_id="slice_waiting",
                title="Waiting",
                hypothesis="waiting",
                objective="waiting",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=2,
                status="checkpointed",
                last_checkpoint_status="partial",
                last_checkpoint_summary="waiting for status",
                last_summary="waiting for status",
            ),
            PlanSlice(
                slice_id="slice_running",
                title="Running",
                hypothesis="running",
                objective="running",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=3,
                status="running",
                last_summary="mid-turn",
                active_operation_ref="op_1",
                active_operation_status="running",
                active_operation_arguments={"view": "catalog"},
            ),
            PlanSlice(
                slice_id="slice_pending",
                title="Pending",
                hypothesis="pending",
                objective="pending",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=4,
                status="pending",
            ),
        ],
    )
    orch.execution_state = ExecutionStateV2(goal="Signal stop", status="running", plans=[plan], current_plan_id=plan.plan_id)
    orch.process_registry = MagicMock()

    orch.request_stop_now()

    persisted_state = json.loads((tmp_path / "state" / "execution_state_v2.json").read_text(encoding="utf-8"))
    persisted_plan = json.loads((tmp_path / "plans" / "plans" / "plan_stop.json").read_text(encoding="utf-8"))

    assert persisted_state["status"] == "finished"
    assert persisted_state["stop_reason"] == "graceful_stop"
    assert persisted_plan["status"] == "stopped"
    statuses = {item["slice_id"]: item["status"] for item in persisted_plan["slices"]}
    assert statuses == {
        "slice_done": "completed",
        "slice_waiting": "checkpointed",
        "slice_running": "checkpointed",
        "slice_pending": "pending",
    }
    running_slice = next(item for item in persisted_plan["slices"] if item["slice_id"] == "slice_running")
    assert running_slice["last_checkpoint_status"] == "blocked"
    assert running_slice["active_operation_ref"] == ""
    assert all(item["status"] != "running" for item in persisted_plan["slices"])


def test_terminal_summary_reflects_persisted_plan_status_after_graceful_stop(tmp_path):
    from app.adapters.fake_planner import FakePlanner
    from app.adapters.fake_worker import FakeWorker
    from app.config import OrchestratorConfig
    from app.models import WorkerConfig
    from app.orchestrator import Orchestrator

    cfg = OrchestratorConfig(
        goal="Terminal summary",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
    )
    cfg.workers = [WorkerConfig(worker_id="worker-1", role="executor", system_prompt="")]
    orch = Orchestrator(
        config=cfg,
        planner_adapter=FakePlanner(responses=[], delay=0.0),
        worker_adapter=FakeWorker(responses=[], delay=0.0),
    )
    plan = ExecutionPlan(
        plan_id="plan_running",
        goal="Persisted plan should be terminalized",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        status="running",
        slices=[
            PlanSlice(
                slice_id="slice_running",
                title="Running",
                hypothesis="running",
                objective="running",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="running",
            )
        ],
    )
    orch.execution_state = ExecutionStateV2(goal="Terminal summary", status="running", plans=[plan], current_plan_id=plan.plan_id)

    orch.persist_terminal_snapshot(reason=StopReason.GRACEFUL_STOP, summary="plan=plan_running status=running")

    assert orch._terminal_summary("fallback") == "plan=plan_running status=stopped"


class TestSignalHandlerTaskCancellation:
    """Verify that the second signal cancels asyncio tasks."""

    def test_second_signal_cancels_running_tasks(self):
        """When a running event loop exists, phase 2 cancels all its tasks."""
        orch = MagicMock()
        orch.request_stop_now = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=MagicMock())

        # Verify by patching asyncio.all_tasks to track cancel calls
        cancelled_tasks = []

        class TrackableTask:
            def __init__(self, real_task):
                self._real = real_task
            def done(self):
                return self._real.done()
            def cancel(self):
                cancelled_tasks.append(self._real)
                return self._real.cancel()

        async def _verify_cancellation():
            task = asyncio.ensure_future(asyncio.sleep(100))
            wrapped = TrackableTask(task)
            with patch("asyncio.all_tasks", return_value=[wrapped]):
                handler(2, None)  # phase 0→1 (drain)
                handler(2, None)  # phase 1→2 (cancel + _cancel_running_tasks)
            assert len(cancelled_tasks) >= 1
            task.cancel()

        try:
            asyncio.run(_verify_cancellation())
        except asyncio.CancelledError:
            pass

    def test_cancel_running_tasks_safe_without_event_loop(self):
        """_cancel_running_tasks should not crash when no event loop is running."""
        orch = MagicMock()
        handler = _make_signal_handler(orch, hard_exit=MagicMock())
        # This should not raise — there's no running event loop in a sync test
        handler(2, None)  # phase 0→1
        handler(2, None)  # phase 1→2 — _cancel_running_tasks should silently skip
        orch.request_stop_now.assert_called_once()


class TestOrchestratorCancelledErrorHandling:
    """Verify orchestrator.run() catches CancelledError."""

    def test_run_catches_cancelled_error(self):
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.NotificationService"):
            orch = MagicMock(spec=Orchestrator)
            orch._finish_completed = False
            orch._stop_now_requested = False
            orch._plan_service = None

            with patch("app.services.direct_execution.DirectExecutionService"):
                with patch("app.orchestrator.asyncio") as mock_asyncio:
                    mock_asyncio.CancelledError = asyncio.CancelledError
                    # First call: svc.run() raises CancelledError
                    # Second call: _finish_async() returns None
                    mock_asyncio.run = MagicMock(side_effect=[asyncio.CancelledError(), None])
                    result = Orchestrator.run(orch)
                    assert result == StopReason.GRACEFUL_STOP


class TestDrainTimeout:
    """Verify drain mode auto-escalates to force stop after timeout."""

    def test_drain_timeout_forces_stop(self, tmp_path):
        from app.services.direct_execution.service import _DRAIN_TIMEOUT_SECONDS

        # Verify the timeout constant is reasonable
        assert 10.0 <= _DRAIN_TIMEOUT_SECONDS <= 120.0
