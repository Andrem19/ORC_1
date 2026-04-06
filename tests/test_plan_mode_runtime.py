"""Targeted unit tests for plan-mode runtime behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.models import OrchestratorState, ProcessInfo, StopReason, Task, TaskResult, TaskStatus
from app.plan_models import PlanTask, ResearchPlan, TaskReport
from app.planner_runtime import PlannerRunSnapshot
from app.plan_prompts import build_plan_repair_prompt, build_plan_revision_prompt
from app.plan_store import PlanStore
from app.plan_validation import PlanRepairRequest, PlanValidationError
from app.services.plan_orchestrator_service import PlanOrchestratorService


def _make_service(*, worker_ids: list[str] | None = None) -> tuple[PlanOrchestratorService, MagicMock]:
    orch = MagicMock()
    orch.config = SimpleNamespace(
        goal="test goal",
        max_concurrent_plan_tasks=2,
        research_config={},
        mcp_review=SimpleNamespace(max_problems_in_context=10),
        plan_task_timeout_seconds=600,
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


def test_process_plan_results_uses_structured_plan_report_without_reparse() -> None:
    svc, orch = _make_service()
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="task-1",
                plan_version=1,
                stage_number=1,
                stage_name="Stage 1",
            ),
        ],
    )
    task = Task(
        task_id="task-1",
        assigned_worker_id="qwen-1",
        status=TaskStatus.COMPLETED,
        metadata={"plan_mode": True, "plan_version": 1, "stage_number": 1},
    )
    svc.state.add_task(task)
    report = TaskReport(
        task_id="task-1",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        what_was_requested="run stage 1",
        what_was_done="completed stage 1",
        results_table=[{"run_id": "r1", "verdict": "PROMOTE"}],
        key_metrics={"net_pnl": 10},
        artifacts=["run:r1"],
        confidence=0.9,
        verdict="PROMOTE",
        raw_output='{"status":"success"}',
    )
    result = TaskResult(
        task_id="task-1",
        worker_id="qwen-1",
        status="success",
        raw_output="not valid json anymore",
        plan_report=report,
    )

    svc._process_plan_results([result])

    pt = svc._current_plan.get_task_by_stage(1)
    assert pt is not None
    assert pt.status == TaskStatus.COMPLETED
    assert pt.verdict == "PROMOTE"
    assert pt.results_table_rows == [{"run_id": "r1", "verdict": "PROMOTE"}]
    svc._plan_store.save_report.assert_called_once_with(report)
    orch._log_event.assert_called()


def test_dispatchable_tasks_use_explicit_dependencies() -> None:
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        tasks=[
            PlanTask(stage_number=0, status=TaskStatus.COMPLETED),
            PlanTask(stage_number=1, depends_on=[0]),
            PlanTask(stage_number=2, depends_on=[0]),
            PlanTask(stage_number=4, depends_on=[0]),
            PlanTask(stage_number=3, depends_on=[1, 2]),
            PlanTask(stage_number=5, depends_on=[3]),
        ],
    )

    assert [task.stage_number for task in plan.dispatchable_tasks()] == [1, 2, 4]


def test_dispatchable_tasks_support_legacy_execution_order() -> None:
    plan = ResearchPlan(
        schema_version=1,
        version=1,
        goal="legacy",
        execution_order=[0, 1, 2],
        tasks=[
            PlanTask(stage_number=0, status=TaskStatus.COMPLETED),
            PlanTask(stage_number=1),
            PlanTask(stage_number=2),
        ],
    )

    assert [task.stage_number for task in plan.dispatchable_tasks()] == [1]


def test_pick_worker_round_robins_idle_workers() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1", "qwen-2"])

    assert svc._pick_worker() == "qwen-1"
    assert svc._pick_worker() == "qwen-2"
    assert svc._pick_worker() == "qwen-1"


def test_dispatch_plan_tasks_uses_both_workers() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1", "qwen-2"])
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(task_id="t1", plan_version=1, stage_number=1, stage_name="A"),
            PlanTask(task_id="t2", plan_version=1, stage_number=2, stage_name="B"),
        ],
    )
    svc.worker_service.start_plan_task.side_effect = [
        ProcessInfo(task_id="t1", worker_id="qwen-1", pid=1),
        ProcessInfo(task_id="t2", worker_id="qwen-2", pid=2),
    ]

    svc._dispatch_plan_tasks(svc._current_plan.tasks)

    assigned = [task.assigned_worker_id for task in svc._current_plan.tasks]
    assert assigned == ["qwen-1", "qwen-2"]
    assert len(svc.state.processes) == 2


def test_run_does_not_increment_empty_cycles_when_worker_active() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc._plan_store = None
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="task-1",
                plan_version=1,
                stage_number=0,
                stage_name="Active",
                status=TaskStatus.RUNNING,
            ),
        ],
    )
    svc.state.current_plan_version = 1
    svc.state.empty_cycles = 0
    svc.state.add_task(
        Task(
            task_id="task-1",
            status=TaskStatus.RUNNING,
            assigned_worker_id="qwen-1",
            metadata={"plan_mode": True, "plan_version": 1, "stage_number": 0},
        )
    )
    orch.scheduler.plan_tasks_to_dispatch.return_value = 0
    orch.scheduler.should_stop_orchestrator.return_value = None
    orch.planner_service.is_running = False

    def _stop_after_one_cycle() -> None:
        svc._stop_requested = True

    svc._plan_sleep = _stop_after_one_cycle

    reason = svc.run()

    assert reason.value == "no_progress"
    assert svc.state.empty_cycles == 0


def test_reconcile_current_plan_state_restores_progress_and_baseline(tmp_path: Path) -> None:
    plans_dir = tmp_path / "plans"
    plan_store = PlanStore(str(plans_dir))
    plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="stage-0",
                plan_version=1,
                stage_number=0,
                stage_name="Baseline",
            ),
        ],
    )
    plan_store.save_plan(plan)
    report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        what_was_requested="baseline",
        what_was_done="measured baseline",
        results_table=[{"run_id": "baseline-run", "snapshot_id": "active-signal-v1@1"}],
        key_metrics={"net_pnl": 920.09, "trades": 279},
        artifacts=["run_id: baseline-run", "snapshot: active-signal-v1@1"],
        verdict="WATCHLIST",
        raw_output="{}",
    )
    plan_store.save_report(report)

    svc, orch = _make_service(worker_ids=["qwen-1"])
    orch._plan_store = plan_store
    svc._plan_store = plan_store
    svc._current_plan = plan_store.load_latest_plan()
    svc.state.add_task(
        Task(
            task_id="stage-0",
            status=TaskStatus.COMPLETED,
            assigned_worker_id="qwen-1",
            metadata={"plan_mode": True, "plan_version": 1, "stage_number": 0},
        )
    )

    svc._reconcile_current_plan_state()

    pt = svc._current_plan.get_task_by_stage(0)
    assert pt is not None
    assert pt.status == TaskStatus.COMPLETED
    assert pt.results_table_rows == [{"run_id": "baseline-run", "snapshot_id": "active-signal-v1@1"}]
    assert svc._current_plan.baseline_run_id == "baseline-run"
    assert svc._current_plan.baseline_snapshot_ref == "active-signal-v1@1"
    assert svc._current_plan.baseline_metrics["trades"] == 279


def test_process_plan_data_rejects_placeholder_instructions() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1"])
    svc.planner_service.last_plan_raw_output = '{"tasks":[{"bad":true}]}'
    svc._build_repair_request = MagicMock(return_value=PlanRepairRequest(
        goal="test goal",
        plan_version=1,
        attempt_number=2,
        invalid_plan_data={"plan_action": "create"},
        validation_errors=[],
    ))

    svc._process_plan_data(
        {
            "schema_version": 2,
            "plan_action": "create",
            "plan_version": 1,
            "tasks": [
                {
                    "stage_number": 5,
                    "stage_name": "Bad Stage",
                    "depends_on": [0],
                    "agent_instructions": [
                        "backtests_strategy(action='clone', source_snapshot_id='<best_snapshot_id>')",
                        "backtests_studies(action='start', ...)",
                    ],
                    "results_table_columns": ["run_id"],
                    "decision_gates": [],
                },
            ],
        }
    )

    assert svc._current_plan is None
    svc._plan_store.save_plan.assert_not_called()
    svc._plan_store.save_rejected_plan_attempt.assert_called_once()
    svc.planner_service.start_plan_repair.assert_called_once()
    assert svc.state.current_plan_attempt_type == "repair"


def test_dispatch_plan_tasks_resolves_symbolic_refs_before_launch() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1"])
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(task_id="stage-0", plan_version=1, stage_number=0, stage_name="Base"),
            PlanTask(
                task_id="stage-1",
                plan_version=1,
                stage_number=1,
                stage_name="Inspect",
                depends_on=[0],
                agent_instructions=[
                    "backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')",
                ],
            ),
        ],
    )
    report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=[{"run_id": "baseline-run"}],
        verdict="PROMOTE",
        raw_output="{}",
    )
    svc._plan_store.load_reports_for_plan.return_value = [report]
    svc.worker_service.start_plan_task.return_value = ProcessInfo(task_id="stage-1", worker_id="qwen-1", pid=10)

    svc._dispatch_plan_tasks([svc._current_plan.get_task_by_stage(1)])

    kwargs = svc.worker_service.start_plan_task.call_args.kwargs
    assert kwargs["agent_instructions"] == [
        "backtests_runs(action='inspect', run_id='baseline-run', view='detail')",
    ]
    assert kwargs["steps"][0].instruction == "backtests_runs(action='inspect', run_id='baseline-run', view='detail')"


def test_dispatch_plan_tasks_marks_stage_failed_when_symbolic_field_missing() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="stage-0",
                plan_version=1,
                stage_number=0,
                stage_name="Base",
                status=TaskStatus.COMPLETED,
            ),
            PlanTask(
                task_id="stage-1",
                plan_version=1,
                stage_number=1,
                stage_name="Inspect",
                depends_on=[0],
                agent_instructions=[
                    "backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')",
                ],
            ),
        ],
    )
    report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=[{"snapshot_id": "snap"}],
        verdict="PROMOTE",
        raw_output="{}",
    )
    svc._plan_store.load_reports_for_plan.return_value = [report]
    stage = svc._current_plan.get_task_by_stage(1)

    svc._dispatch_plan_tasks([stage])

    assert stage.status == TaskStatus.FAILED
    svc.worker_service.start_plan_task.assert_not_called()
    orch.memory_service.record_error.assert_called_once()


def test_invalid_plan_loop_stops_after_max_attempts() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc.state.current_plan_attempt = 3
    svc.state.current_plan_attempt_type = "repair"
    svc.planner_service.last_plan_raw_output = '{"tasks":[]}'

    svc._process_plan_data(
        {
            "schema_version": 2,
            "plan_action": "create",
            "plan_version": 1,
            "tasks": [
                {
                    "stage_number": 0,
                    "stage_name": "Bad",
                    "agent_instructions": ["backtests_runs(action='inspect', run_id='<run_id>', view='detail')"],
                    "decision_gates": [],
                    "results_table_columns": [],
                },
            ],
        }
    )

    assert svc._terminal_stop_reason == StopReason.INVALID_PLAN_LOOP
    svc.planner_service.start_plan_repair.assert_not_called()
    orch.notification_service.send_error.assert_called()


def test_build_plan_revision_prompt_includes_measured_baseline() -> None:
    plan = ResearchPlan(
        schema_version=2,
        version=1,
        goal="test",
        baseline_run_id="baseline-run",
        baseline_snapshot_ref="active-signal-v1@1",
        baseline_metrics={"sharpe": 1.446, "trades": 279},
    )

    prompt = build_plan_revision_prompt(goal="test", current_plan=plan, reports=[])

    assert "Measured Baseline" in prompt
    assert "baseline-run" in prompt
    assert "active-signal-v1@1" in prompt
    assert "1.446" in prompt
    assert "tasks_to_dispatch" not in prompt


def test_build_plan_repair_prompt_includes_validation_errors() -> None:
    prompt = build_plan_repair_prompt(
        PlanRepairRequest(
            goal="test goal",
            plan_version=1,
            attempt_number=2,
            invalid_plan_data={"plan_action": "create", "tasks": []},
            validation_errors=[
                PlanValidationError(
                    stage_number=0,
                    instruction_index=0,
                    code="legacy_placeholder",
                    message="Legacy <...> placeholder is not allowed",
                    offending_text="<run_id>",
                )
            ],
        )
    )

    assert "Repair Rules" in prompt
    assert "legacy_placeholder" in prompt
    assert "<run_id>" in prompt


def test_plan_sleep_shortens_while_planner_running() -> None:
    svc, orch = _make_service()
    svc.planner_service.is_running = True
    orch.scheduler.sleep_interval.return_value = 120

    svc._plan_sleep()

    orch.scheduler.sleep.assert_called_once_with(seconds=30)


def test_handle_planner_timeout_retries_once() -> None:
    svc, orch = _make_service()
    snapshot = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=6000,
        timeout_retry_count=0,
    )
    svc.planner_service.terminate_plan_run.return_value = snapshot
    svc.planner_service.restart_plan_request.return_value = True
    svc._plan_store.save_planner_run.return_value = Path("/tmp/planner_run.json")

    svc._handle_planner_timeout(snapshot)

    assert svc._terminal_stop_reason is None
    svc.planner_service.restart_plan_request.assert_called_once()
    orch.notification_service.send_error.assert_called_once()


def test_handle_planner_timeout_stops_after_retry_budget() -> None:
    svc, orch = _make_service()
    snapshot = PlannerRunSnapshot(
        request_type="repair",
        request_version=1,
        attempt_number=2,
        prompt_length=14000,
        timeout_retry_count=1,
    )
    svc.planner_service.terminate_plan_run.return_value = snapshot
    svc._plan_store.save_planner_run.return_value = Path("/tmp/planner_run.json")

    svc._handle_planner_timeout(snapshot)

    assert svc._terminal_stop_reason == StopReason.PLANNER_TIMEOUT
    orch.notification_service.send_error.assert_called_once()


def test_check_silent_workers_classifies_stderr_only() -> None:
    svc, _orch = _make_service()
    task = Task(
        task_id="task-1",
        status=TaskStatus.RUNNING,
        assigned_worker_id="qwen-1",
        metadata={"plan_mode": True},
    )
    svc.state.add_task(task)
    svc.state.processes.append(
        ProcessInfo(
            task_id="task-1",
            worker_id="qwen-1",
            pid=123,
            started_at=datetime(2026, 4, 6, 12, 0, tzinfo=timezone.utc).isoformat(),
            partial_output="",
            partial_error_output="auth warning",
            stderr_bytes=len("auth warning".encode()),
        )
    )
    handle = MagicMock()
    handle.process.poll.return_value = None
    svc.worker_service._active_handles = {"task-1": handle}

    svc._check_silent_workers()

    assert svc.state.processes[0].monitor_state == "stderr_only"
