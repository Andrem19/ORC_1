"""Targeted unit tests for plan-mode runtime behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.adapters.base import ProcessHandle
from app.models import OrchestratorState, ProcessInfo, StopReason, Task, TaskResult, TaskStatus
from app.plan_models import PlanTask, ResearchPlan, TaskReport
from app.planner_runtime import PlannerRunSnapshot
from app.plan_prompts import build_plan_repair_prompt, build_plan_revision_prompt
from app.plan_store import PlanStore
from app.plan_validation import PlanRepairRequest, PlanValidationError
from app.result_parser import parse_plan_output
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


def test_process_plan_data_accepts_decision_gate_reason_field() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1"])
    svc.planner_service.last_plan_raw_output = '{"tasks":[{"stage_number":0}]}'

    svc._process_plan_data(
        {
            "schema_version": 3,
            "plan_action": "create",
            "plan_version": 1,
            "_request_type": "create",
            "_request_version": 1,
            "_attempt_number": 1,
            "_failure_class": "none",
            "tasks": [
                {
                    "stage_number": 0,
                    "stage_name": "Baseline",
                    "steps": [
                        {
                            "step_id": "baseline_run",
                            "kind": "work",
                            "instruction": "Run baseline",
                        }
                    ],
                    "results_table_columns": ["run_id"],
                    "decision_gates": [
                        {
                            "metric": "sharpe",
                            "threshold": 0.9,
                            "comparator": "gte",
                            "verdict_pass": "PROMOTE",
                            "verdict_fail": "REJECT",
                            "reason": "Baseline must reproduce expected Sharpe.",
                        }
                    ],
                },
            ],
        }
    )

    assert svc._current_plan is not None
    gate = svc._current_plan.tasks[0].decision_gates[0]
    assert gate.metric == "sharpe"
    assert gate.reason == "Baseline must reproduce expected Sharpe."


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
    # dependency_reports should contain the report from stage 0
    dep_reports = kwargs.get("dependency_reports")
    assert dep_reports is not None
    assert len(dep_reports) == 1
    assert dep_reports[0].task_id == "stage-0"
    assert dep_reports[0].results_table == [{"run_id": "baseline-run"}]


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


def test_dispatch_plan_tasks_passes_dependency_reports_to_worker() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1"])
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
                stage_name="UsesBase",
                depends_on=[0],
                agent_instructions=["do something"],
            ),
        ],
    )
    report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=[{"run_id": "r1", "snapshot_id": "snap1"}],
        key_metrics={"sharpe": 1.2},
        artifacts=["run:r1"],
        verdict="PROMOTE",
        raw_output="{}",
    )
    svc._plan_store.load_reports_for_plan.return_value = [report]
    svc.worker_service.start_plan_task.return_value = ProcessInfo(task_id="stage-1", worker_id="qwen-1", pid=10)

    svc._dispatch_plan_tasks([svc._current_plan.get_task_by_stage(1)])

    kwargs = svc.worker_service.start_plan_task.call_args.kwargs
    dep_reports = kwargs.get("dependency_reports")
    assert dep_reports is not None
    assert len(dep_reports) == 1
    assert dep_reports[0].results_table == [{"run_id": "r1", "snapshot_id": "snap1"}]


def test_dispatch_plan_tasks_no_dependency_reports_when_no_deps() -> None:
    svc, _orch = _make_service(worker_ids=["qwen-1"])
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="stage-0",
                plan_version=1,
                stage_number=0,
                stage_name="Base",
                agent_instructions=["do baseline"],
            ),
        ],
    )
    svc._plan_store.load_reports_for_plan.return_value = []
    svc.worker_service.start_plan_task.return_value = ProcessInfo(task_id="stage-0", worker_id="qwen-1", pid=10)

    svc._dispatch_plan_tasks([svc._current_plan.get_task_by_stage(0)])

    kwargs = svc.worker_service.start_plan_task.call_args.kwargs
    assert kwargs.get("dependency_reports") is None


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


def test_transport_failure_retries_same_request_without_repair() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc.planner_service.plan_transport_retry_count = 0
    svc.planner_service.restart_plan_request.return_value = True

    svc._process_plan_data(
        {
            "plan_action": "continue",
            "plan_version": 0,
            "_parse_failed": True,
            "_failure_class": "transport_error",
            "_request_type": "create",
            "_request_version": 1,
            "_attempt_number": 1,
            "_transport_errors": ["StructuredOutput activity detected but no recoverable structured payload was found"],
        }
    )

    svc.planner_service.restart_plan_request.assert_called_once_with(reason="transport_error")
    svc.planner_service.start_plan_repair.assert_not_called()
    svc._plan_store.save_rejected_plan_attempt.assert_called_once()
    kwargs = svc._plan_store.save_rejected_plan_attempt.call_args.kwargs
    assert kwargs["plan_version"] == 1
    assert kwargs["failure_class"] == "transport_error"
    assert svc._terminal_stop_reason is None


def test_transport_failure_stops_after_retry_budget() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc.planner_service.plan_transport_retry_count = 1

    svc._process_plan_data(
        {
            "plan_action": "continue",
            "plan_version": 0,
            "_parse_failed": True,
            "_failure_class": "transport_error",
            "_request_type": "create",
            "_request_version": 1,
            "_attempt_number": 1,
            "_transport_errors": ["lost StructuredOutput payload"],
        }
    )

    assert svc._terminal_stop_reason == StopReason.INVALID_OUTPUT
    svc.planner_service.restart_plan_request.assert_not_called()
    svc.planner_service.start_plan_repair.assert_not_called()
    orch.notification_service.send_error.assert_called()


def test_parse_error_does_not_enter_repair_flow() -> None:
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc.planner_service.plan_transport_retry_count = 1

    svc._process_plan_data(
        {
            "plan_action": "continue",
            "plan_version": 0,
            "_parse_failed": True,
            "_failure_class": "parse_error",
            "_request_type": "create",
            "_request_version": 1,
            "_attempt_number": 1,
            "reason": "Failed to parse planner output as JSON",
        }
    )

    assert svc._terminal_stop_reason == StopReason.INVALID_OUTPUT
    svc.planner_service.start_plan_repair.assert_not_called()
    svc.planner_service.restart_plan_request.assert_not_called()
    kwargs = svc._plan_store.save_rejected_plan_attempt.call_args.kwargs
    assert kwargs["failure_class"] == "parse_error"
    assert kwargs["plan_version"] == 1


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


def test_planner_service_prefers_structured_payload_over_tool_result_text() -> None:
    from app.services.planner_service import PlannerService

    adapter = MagicMock()
    service = PlannerService(adapter=adapter, timeout=180)
    adapter.check.return_value = ("", True)

    transcript = "\n".join([
        '{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":"StructuredOutput","input":{}}}}',
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"StructuredOutput","input":{"schema_version":3,"plan_action":"create","plan_version":1,"tasks":[{"stage_number":0,"stage_name":"Baseline"}]}}]}}',
        '{"type":"user","message":{"content":[{"type":"tool_result","content":"Structured output provided successfully"}]}}',
    ])
    process = MagicMock()
    process.returncode = 0
    handle = ProcessHandle(
        process=process,
        task_id="planner",
        worker_id="planner",
        partial_output="Structured output provided successfully",
        metadata={"raw_stdout": transcript, "raw_stderr": "", "stream_event_count": 3, "output_mode": "stream-json"},
    )
    service._active_handle = handle
    service._plan_runtime = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        output_mode="stream-json",
    )
    service._plan_request_type = "create"
    service._plan_request_version = 1
    service._plan_request_attempt = 1

    parsed, finished = service.check_plan_output()

    assert finished is True
    assert parsed is not None
    assert parsed["plan_version"] == 1
    assert parsed["_failure_class"] == "none"
    assert service.last_plan_raw_output.startswith('{"schema_version": 3')


def test_planner_service_sets_transport_error_when_structured_output_lost() -> None:
    from app.services.planner_service import PlannerService

    adapter = MagicMock()
    service = PlannerService(adapter=adapter, timeout=180)
    adapter.check.return_value = ("", True)

    transcript = "\n".join([
        '{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":"StructuredOutput","input":{}}}}',
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\\"schema_version\\":3"}}}',
    ])
    process = MagicMock()
    process.returncode = 0
    handle = ProcessHandle(
        process=process,
        task_id="planner",
        worker_id="planner",
        partial_output="Structured output provided successfully",
        metadata={"raw_stdout": transcript, "raw_stderr": "", "stream_event_count": 2, "output_mode": "stream-json"},
    )
    service._active_handle = handle
    service._plan_runtime = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        output_mode="stream-json",
    )
    service._plan_request_type = "create"
    service._plan_request_version = 1
    service._plan_request_attempt = 1

    parsed, finished = service.check_plan_output()

    assert finished is True
    assert parsed is not None
    assert parsed["_failure_class"] == "transport_error"
    assert parsed["_request_version"] == 1
    assert service.last_plan_raw_output == "Structured output provided successfully"


def test_planner_service_sets_parse_error_when_no_structured_activity() -> None:
    from app.services.planner_service import PlannerService

    adapter = MagicMock()
    service = PlannerService(adapter=adapter, timeout=180)
    adapter.check.return_value = ("", True)

    process = MagicMock()
    process.returncode = 0
    handle = ProcessHandle(
        process=process,
        task_id="planner",
        worker_id="planner",
        partial_output="just chatter",
        metadata={"raw_stdout": "just chatter", "raw_stderr": "", "stream_event_count": 0},
    )
    service._active_handle = handle
    service._plan_runtime = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        output_mode="stream-json",
    )
    service._plan_request_type = "create"
    service._plan_request_version = 1
    service._plan_request_attempt = 1

    parsed, finished = service.check_plan_output()

    assert finished is True
    assert parsed is not None
    assert parsed["_failure_class"] == "parse_error"
    assert parsed["_request_version"] == 1


def test_restart_plan_request_transport_retry_does_not_touch_timeout_retry() -> None:
    from app.services.planner_service import PlannerService

    adapter = MagicMock()
    service = PlannerService(adapter=adapter, timeout=180)
    service._plan_request_payload = {
        "request_type": "create",
        "goal": "goal",
        "research_context": None,
        "anti_patterns": None,
        "cumulative_summary": "",
        "worker_ids": ["qwen-1"],
        "mcp_problem_summary": None,
        "previous_plan_markdown": None,
        "plan_version": 1,
        "attempt_number": 1,
    }
    service._launch_plan_request = MagicMock()

    assert service.restart_plan_request(reason="transport_error") is True
    assert service.plan_transport_retry_count == 1
    assert service._plan_timeout_retry_count == 0


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


def test_check_silent_workers_classifies_slow_active() -> None:
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
            first_output_at=datetime(2026, 4, 6, 12, 0, 5, tzinfo=timezone.utc).isoformat(),
            last_output_at=datetime.now(timezone.utc).isoformat(),
            partial_output="running",
            stdout_bytes=len("running".encode()),
        )
    )
    handle = MagicMock()
    handle.process.poll.return_value = None
    svc.worker_service._active_handles = {"task-1": handle}

    svc._check_silent_workers()

    assert svc.state.processes[0].monitor_state == "slow_active"


def test_check_silent_workers_classifies_stalled() -> None:
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
            first_output_at=datetime(2026, 4, 6, 12, 0, 5, tzinfo=timezone.utc).isoformat(),
            last_output_at=datetime(2026, 4, 6, 12, 1, tzinfo=timezone.utc).isoformat(),
            partial_output="running",
            stdout_bytes=len("running".encode()),
        )
    )
    handle = MagicMock()
    handle.process.poll.return_value = None
    svc.worker_service._active_handles = {"task-1": handle}

    svc._check_silent_workers()

    assert svc.state.processes[0].monitor_state == "stalled"
    assert task.status == TaskStatus.RUNNING


def test_check_silent_workers_no_output_does_not_change_task_status() -> None:
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
        )
    )
    handle = MagicMock()
    handle.process.poll.return_value = None
    svc.worker_service._active_handles = {"task-1": handle}

    svc._check_silent_workers()

    assert svc.state.processes[0].monitor_state == "no_output"
    assert task.status == TaskStatus.RUNNING


def test_planner_output_state_classifies_structured_stream() -> None:
    snapshot = PlannerRunSnapshot(structured_delta_bytes=10)
    assert PlanOrchestratorService._planner_output_state(snapshot) == "structured_output_stream_active"


def test_planner_output_state_classifies_text_stream() -> None:
    snapshot = PlannerRunSnapshot(rendered_output="summary")
    assert PlanOrchestratorService._planner_output_state(snapshot) == "text_stream_active"


def test_planner_output_state_classifies_raw_stream_only() -> None:
    snapshot = PlannerRunSnapshot(output_bytes=10)
    assert PlanOrchestratorService._planner_output_state(snapshot) == "raw_stream_only"


def test_planner_output_state_classifies_stderr_only() -> None:
    snapshot = PlannerRunSnapshot(stderr_bytes=10)
    assert PlanOrchestratorService._planner_output_state(snapshot) == "stderr_only"


def test_check_planner_watchdog_soft_active_uses_recent_output() -> None:
    svc, orch = _make_service()
    snapshot = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        output_bytes=20,
        first_output_at_monotonic=1.0,
        last_output_at_monotonic=1_000_000.0,
    )
    svc.planner_service.plan_runtime_snapshot.return_value = snapshot

    import app.services.plan_orchestrator._planner_monitor as mod
    from unittest.mock import patch
    with patch.object(mod.time, "monotonic", return_value=1_000_010.0):
        snapshot.started_at_monotonic = 1_000_000.0 - 400.0
        svc._check_planner_watchdog()

    orch.notification_service.send_error.assert_called_once()
    svc.planner_service.terminate_plan_run.assert_not_called()


def test_check_planner_watchdog_stalled_uses_last_output_age() -> None:
    svc, orch = _make_service()
    snapshot = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        output_bytes=20,
        first_output_at_monotonic=1.0,
        last_output_at_monotonic=1_000_000.0,
    )
    svc.planner_service.plan_runtime_snapshot.return_value = snapshot

    import app.services.plan_orchestrator._planner_monitor as mod
    from unittest.mock import patch
    with patch.object(mod.time, "monotonic", return_value=1_000_400.0):
        snapshot.started_at_monotonic = 1_000_000.0 - 400.0
        svc._check_planner_watchdog()

    orch.notification_service.send_error.assert_called_once()
    svc.planner_service.terminate_plan_run.assert_not_called()


def test_check_planner_watchdog_hard_timeout_ignores_zero_rendered_text_when_active() -> None:
    svc, _orch = _make_service()
    snapshot = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=1000,
        structured_delta_bytes=50,
        first_output_at_monotonic=1.0,
        last_output_at_monotonic=1_000_000.0,
    )
    svc.planner_service.plan_runtime_snapshot.return_value = snapshot
    svc._handle_planner_timeout = MagicMock()

    import app.services.plan_orchestrator._planner_monitor as mod
    from unittest.mock import patch
    with patch.object(mod.time, "monotonic", return_value=1_000_100.0):
        snapshot.started_at_monotonic = 1_000_000.0 - 100.0
        svc._check_planner_watchdog()

    svc._handle_planner_timeout.assert_not_called()


# ---------------------------------------------------------------------------
# Phase 1: Fix double status modification
# ---------------------------------------------------------------------------


def test_plan_mode_partial_result_task_and_plan_task_both_completed() -> None:
    """Plan-mode partial results should mark BOTH Task and PlanTask as COMPLETED."""
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc._current_plan = ResearchPlan(
        version=1,
        goal="test",
        tasks=[
            PlanTask(
                task_id="stage-0",
                plan_version=1,
                stage_number=0,
                stage_name="Baseline",
                status=TaskStatus.RUNNING,
            ),
        ],
    )
    # Simulate Task created by dispatch
    task = Task(
        task_id="stage-0",
        description="baseline",
        assigned_worker_id="qwen-1",
        metadata={"plan_mode": True, "plan_version": 1, "stage_number": 0},
    )
    task.mark_running()
    orch.state.add_task(task)

    report = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="partial",
        what_was_done="partial baseline",
        verdict="WATCHLIST",
        raw_output="{}",
    )
    result = TaskResult(
        task_id="stage-0",
        worker_id="qwen-1",
        status="partial",
        plan_report=report,
        raw_output="{}",
    )

    # _collect_results calls _handle_task_result → then _process_plan_results
    from app.orchestrator import Orchestrator
    Orchestrator._handle_task_result(orch, task, result)

    assert task.status == TaskStatus.COMPLETED, f"Task should be COMPLETED, got {task.status}"
    assert orch.state.total_errors == 0, "Partial plan-mode result should NOT increment total_errors"


def test_reconciliation_does_not_overwrite_resolved_plan_task() -> None:
    """_reconcile_current_plan_state should not overwrite a resolved PlanTask."""
    svc, orch = _make_service(worker_ids=["qwen-1"])
    pt = PlanTask(
        task_id="stage-0",
        plan_version=1,
        stage_number=0,
        stage_name="Baseline",
        status=TaskStatus.COMPLETED,
    )
    svc._current_plan = ResearchPlan(version=1, goal="test", tasks=[pt])

    # Simulate a Task that is FAILED (from stale state before partial fix)
    task = Task(
        task_id="stage-0",
        description="baseline",
        metadata={"plan_mode": True, "plan_version": 1},
    )
    task.status = TaskStatus.FAILED
    orch.state.add_task(task)

    svc._plan_store.load_reports_for_plan.return_value = []
    svc._reconcile_current_plan_state()

    # PlanTask should stay COMPLETED (it was already resolved)
    assert pt.status == TaskStatus.COMPLETED, f"PlanTask should stay COMPLETED, got {pt.status}"


def test_maybe_update_plan_baseline_captures_partial_stage0() -> None:
    """_maybe_update_plan_baseline should capture baseline from partial stage 0."""
    svc, orch = _make_service(worker_ids=["qwen-1"])
    svc._current_plan = ResearchPlan(version=1, goal="test")

    pt = PlanTask(task_id="s0", plan_version=1, stage_number=0, stage_name="Base")
    report = TaskReport(
        task_id="s0",
        worker_id="qwen-1",
        plan_version=1,
        status="partial",
        results_table=[{"run_id": "r-partial", "snapshot_id": "snap1", "version": "2"}],
        key_metrics={"net_pnl": 500.0, "sharpe": 1.2},
        artifacts=["run_id:r-partial", "snapshot:snap1@2"],
        raw_output="{}",
    )

    svc._maybe_update_plan_baseline(pt, report)

    assert svc._current_plan.baseline_run_id == "r-partial"
    assert svc._current_plan.baseline_snapshot_ref == "snap1@2"
    assert svc._current_plan.baseline_metrics["net_pnl"] == 500.0


# ---------------------------------------------------------------------------
# Phase 2: Deadlock and recovery fixes
# ---------------------------------------------------------------------------


def test_symbolic_ref_failure_writes_report_and_downstream_not_deadlocked() -> None:
    """When a symbolic ref runtime error occurs, a failure report is saved so
    downstream stages are not permanently deadlocked."""
    svc, orch = _make_service(worker_ids=["qwen-1"])
    pt0 = PlanTask(
        task_id="stage-0",
        plan_version=1,
        stage_number=0,
        stage_name="Base",
        status=TaskStatus.COMPLETED,
    )
    pt1 = PlanTask(
        task_id="stage-1",
        plan_version=1,
        stage_number=1,
        stage_name="UsesBase",
        depends_on=[0],
        steps=[],
        agent_instructions=["backtests_runs(action='inspect', run_id='{{stage:0.missing_field}}')"],
    )
    svc._current_plan = ResearchPlan(version=1, goal="test", tasks=[pt0, pt1])

    # Stage 0 report exists but has no 'missing_field' column
    report_s0 = TaskReport(
        task_id="stage-0",
        worker_id="qwen-1",
        plan_version=1,
        status="success",
        results_table=[{"run_id": "r1"}],
        raw_output="{}",
    )
    svc._plan_store.load_reports_for_plan.return_value = [report_s0]

    svc._dispatch_plan_tasks([pt1])

    # Plan store should have a failure report for stage 1
    saved_reports = [
        call.kwargs.get("report") or call.args[0]
        for call in svc._plan_store.save_report.call_args_list
    ]
    assert any(
        r.status == "error" and "symbolic ref" in (r.error or "").lower()
        for r in saved_reports
    ), "Expected a symbolic-ref failure report to be saved"


def test_mcp_death_stop_condition() -> None:
    """Consecutive MCP health failures should trigger MCP_UNHEALTHY stop."""
    from app.scheduler import Scheduler
    state = OrchestratorState(goal="test")
    scheduler = Scheduler(max_mcp_failures=3)

    # No failures → None
    assert scheduler.should_stop_orchestrator(state) is None

    # 2 failures → not enough
    state.mcp_consecutive_failures = 2
    assert scheduler.should_stop_orchestrator(state) is None

    # 3 failures → stop
    state.mcp_consecutive_failures = 3
    result = scheduler.should_stop_orchestrator(state)
    assert result == "mcp_unhealthy"


def test_timed_out_plan_task_gets_retried() -> None:
    """Timed-out plan tasks should be reset to PENDING when retries remain."""
    svc, orch = _make_service(worker_ids=["qwen-1"])
    pt = PlanTask(
        task_id="stage-0",
        plan_version=1,
        stage_number=0,
        stage_name="SlowStage",
    )
    svc._current_plan = ResearchPlan(version=1, goal="test", tasks=[pt])

    task = Task(
        task_id="stage-0",
        description="slow",
        assigned_worker_id="qwen-1",
        attempts=1,
        metadata={"plan_mode": True, "plan_version": 1, "stage_number": 0},
    )
    task.mark_running()
    orch.state.add_task(task)

    pi = ProcessInfo(
        task_id="stage-0",
        worker_id="qwen-1",
        pid=42,
        started_at=(datetime.now(timezone.utc).replace(hour=0)).isoformat(),
    )
    orch.state.processes.append(pi)

    svc.worker_service.terminate_task = MagicMock()
    orch.state.remove_process = MagicMock()

    svc._check_timeouts()

    # Task should be PENDING (retry), not TIMED_OUT
    assert task.status == TaskStatus.PENDING, f"Task should be PENDING for retry, got {task.status}"
    assert pt.status == TaskStatus.PENDING, f"PlanTask should be PENDING for retry, got {pt.status}"
