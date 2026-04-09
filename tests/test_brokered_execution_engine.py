from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict
from time import monotonic
from types import SimpleNamespace

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.compiled_plan_store import CompiledPlanStore
from app.config import OrchestratorConfig
from app.execution_models import BaselineRef, BrokerHealth, ExecutionPlan, ExecutionStateV2, PlanSlice, ToolPolicy, ToolResultEnvelope, WorkerAction, WorkerReportableIssue
from app.models import StopReason, WorkerConfig
from app.orchestrator import Orchestrator
from app.raw_plan_models import CompileReport, CompiledPlanSequence, SemanticRawPlan, SemanticStage
from app.services.brokered_execution.engine import BrokeredExecutionService
from app.services.brokered_execution.worker import WorkerContractViolationError, WorkerParseFailureError


class _StubPlanner:
    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan
        self.calls = 0
        self.saved_statuses: list[str] = []

    async def create_plan(self, **_: object) -> ExecutionPlan:
        self.calls += 1
        return deepcopy(self.plan)

    def save_plan_snapshot(self, plan: ExecutionPlan) -> None:
        self.saved_statuses.append(plan.status)


class _StubWorker:
    def __init__(self, scripted: dict[str, list[dict[str, object]]]) -> None:
        self.scripted = {key: list(value) for key, value in scripted.items()}
        self.calls: list[dict[str, object]] = []

    async def choose_action(self, *, slice_obj: PlanSlice, remaining_budget: dict[str, int], checkpoint_summary: str, active_operation: dict[str, object], **_: object) -> WorkerAction:
        self.calls.append(
            {
                "slice_id": slice_obj.slice_id,
                "remaining_budget": dict(remaining_budget),
                "checkpoint_summary": checkpoint_summary,
                "active_operation": dict(active_operation),
            }
        )
        payload = self.scripted[slice_obj.slice_id].pop(0)
        return WorkerAction(
            action_id="action_test",
            action_type=str(payload.get("type")),
            tool=str(payload.get("tool", "")),
            arguments=dict(payload.get("arguments", {}) or {}),
            reason=str(payload.get("reason", "")),
            expected_evidence=list(payload.get("expected_evidence", []) or []),
            status=str(payload.get("status", "")),
            summary=str(payload.get("summary", "")),
            facts=dict(payload.get("facts", {}) or {}),
            artifacts=list(payload.get("artifacts", []) or []),
            pending_questions=list(payload.get("pending_questions", []) or []),
            reportable_issues=[
                WorkerReportableIssue(
                    summary=str(item.get("summary", "")),
                    severity=str(item.get("severity", "medium")),
                    details=str(item.get("details", "")),
                    affected_tool=str(item.get("affected_tool", "")),
                    category=str(item.get("category", "runtime")),
                )
                for item in list(payload.get("reportable_issues", []) or [])
            ],
            verdict=str(payload.get("verdict", "")),
            key_metrics=dict(payload.get("key_metrics", {}) or {}),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            reason_code=str(payload.get("reason_code", "")),
            retryable=bool(payload.get("retryable", False)),
        )


class _StubBroker:
    def __init__(self, scripted_calls: dict[tuple[str, str], list[ToolResultEnvelope]], *, delay_seconds: float = 0.0) -> None:
        self.scripted_calls = {key: list(value) for key, value in scripted_calls.items()}
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[str, dict[str, object], float, float]] = []
        self.recorded_issues: list[tuple[str, list[dict[str, object]]]] = []

    def validate_runtime_requirements(self) -> None:
        return None

    async def bootstrap(self) -> BrokerHealth:
        return BrokerHealth(
            endpoint_url="http://127.0.0.1:8766/mcp",
            bootstrapped_at="2026-04-09T00:00:00Z",
            session_id="session_1",
            tool_count=2,
            status="healthy",
            summary="bootstrapped",
        )

    def allowlist(self) -> set[str]:
        return {"events", "features_dataset"}

    def policy_for(self, tool_name: str) -> ToolPolicy:
        return ToolPolicy(tool_name=tool_name, expensive=False, autopoll_enabled=False)

    def policy_for_call(self, tool_name: str, arguments: dict[str, object]) -> ToolPolicy:
        del arguments
        return self.policy_for(tool_name)

    async def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> ToolResultEnvelope:
        started = monotonic()
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        key = (tool_name, repr(sorted(arguments.items())))
        result = self.scripted_calls[key].pop(0)
        finished = monotonic()
        self.calls.append((tool_name, dict(arguments), started, finished))
        return result

    async def record_worker_issues(self, issues, *, tool_name: str = "") -> None:
        self.recorded_issues.append((tool_name, list(issues)))

    async def report_incident(self, *, summary: str, error: str, affected_tool: str, metadata: dict[str, object], severity: str = "medium") -> None:
        self.recorded_issues.append(
            (
                affected_tool,
                [{"summary": summary, "details": error, "metadata": dict(metadata), "severity": severity}],
            )
        )

    async def close(self) -> None:
        return None


class _FailingPlanner:
    async def create_plan(self, **_: object):
        from app.services.brokered_execution.planner import PlannerDecisionError

        raise PlannerDecisionError("planner failed")

    def save_plan_snapshot(self, plan) -> None:
        del plan


class _FailingBroker(_StubBroker):
    async def call_tool(self, *, tool_name: str, arguments: dict[str, object]) -> ToolResultEnvelope:
        from app.broker.service import BrokerServiceError

        self.calls.append((tool_name, dict(arguments), monotonic(), monotonic()))
        raise BrokerServiceError("broker failed")


class _ExpensiveBroker(_StubBroker):
    def policy_for(self, tool_name: str) -> ToolPolicy:
        return ToolPolicy(tool_name=tool_name, expensive=True, autopoll_enabled=False)


class _ContractFailingWorker:
    def __init__(self, *, by_slice: dict[str, str]) -> None:
        self.by_slice = by_slice

    async def choose_action(self, *, slice_obj: PlanSlice, **_: object) -> WorkerAction:
        message = self.by_slice[slice_obj.slice_id]
        raise WorkerContractViolationError(
            message,
            artifact_path=f"/tmp/{slice_obj.slice_id}.json",
            parse_error=message,
            raw_output='{"type":"tool_call"}',
        )


class _ParseFailingWorker:
    def __init__(self, *, by_slice: dict[str, str]) -> None:
        self.by_slice = by_slice

    async def choose_action(self, *, slice_obj: PlanSlice, **_: object) -> WorkerAction:
        message = self.by_slice[slice_obj.slice_id]
        raise WorkerParseFailureError(
            message,
            artifact_path=f"/tmp/{slice_obj.slice_id}.json",
            parse_error=message,
            raw_output="Файл действия сохранён. Ожидаю следующий шаг.",
        )


class _StubConsoleController:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def __getattr__(self, name: str):
        if name.startswith("on_"):
            def _record(*args, **kwargs):
                payload = dict(kwargs)
                if args:
                    payload["args"] = list(args)
                self.events.append((name, payload))
            return _record
        raise AttributeError(name)


def _make_tool_result(
    *,
    call_id: str,
    tool: str,
    summary: str,
    ok: bool = True,
    response_status: str = "completed",
    operation_ref: str = "",
    resume_arguments: dict[str, object] | None = None,
) -> ToolResultEnvelope:
    return ToolResultEnvelope(
        call_id=call_id,
        tool=tool,
        ok=ok,
        retryable=False,
        duration_ms=10,
        summary=summary,
        key_facts={"summary": summary},
        artifact_ids=["artifact_1"] if ok else [],
        warnings=[],
        error_class="" if ok else "server_error",
        request_arguments={},
        response_status=response_status,
        tool_response_status=response_status if ok else "error",
        operation_ref=operation_ref,
        resume_arguments=resume_arguments or {},
    )


def _make_orchestrator(tmp_path, *, max_concurrent: int = 2) -> Orchestrator:
    cfg = OrchestratorConfig(
        goal="Test brokered execution",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        max_concurrent_plan_tasks=max_concurrent,
        max_plans_per_run=1,
        broker_autopoll_budget_seconds=0.0,
        decision_cycle_sleep_seconds=0.01,
    )
    cfg.workers = [WorkerConfig(worker_id=f"worker-{index}", role="executor", system_prompt="") for index in range(1, max_concurrent + 1)]
    orch = Orchestrator(
        config=cfg,
        planner_adapter=FakePlanner(responses=[], delay=0.0),
        worker_adapter=FakeWorker(responses=[], delay=0.0),
    )
    orch.notification_service = SimpleNamespace(
        send_worker_result=lambda *args, **kwargs: True,
        send_lifecycle=lambda *args, **kwargs: True,
        send_run_complete=lambda *args, **kwargs: True,
        flush=lambda: None,
    )
    orch.console_controller = _StubConsoleController()
    return orch


def _save_compiled_sequence(
    tmp_path,
    *,
    raw_name: str,
    plan_specs: list[tuple[str, list[PlanSlice]]],
) -> None:
    raw_dir = tmp_path / "raw_plans"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / raw_name).write_text(f"# {raw_name}\n", encoding="utf-8")
    stem = raw_name.rsplit(".", 1)[0]
    semantic = SemanticRawPlan(
        source_file=str(raw_dir / raw_name),
        source_hash=f"hash_{stem}",
        source_title=stem,
        goal=f"Goal for {stem}",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        stages=[
            SemanticStage(
                stage_id=f"stage_{index + 1}",
                title=plan_id,
                objective=plan_id,
                actions=["Execute stage"],
                success_criteria=["Stage done"],
            )
            for index, (plan_id, _slices) in enumerate(plan_specs)
        ],
    )
    plans = [
        ExecutionPlan(
            plan_id=plan_id,
            goal=semantic.goal,
            baseline_ref=semantic.baseline_ref,
            global_constraints=list(semantic.global_constraints),
            slices=slices,
        )
        for plan_id, slices in plan_specs
    ]
    sequence = CompiledPlanSequence(
        source_file=str(raw_dir / raw_name),
        source_hash=f"hash_{stem}",
        sequence_id=f"compiled_{stem}",
        semantic_plan=semantic,
        plans=plans,
        report=CompileReport(
            source_file=str(raw_dir / raw_name),
            source_hash=f"hash_{stem}",
            sequence_id=f"compiled_{stem}",
            compile_status="compiled",
            parser_confidence=0.8,
            semantic_method="llm",
            stage_count=len(plan_specs),
            compiled_plan_count=len(plan_specs),
        ),
    )
    CompiledPlanStore(tmp_path / "compiled_plans").save_sequence(sequence)


def test_brokered_engine_end_to_end_tool_call_then_final_report(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Validate funding signal",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Funding slice",
                hypothesis="funding can add signal value",
                objective="check funding data first",
                success_criteria=["catalog fetched"],
                allowed_tools=["events"],
                evidence_requirements=["catalog rows"],
                policy_tags=["cheap_first"],
                max_turns=3,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_1": [
                {
                    "type": "tool_call",
                    "tool": "events",
                    "arguments": {"view": "catalog", "family": "funding", "symbol": "BTCUSDT"},
                    "reason": "Need to verify funding data exists.",
                    "expected_evidence": ["catalog rows"],
                },
                {
                    "type": "final_report",
                    "summary": "Funding data is present and ready for feature validation.",
                    "verdict": "WATCHLIST",
                    "facts": {"funding_rows": 7147},
                    "artifacts": ["artifact_1"],
                    "key_metrics": {"funding_rows": 7147},
                    "confidence": 0.8,
                },
            ]
        }
    )
    broker = _StubBroker(
        {
            ("events", repr(sorted({"view": "catalog", "family": "funding", "symbol": "BTCUSDT"}.items()))): [
                _make_tool_result(call_id="tool_1", tool="events", summary="Funding catalog found, 7147 rows.")
            ]
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    active_plan = engine.state.find_plan("plan_1")
    assert active_plan is not None
    assert active_plan.status == "completed"
    assert active_plan.slices[0].status == "completed"
    assert [call[:2] for call in broker.calls] == [("events", {"view": "catalog", "family": "funding", "symbol": "BTCUSDT"})]
    assert worker.calls[0]["remaining_budget"]["turns_remaining"] == 3
    assert worker.calls[1]["remaining_budget"]["turns_remaining"] == 2
    assert engine.state.tool_call_ledger[0].plan_id == "plan_1"
    assert engine.state.tool_call_ledger[0].slice_id == "slice_1"
    assert any(name == "on_runtime_started" for name, _ in orch.console_controller.events)
    assert any(name == "on_slice_completed" for name, _ in orch.console_controller.events)


def test_brokered_engine_resume_skips_completed_tool_call(tmp_path) -> None:
    resumed_plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Resume funding slice",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        status="running",
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Funding slice",
                hypothesis="funding can add signal value",
                objective="close the slice after the catalog check",
                success_criteria=["final report emitted"],
                allowed_tools=["events"],
                evidence_requirements=["catalog rows"],
                policy_tags=["cheap_first"],
                max_turns=3,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="checkpointed",
                turn_count=1,
                tool_call_count=1,
                facts={"funding_rows": 7147},
                artifacts=["artifact_1"],
                last_summary="Funding catalog found, 7147 rows.",
                last_checkpoint_summary="Funding catalog found, 7147 rows.",
                latest_tool_result_summary="Funding catalog found, 7147 rows.",
            )
        ],
    )
    state = ExecutionStateV2(goal="Test brokered execution", status="running", plans=[resumed_plan], current_plan_id="plan_1")
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.execution_state = state
    orch.execution_store.save(state)
    assert orch.load_state() is True
    worker = _StubWorker(
        {
            "slice_1": [
                {
                    "type": "final_report",
                    "summary": "Resume completed without another tool call.",
                    "verdict": "PROMOTE",
                    "facts": {"funding_rows": 7147},
                    "confidence": 0.9,
                }
            ]
        }
    )
    broker = _StubBroker({})
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(resumed_plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert broker.calls == []
    assert worker.calls[0]["checkpoint_summary"] == "Funding catalog found, 7147 rows."


def test_brokered_engine_resumes_active_operation_before_new_worker_turn(tmp_path) -> None:
    resumed_plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Resume async operation",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        status="running",
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Async slice",
                hypothesis="async jobs should resume",
                objective="resume operation then finalize",
                success_criteria=["status polled"],
                allowed_tools=["features_dataset"],
                evidence_requirements=["operation status"],
                policy_tags=["async"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="checkpointed",
                turn_count=1,
                tool_call_count=1,
                active_operation_tool="features_dataset",
                active_operation_ref="op_123",
                active_operation_status="running",
                active_operation_arguments={"action": "status", "operation_id": "op_123"},
            )
        ],
    )
    state = ExecutionStateV2(goal="Resume async operation", status="running", plans=[resumed_plan], current_plan_id="plan_1")
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.execution_state = state
    worker = _StubWorker(
        {
            "slice_1": [
                {
                    "type": "final_report",
                    "summary": "Async operation completed and slice is done.",
                    "verdict": "WATCHLIST",
                    "confidence": 0.7,
                }
            ]
        }
    )
    broker = _StubBroker(
        {
            ("features_dataset", repr(sorted({"action": "status", "operation_id": "op_123"}.items()))): [
                _make_tool_result(
                    call_id="tool_status",
                    tool="features_dataset",
                    summary="Dataset build completed.",
                    response_status="completed",
                )
            ]
        }
    )
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(resumed_plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert broker.calls[0][1] == {"action": "status", "operation_id": "op_123"}
    assert worker.calls[0]["active_operation"] == {
        "tool": "",
        "ref": "",
        "status": "completed",
        "token": "",
        "tool_response_status": "completed",
    }


def test_brokered_engine_drain_waits_for_active_operation_then_stops(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Drain with in-flight op",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        status="running",
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Async slice",
                hypothesis="drain should wait for active ops",
                objective="poll status only",
                success_criteria=["operation terminalized"],
                allowed_tools=["features_dataset"],
                evidence_requirements=["operation status"],
                policy_tags=["async"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="checkpointed",
                active_operation_tool="features_dataset",
                active_operation_ref="op_123",
                active_operation_status="running",
                active_operation_arguments={"action": "status", "operation_id": "op_123"},
            )
        ],
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.execution_state = ExecutionStateV2(goal="Drain", status="running", plans=[plan], current_plan_id="plan_1")
    orch.request_drain()
    broker = _StubBroker(
        {
            ("features_dataset", repr(sorted({"action": "status", "operation_id": "op_123"}.items()))): [
                _make_tool_result(
                    call_id="tool_status_1",
                    tool="features_dataset",
                    summary="Still running.",
                    response_status="running",
                    operation_ref="op_123",
                    resume_arguments={"action": "status", "operation_id": "op_123"},
                ),
                _make_tool_result(
                    call_id="tool_status_2",
                    tool="features_dataset",
                    summary="Completed.",
                    response_status="completed",
                ),
            ]
        }
    )
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(plan), worker=_StubWorker({"slice_1": []}))

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GRACEFUL_STOP
    assert len(broker.calls) == 2


def test_brokered_engine_runs_two_slices_in_parallel(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_parallel",
        goal="Parallel slices",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Slice 1",
                hypothesis="parallelism works",
                objective="make one tool call then finish",
                success_criteria=["tool call done"],
                allowed_tools=["events"],
                evidence_requirements=["summary"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            ),
            PlanSlice(
                slice_id="slice_2",
                title="Slice 2",
                hypothesis="parallelism works",
                objective="make one tool call then finish",
                success_criteria=["tool call done"],
                allowed_tools=["events"],
                evidence_requirements=["summary"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=2,
            ),
        ],
    )
    worker = _StubWorker(
        {
            "slice_1": [
                {"type": "tool_call", "tool": "events", "arguments": {"view": "catalog", "symbol": "BTCUSDT"}, "reason": "slice 1"},
                {"type": "final_report", "summary": "slice 1 done", "verdict": "WATCHLIST", "confidence": 0.5},
            ],
            "slice_2": [
                {"type": "tool_call", "tool": "events", "arguments": {"view": "catalog", "symbol": "ETHUSDT"}, "reason": "slice 2"},
                {"type": "final_report", "summary": "slice 2 done", "verdict": "WATCHLIST", "confidence": 0.5},
            ],
        }
    )
    broker = _StubBroker(
        {
            ("events", repr(sorted({"view": "catalog", "symbol": "BTCUSDT"}.items()))): [
                _make_tool_result(call_id="tool_1", tool="events", summary="BTC done")
            ],
            ("events", repr(sorted({"view": "catalog", "symbol": "ETHUSDT"}.items()))): [
                _make_tool_result(call_id="tool_2", tool="events", summary="ETH done")
            ],
        },
        delay_seconds=0.05,
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=2)
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert len(broker.calls) == 2
    first = broker.calls[0]
    second = broker.calls[1]
    assert min(first[3], second[3]) > max(first[2], second[2])


def test_brokered_engine_terminalizes_contract_violations_without_looping(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_contract",
        goal="Contract failures",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_bad_1",
                title="Bad 1",
                hypothesis="bad worker output",
                objective="fail fast",
                success_criteria=["abort"],
                allowed_tools=["features_catalog"],
                evidence_requirements=["artifact"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            ),
            PlanSlice(
                slice_id="slice_bad_2",
                title="Bad 2",
                hypothesis="bad worker output",
                objective="fail fast",
                success_criteria=["abort"],
                allowed_tools=["features_catalog"],
                evidence_requirements=["artifact"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=2,
            ),
        ],
    )
    worker = _ContractFailingWorker(
        by_slice={
            "slice_bad_1": "tool_prefixed_namespace_forbidden:mcp__dev_space1__features_catalog",
            "slice_bad_2": "tool_prefixed_namespace_forbidden:mcp__dev_space1__features_catalog",
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=2)
    engine = BrokeredExecutionService(orch, broker=_StubBroker({}), planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_IMPOSSIBLE
    active_plan = engine.state.find_plan("plan_contract")
    assert active_plan is not None
    assert active_plan.status == "failed"
    assert all(slice_obj.status == "aborted" for slice_obj in active_plan.slices)
    assert len(engine.state.turn_history) == 2


def test_brokered_engine_terminalizes_parse_failures_without_looping(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_parse",
        goal="Parse failures",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_bad_1",
                title="Bad 1",
                hypothesis="worker emits prose",
                objective="fail fast",
                success_criteria=["abort"],
                allowed_tools=["features_catalog"],
                evidence_requirements=["artifact"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            ),
            PlanSlice(
                slice_id="slice_bad_2",
                title="Bad 2",
                hypothesis="worker emits prose",
                objective="fail fast",
                success_criteria=["abort"],
                allowed_tools=["features_catalog"],
                evidence_requirements=["artifact"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=2,
            ),
        ],
    )
    worker = _ParseFailingWorker(
        by_slice={
            "slice_bad_1": "json_object_not_found",
            "slice_bad_2": "json_object_not_found",
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=2)
    engine = BrokeredExecutionService(orch, broker=_StubBroker({}), planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_IMPOSSIBLE
    active_plan = engine.state.find_plan("plan_parse")
    assert active_plan is not None
    assert active_plan.status == "failed"
    assert all(slice_obj.status == "aborted" for slice_obj in active_plan.slices)
    assert all(slice_obj.last_error == "worker_parse_failure" for slice_obj in active_plan.slices)
    assert len(engine.state.turn_history) == 2


def test_brokered_engine_stops_on_max_errors_total(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.config.max_errors_total = 1
    engine = BrokeredExecutionService(
        orch,
        broker=_StubBroker({}, delay_seconds=0.0),
        planner=_FailingPlanner(),
        worker=_StubWorker({}),
    )

    reason = asyncio.run(engine.run())

    assert reason == StopReason.MAX_ERRORS


def test_brokered_engine_stops_on_max_broker_failures(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_fail",
        goal="Broker failure",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Failing slice",
                hypothesis="broker fails",
                objective="emit one failing tool call",
                success_criteria=["tool call attempted"],
                allowed_tools=["events"],
                evidence_requirements=["summary"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_1": [
                {"type": "tool_call", "tool": "events", "arguments": {"view": "catalog", "symbol": "BTCUSDT"}, "reason": "fail it"},
            ]
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.config.max_broker_failures = 1
    engine = BrokeredExecutionService(
        orch,
        broker=_FailingBroker({}, delay_seconds=0.0),
        planner=_StubPlanner(plan),
        worker=worker,
    )

    reason = asyncio.run(engine.run())

    assert reason == StopReason.MCP_UNHEALTHY


def test_brokered_engine_suppresses_contradictory_worker_registry_issues(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_issues",
        goal="Issue suppression",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Issue slice",
                hypothesis="worker may hallucinate registry failure",
                objective="finish cleanly",
                success_criteria=["final report emitted"],
                allowed_tools=["events"],
                evidence_requirements=["summary"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_1": [
                {
                    "type": "final_report",
                    "summary": "Completed after successful broker call.",
                    "verdict": "WATCHLIST",
                    "reportable_issues": [
                        {
                            "summary": "events not found in tool registry",
                            "details": "events tool not found in registry",
                            "affected_tool": "events",
                        }
                    ],
                }
            ]
        }
    )
    broker = _StubBroker({})
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert broker.recorded_issues == [("", [])]


def test_brokered_engine_auto_extends_expensive_budget_for_allowed_tool(tmp_path) -> None:
    plan = ExecutionPlan(
        plan_id="plan_expensive",
        goal="Expensive budget extension",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Expensive slice",
                hypothesis="One extra expensive call should not abort the slice",
                objective="Start one allowed expensive sync operation",
                success_criteria=["sync starts"],
                allowed_tools=["events_sync"],
                evidence_requirements=["operation ref"],
                policy_tags=["cheap_first"],
                max_turns=2,
                max_tool_calls=2,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_1": [
                {
                    "type": "tool_call",
                    "tool": "events_sync",
                    "arguments": {"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "started"},
                    "reason": "Need sync first.",
                },
                {
                    "type": "final_report",
                    "summary": "Sync started successfully.",
                    "verdict": "WATCHLIST",
                },
            ]
        }
    )
    broker = _ExpensiveBroker(
        {
            ("events_sync", repr(sorted({"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "started"}.items()))): [
                _make_tool_result(
                    call_id="call_sync",
                    tool="events_sync",
                    summary="Sync started.",
                    response_status="completed",
                )
            ]
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    engine = BrokeredExecutionService(orch, broker=broker, planner=_StubPlanner(plan), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    active_plan = engine.state.find_plan("plan_expensive")
    assert active_plan is not None
    slice_state = active_plan.slices[0]
    assert slice_state.max_expensive_calls == 1
    assert broker.calls


def test_brokered_engine_compiled_raw_consumes_batches_in_raw_file_order(tmp_path) -> None:
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v1.md",
        plan_specs=[
            (
                "compiled_plan_v1_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_v1_a",
                        title="Slice A",
                        hypothesis="A",
                        objective="A",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            ),
            (
                "compiled_plan_v1_batch_2",
                [
                    PlanSlice(
                        slice_id="slice_v1_b",
                        title="Slice B",
                        hypothesis="B",
                        objective="B",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            ),
        ],
    )
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v10.md",
        plan_specs=[
            (
                "compiled_plan_v10_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_v10_a",
                        title="Slice C",
                        hypothesis="C",
                        objective="C",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            )
        ],
    )
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v2.md",
        plan_specs=[
            (
                "compiled_plan_v2_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_v2_a",
                        title="Slice D",
                        hypothesis="D",
                        objective="D",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_v1_a": [{"type": "final_report", "summary": "v1a", "verdict": "WATCHLIST"}],
            "slice_v1_b": [{"type": "final_report", "summary": "v1b", "verdict": "WATCHLIST"}],
            "slice_v2_a": [{"type": "final_report", "summary": "v2a", "verdict": "WATCHLIST"}],
            "slice_v10_a": [{"type": "final_report", "summary": "v10a", "verdict": "WATCHLIST"}],
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.config.plan_source = "compiled_raw"
    orch.config.raw_plan_dir = str(tmp_path / "raw_plans")
    orch.config.compiled_plan_dir = str(tmp_path / "compiled_plans")
    engine = BrokeredExecutionService(orch, broker=_StubBroker({}), planner=_FailingPlanner(), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert [plan.plan_id for plan in engine.state.plans] == [
        "compiled_plan_v1_batch_1",
        "compiled_plan_v1_batch_2",
        "compiled_plan_v2_batch_1",
        "compiled_plan_v10_batch_1",
    ]


def test_brokered_engine_compiled_raw_skips_failed_sequence_and_moves_to_next(tmp_path) -> None:
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v1.md",
        plan_specs=[
            (
                "compiled_plan_v1_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_fail",
                        title="Slice fail",
                        hypothesis="fail",
                        objective="fail",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            ),
            (
                "compiled_plan_v1_batch_2",
                [
                    PlanSlice(
                        slice_id="slice_skipped",
                        title="Slice skipped",
                        hypothesis="skip",
                        objective="skip",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            ),
        ],
    )
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v2.md",
        plan_specs=[
            (
                "compiled_plan_v2_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_ok",
                        title="Slice ok",
                        hypothesis="ok",
                        objective="ok",
                        success_criteria=["done"],
                        allowed_tools=["events"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    )
                ],
            )
        ],
    )
    worker = _StubWorker(
        {
            "slice_fail": [{"type": "abort", "summary": "failed early", "reason_code": "test_failure"}],
            "slice_ok": [{"type": "final_report", "summary": "completed", "verdict": "WATCHLIST"}],
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=1)
    orch.config.plan_source = "compiled_raw"
    orch.config.raw_plan_dir = str(tmp_path / "raw_plans")
    orch.config.compiled_plan_dir = str(tmp_path / "compiled_plans")
    orch.config.compiled_queue_skip_failures = True
    engine = BrokeredExecutionService(orch, broker=_StubBroker({}), planner=_FailingPlanner(), worker=worker)

    reason = asyncio.run(engine.run())

    assert reason == StopReason.GOAL_REACHED
    assert [plan.plan_id for plan in engine.state.plans] == [
        "compiled_plan_v1_batch_1",
        "compiled_plan_v2_batch_1",
    ]


def test_brokered_engine_infers_missing_project_id_from_previous_slice_facts(tmp_path) -> None:
    _save_compiled_sequence(
        tmp_path,
        raw_name="plan_v1.md",
        plan_specs=[
            (
                "compiled_plan_v1_batch_1",
                [
                    PlanSlice(
                        slice_id="slice_setup",
                        title="Setup",
                        hypothesis="setup",
                        objective="setup",
                        success_criteria=["done"],
                        allowed_tools=["research_project"],
                        evidence_requirements=["done"],
                        policy_tags=["setup"],
                        max_turns=1,
                        max_tool_calls=0,
                        max_expensive_calls=0,
                        parallel_slot=1,
                    ),
                    PlanSlice(
                        slice_id="slice_dep",
                        title="Dependent",
                        hypothesis="dep",
                        objective="dep",
                        success_criteria=["done"],
                        allowed_tools=["research_search"],
                        evidence_requirements=["done"],
                        policy_tags=["analysis"],
                        max_turns=1,
                        max_tool_calls=1,
                        max_expensive_calls=0,
                        parallel_slot=1,
                        depends_on=["slice_setup"],
                    ),
                ],
            ),
        ],
    )
    worker = _StubWorker(
        {
            "slice_setup": [
                {
                    "type": "final_report",
                    "summary": "setup done",
                    "verdict": "WATCHLIST",
                    "facts": {"project.project_id": "proj_123"},
                }
            ],
            "slice_dep": [
                {
                    "type": "tool_call",
                    "tool": "research_search",
                    "arguments": {"query": "orthogonal funding"},
                    "reason": "search project memory",
                }
            ],
        }
    )
    broker = _StubBroker(
        {
            ("research_search", repr(sorted({"project_id": "proj_123", "query": "orthogonal funding"}.items()))): [
                _make_tool_result(call_id="call_search", tool="research_search", summary="search ok")
            ]
        }
    )
    orch = _make_orchestrator(tmp_path, max_concurrent=2)
    orch.config.plan_source = "compiled_raw"
    orch.config.raw_plan_dir = str(tmp_path / "raw_plans")
    orch.config.compiled_plan_dir = str(tmp_path / "compiled_plans")
    engine = BrokeredExecutionService(orch, broker=broker, planner=_FailingPlanner(), worker=worker)

    asyncio.run(engine._create_next_plan())
    plan = engine.state.active_plan()
    assert plan is not None
    asyncio.run(engine._run_plan_round(plan))
    asyncio.run(engine._run_plan_round(plan))

    assert broker.calls[-1][0] == "research_search"
    assert broker.calls[-1][1]["project_id"] == "proj_123"
