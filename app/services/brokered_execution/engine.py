"""
Async brokered execution engine.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import asdict
from typing import Any

from app.broker import BrokerService, BrokerServiceError
from app.compiled_plan_store import CompiledPlanStore
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionPlan, ExecutionStateV2, ExecutionTurn, PlanSlice, WorkerAction, WorkerReportableIssue, make_id
from app.execution_store import ExecutionStateStore
from app.models import OrchestratorEvent, StopReason, TaskResult
from app.plan_sources import CompiledPlanSource, PlanSource, PlannerPlanSource
from app.runtime_console import slot_for_slice
from app.runtime_incidents import LocalIncidentStore
from app.services.brokered_execution.budgeting import maybe_extend_expensive_budget
from app.services.brokered_execution.planner import PlannerDecisionError, PlannerDecisionService
from app.services.brokered_execution.worker import (
    WorkerContractViolationError,
    WorkerDecisionError,
    WorkerDecisionService,
    WorkerTerminalActionError,
)

logger = logging.getLogger("orchestrator.brokered")

_RUNNING_STATUSES = {"queued", "pending", "running", "started", "in_progress"}


class BrokeredExecutionService:
    def __init__(
        self,
        orch: Any,
        *,
        broker: BrokerService | None = None,
        planner: PlannerDecisionService | None = None,
        worker: WorkerDecisionService | None = None,
        plan_source: PlanSource | None = None,
    ) -> None:
        self.orch = orch
        self.config = orch.config
        self.state: ExecutionStateV2 = orch.execution_state
        self.state_store: ExecutionStateStore = orch.execution_store
        self.artifact_store: ExecutionArtifactStore = orch.artifact_store
        self.notification_service = orch.notification_service
        self.console_controller = getattr(orch, "console_controller", None)
        self.worker_ids = [worker_cfg.worker_id for worker_cfg in self.config.workers] or ["worker-1"]
        self.max_slots = max(1, min(3, len(self.worker_ids), int(self.config.max_concurrent_plan_tasks or 1)))
        self.incident_store = LocalIncidentStore(self.config.state_dir, run_id=self.config.current_run_id)
        self.broker = broker or BrokerService(
            transport_config=self.config.broker,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            autopoll_budget_seconds=self.config.broker_autopoll_budget_seconds,
            autopoll_interval_seconds=self.config.broker_autopoll_interval_seconds,
            console_controller=self.console_controller,
        )
        self.planner = planner
        if self.planner is None and self.config.plan_source == "planner":
            self.planner = PlannerDecisionService(
                adapter=orch.planner_adapter,
                artifact_store=self.artifact_store,
                timeout_seconds=self.config.planner_decision_timeout_seconds,
                retry_attempts=self.config.planner_decision_retry_attempts,
                retry_backoff_seconds=self.config.decision_retry_backoff_seconds,
                operator_directives=self.config.operator_directives,
                observer=self.console_controller,
            )
        self.worker = worker or WorkerDecisionService(
            adapter=orch.worker_adapter,
            artifact_store=self.artifact_store,
            timeout_seconds=self.config.worker_decision_timeout_seconds,
            retry_attempts=self.config.worker_decision_retry_attempts,
            retry_backoff_seconds=self.config.decision_retry_backoff_seconds,
            worker_system_prompt=self.config.worker_system_prompt,
        )
        self.plan_source = plan_source or self._build_plan_source()
        self._save_lock = asyncio.Lock()

    def _build_plan_source(self) -> PlanSource:
        if self.config.plan_source == "compiled_raw":
            return CompiledPlanSource(
                store=CompiledPlanStore(self.config.compiled_plan_dir),
                raw_plan_dir=self.config.raw_plan_dir,
                skip_failures=bool(self.config.compiled_queue_skip_failures),
                notification_service=self.notification_service,
            )
        assert self.planner is not None
        return PlannerPlanSource(
            planner=self.planner,
            goal=self.config.goal,
            baseline_bootstrap=self.config.research_config,
            max_worker_count=self.max_slots,
            max_plans_per_run=self.config.max_plans_per_run,
            broker_allowlist_provider=self.broker.allowlist,
            state_summary_provider=self._state_summary,
        )

    async def run(self) -> StopReason:
        self.orch.state.status = "running"
        self.state.status = "running"
        if self.console_controller is not None:
            self.console_controller.on_runtime_started(goal=self.config.goal)
        await self._save_state()
        try:
            self.broker.validate_runtime_requirements()
            self.state.broker_health = await self.broker.bootstrap()
            if self.console_controller is not None:
                self.console_controller.on_broker_bootstrap(health=self.state.broker_health)
            await self._save_state()
        except (Exception, asyncio.CancelledError) as exc:
            logger.exception("Broker bootstrap failed")
            self._record_runtime_incident("broker_bootstrap_failed", {"error": str(exc)})
            await self._finish(StopReason.SUBPROCESS_ERROR, f"broker_bootstrap_failed: {exc}")
            return StopReason.SUBPROCESS_ERROR

        try:
            while True:
                try:
                    if self.orch._stop_requested:
                        await self._finish(StopReason.GRACEFUL_STOP, "stop_requested")
                        return StopReason.GRACEFUL_STOP
                    plan = self.state.active_plan()
                    if plan is None or plan.is_terminal:
                        if self.orch._drain_mode:
                            await self._finish(StopReason.GRACEFUL_STOP, self.plan_source.summary())
                            return StopReason.GRACEFUL_STOP
                        plan = await self._create_next_plan()
                        if plan is None:
                            stop_reason = self.plan_source.stop_reason(self.state, drain_mode=self.orch._drain_mode) or StopReason.GOAL_IMPOSSIBLE
                            await self._finish(stop_reason, self.plan_source.summary())
                            return stop_reason
                    did_work = await self._run_plan_round(plan)
                    self._finalize_plan_if_ready(plan)
                    if did_work:
                        self.state.no_progress_cycles = 0
                    else:
                        self.state.no_progress_cycles += 1
                    await self._save_state()
                    stop_reason = self._stop_reason_after_round(plan, did_work=did_work)
                    if stop_reason is not None:
                        await self._finish(stop_reason, f"plan={plan.plan_id} status={plan.status}")
                        return stop_reason
                    if not did_work:
                        await asyncio.sleep(max(0.05, float(self.config.decision_cycle_sleep_seconds or 0.25)))
                except (PlannerDecisionError, WorkerDecisionError, BrokerServiceError) as exc:
                    await self._handle_runtime_error(exc)
                    if self.orch.state.total_errors >= int(self.config.max_errors_total or 1):
                        await self._finish(StopReason.MAX_ERRORS, str(exc))
                        return StopReason.MAX_ERRORS
                    if self.state.broker_failure_count >= int(self.config.max_broker_failures or 1):
                        await self._finish(StopReason.MCP_UNHEALTHY, str(exc))
                        return StopReason.MCP_UNHEALTHY
                    await asyncio.sleep(max(0.05, float(self.config.decision_cycle_sleep_seconds or 0.25)))
        finally:
            await self.broker.close()

    async def _create_next_plan(self) -> ExecutionPlan | None:
        plan = await self.plan_source.next_plan_batch(self.state)
        if plan is None:
            return None
        plan.status = "running"
        self.state.plans.append(plan)
        self.state.current_plan_id = plan.plan_id
        self.artifact_store.save_plan(plan)
        if self.config.plan_source == "planner":
            await self._capture_planner_surface_assumptions(plan)
        if self.console_controller is not None:
            self.console_controller.on_plan_created(plan_id=plan.plan_id)
        logger.info("Created execution plan %s with %d slices", plan.plan_id, len(plan.slices))
        return plan

    async def _run_plan_round(self, plan: ExecutionPlan) -> bool:
        candidate_slices = [
            slice_obj
            for slice_obj in sorted(plan.slices, key=lambda item: (item.parallel_slot, item.slice_id))
            if not slice_obj.is_terminal and slice_obj.parallel_slot <= self.max_slots
        ]
        if not candidate_slices:
            return False
        semaphore = asyncio.Semaphore(self.max_slots)
        tasks = [asyncio.create_task(self._process_slice_round(plan, slice_obj, semaphore)) for slice_obj in candidate_slices]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        did_work = False
        for result in results:
            if isinstance(result, Exception):
                await self._handle_runtime_error(result)
                continue
            did_work = bool(result) or did_work
        return did_work

    async def _process_slice_round(self, plan: ExecutionPlan, slice_obj: PlanSlice, semaphore: asyncio.Semaphore) -> bool:
        async with semaphore:
            if slice_obj.is_terminal:
                return False
            if slice_obj.active_operation_ref and slice_obj.active_operation_status.lower() in _RUNNING_STATUSES:
                return await self._resume_active_operation(plan, slice_obj)
            if self.orch._drain_mode:
                return False
            return await self._execute_slice_turn(plan, slice_obj)

    async def _execute_slice_turn(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        if slice_obj.turn_count >= slice_obj.max_turns:
            self._abort_slice(
                plan,
                slice_obj,
                summary="Slice turn budget exhausted before terminal report.",
                reason_code="slice_turn_budget_exhausted",
                retryable=False,
            )
            return True
        worker_id = self.worker_ids[(slice_obj.parallel_slot - 1) % len(self.worker_ids)]
        slice_obj.assigned_worker_id = worker_id
        if self.console_controller is not None:
            self.console_controller.on_slice_turn_started(
                slot=slot_for_slice(slice_obj.parallel_slot),
                plan_id=plan.plan_id,
                slice_id=slice_obj.slice_id,
                title=slice_obj.title,
                worker_id=worker_id,
                turns_used=slice_obj.turn_count,
                turns_total=slice_obj.max_turns,
                tool_calls_used=slice_obj.tool_call_count,
                tool_calls_total=slice_obj.max_tool_calls,
                summary=slice_obj.last_summary or slice_obj.last_checkpoint_summary,
                operation_ref=slice_obj.active_operation_ref,
                operation_status=slice_obj.active_operation_status,
            )
        try:
            action = await self.worker.choose_action(
                plan_id=plan.plan_id,
                slice_obj=slice_obj,
                baseline_bootstrap=self.config.research_config,
                known_facts=slice_obj.facts,
                recent_turn_summaries=self._recent_turn_summaries(plan.plan_id, slice_obj.slice_id),
                latest_tool_summary=slice_obj.latest_tool_result_summary or slice_obj.last_summary,
                remaining_budget=self._remaining_budget(slice_obj),
                checkpoint_summary=slice_obj.last_checkpoint_summary,
                active_operation=self._active_operation_payload(slice_obj),
            )
        except WorkerTerminalActionError as exc:
            await self._terminalize_worker_action_error(plan, slice_obj, worker_id=worker_id, exc=exc)
            return True
        action = self._sanitize_worker_action(slice_obj, action)
        await self.broker.record_worker_issues([asdict(item) for item in action.reportable_issues], tool_name=action.tool)
        turn = ExecutionTurn(
            turn_id=make_id("turn"),
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            worker_id=worker_id,
            turn_index=slice_obj.turn_count + 1,
            action=action,
        )
        self.artifact_store.save_turn_action(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload=action,
        )
        slice_obj.turn_count += 1
        slice_obj.status = "running"
        if action.action_type == "tool_call":
            await self._execute_tool_call(plan, slice_obj, turn)
        elif action.action_type == "checkpoint":
            self._apply_checkpoint(plan, slice_obj, turn)
        elif action.action_type == "final_report":
            self._apply_final_report(plan, slice_obj, turn)
        else:
            self._apply_abort(plan, slice_obj, turn)
        self.state.turn_history.append(turn)
        plan.touch()
        self._sync_runtime_summary(turn)
        return True

    async def _execute_tool_call(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        if slice_obj.tool_call_count >= slice_obj.max_tool_calls:
            self._abort_slice(
                plan,
                slice_obj,
                summary="Slice tool-call budget exhausted before the requested tool call.",
                reason_code="slice_tool_budget_exhausted",
                retryable=False,
            )
            return
        policy_resolver = getattr(self.broker, "policy_for_call", None)
        if callable(policy_resolver):
            policy = policy_resolver(turn.action.tool, turn.action.arguments)
        else:
            policy = self.broker.policy_for(turn.action.tool)
        if policy.expensive and slice_obj.expensive_call_count >= slice_obj.max_expensive_calls:
            extended_budget = maybe_extend_expensive_budget(
                allowed_tools=slice_obj.allowed_tools,
                current_budget=slice_obj.max_expensive_calls,
                requested_tool=turn.action.tool,
            )
            if extended_budget > slice_obj.max_expensive_calls:
                previous_budget = slice_obj.max_expensive_calls
                slice_obj.max_expensive_calls = extended_budget
                await self.broker.report_incident(
                    summary="planner_expensive_budget_underestimated",
                    error=(
                        f"Slice {slice_obj.slice_id} requested allowed expensive tool {turn.action.tool} "
                        f"after exhausting budget {previous_budget}; broker auto-extended budget to {extended_budget}."
                    ),
                    affected_tool=turn.action.tool,
                    metadata={
                        "plan_id": plan.plan_id,
                        "slice_id": slice_obj.slice_id,
                        "previous_budget": previous_budget,
                        "extended_budget": extended_budget,
                        "requested_tool": turn.action.tool,
                    },
                    severity="medium",
                )
            else:
                self._abort_slice(
                    plan,
                    slice_obj,
                    summary="Slice expensive-call budget exhausted before the requested tool call.",
                    reason_code="slice_expensive_budget_exhausted",
                    retryable=False,
                )
                return
        result = await self._broker_call_tool(
            tool_name=turn.action.tool,
            arguments=turn.action.arguments,
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            slot=slot_for_slice(slice_obj.parallel_slot),
            phase="call",
        )
        self._stamp_result_context(result, plan_id=plan.plan_id, slice_id=slice_obj.slice_id)
        turn.tool_result = result
        self.state.tool_call_ledger.append(result)
        slice_obj.tool_call_count += 1
        if policy.expensive:
            slice_obj.expensive_call_count += 1
        self._apply_tool_result(slice_obj, result, tool_name=turn.action.tool)
        if not result.ok and not result.retryable:
            slice_obj.status = "failed"

    async def _resume_active_operation(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        resume_tool = slice_obj.active_resume_tool or slice_obj.active_operation_tool
        if not resume_tool or not slice_obj.active_operation_arguments:
            slice_obj.active_operation_ref = ""
            slice_obj.active_operation_status = ""
            slice_obj.active_operation_tool = ""
            slice_obj.active_operation_arguments = {}
            slice_obj.active_resume_tool = ""
            slice_obj.active_resume_token = ""
            return False
        result = await self._broker_call_tool(
            tool_name=resume_tool,
            arguments=slice_obj.active_operation_arguments,
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            slot=slot_for_slice(slice_obj.parallel_slot),
            phase="resume",
        )
        self._stamp_result_context(result, plan_id=plan.plan_id, slice_id=slice_obj.slice_id)
        self.state.tool_call_ledger.append(result)
        self._apply_tool_result(slice_obj, result, tool_name=resume_tool)
        plan.touch()
        return True

    def _apply_tool_result(self, slice_obj: PlanSlice, result: Any, *, tool_name: str) -> None:
        slice_obj.last_summary = result.summary
        slice_obj.latest_tool_result_summary = result.summary
        slice_obj.last_tool_response_status = result.tool_response_status
        slice_obj.artifacts.extend(item for item in result.artifact_ids if item not in slice_obj.artifacts)
        slice_obj.facts.update(result.key_facts)
        slice_obj.last_error = result.summary if not result.ok else ""
        if result.operation_ref and result.response_status.lower() in _RUNNING_STATUSES:
            slice_obj.status = "checkpointed"
            slice_obj.active_operation_tool = tool_name
            slice_obj.active_operation_ref = result.operation_ref
            slice_obj.active_operation_status = result.response_status
            slice_obj.active_operation_arguments = dict(result.resume_arguments)
            slice_obj.active_resume_tool = result.resume_tool or tool_name
            slice_obj.active_resume_token = result.resume_token or result.operation_ref
            slice_obj.last_checkpoint_summary = result.summary
            if self.console_controller is not None:
                self.console_controller.on_slice_checkpoint(
                    slot=slot_for_slice(slice_obj.parallel_slot),
                    summary=result.summary,
                    operation_ref=result.operation_ref,
                    operation_status=result.response_status,
                )
            return
        slice_obj.active_operation_tool = ""
        slice_obj.active_operation_ref = ""
        slice_obj.active_operation_status = result.response_status
        slice_obj.active_operation_arguments = {}
        slice_obj.active_resume_tool = ""
        slice_obj.active_resume_token = ""
        if not result.ok and result.retryable:
            slice_obj.status = "checkpointed"
            slice_obj.last_checkpoint_summary = result.summary
            if self.console_controller is not None:
                self.console_controller.on_slice_checkpoint(
                    slot=slot_for_slice(slice_obj.parallel_slot),
                    summary=result.summary,
                    operation_ref=result.operation_ref,
                    operation_status=result.response_status,
                )
        elif not result.ok:
            slice_obj.status = "failed"
            if self.console_controller is not None:
                self.console_controller.on_slice_failed(slot=slot_for_slice(slice_obj.parallel_slot), summary=result.summary)
        else:
            slice_obj.status = "checkpointed"
            if self.console_controller is not None:
                self.console_controller.on_slice_checkpoint(
                    slot=slot_for_slice(slice_obj.parallel_slot),
                    summary=result.summary,
                    operation_ref=result.operation_ref,
                    operation_status=result.response_status,
                )

    def _apply_checkpoint(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        slice_obj.status = "checkpointed"
        slice_obj.last_checkpoint_turn_id = turn.turn_id
        slice_obj.last_checkpoint_summary = turn.action.summary
        slice_obj.last_summary = turn.action.summary
        slice_obj.facts.update(turn.action.facts)
        slice_obj.artifacts.extend(item for item in turn.action.artifacts if item not in slice_obj.artifacts)
        self.artifact_store.save_report(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload={"type": "checkpoint", **asdict(turn.action)},
        )
        if self.console_controller is not None:
            self.console_controller.on_slice_checkpoint(
                slot=slot_for_slice(slice_obj.parallel_slot),
                summary=turn.action.summary,
                operation_ref=slice_obj.active_operation_ref,
                operation_status=slice_obj.active_operation_status,
            )

    def _apply_final_report(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        slice_obj.status = "completed"
        slice_obj.final_report_turn_id = turn.turn_id
        slice_obj.last_summary = turn.action.summary
        slice_obj.latest_tool_result_summary = turn.action.summary
        slice_obj.facts.update(turn.action.facts)
        slice_obj.artifacts.extend(item for item in turn.action.artifacts if item not in slice_obj.artifacts)
        slice_obj.active_operation_tool = ""
        slice_obj.active_operation_ref = ""
        slice_obj.active_operation_status = ""
        slice_obj.active_operation_arguments = {}
        slice_obj.active_resume_tool = ""
        slice_obj.active_resume_token = ""
        self.artifact_store.save_report(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload={"type": "final_report", **asdict(turn.action)},
        )
        if self.console_controller is not None:
            self.console_controller.on_slice_completed(slot=slot_for_slice(slice_obj.parallel_slot), summary=turn.action.summary)
        self._emit_slice_notification(plan, slice_obj, turn.action)

    def _apply_abort(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        slice_obj.status = "aborted"
        slice_obj.last_error = turn.action.reason_code
        slice_obj.last_summary = turn.action.summary
        slice_obj.active_operation_tool = ""
        slice_obj.active_operation_ref = ""
        slice_obj.active_operation_status = ""
        slice_obj.active_operation_arguments = {}
        slice_obj.active_resume_tool = ""
        slice_obj.active_resume_token = ""
        self.artifact_store.save_report(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload={"type": "abort", **asdict(turn.action)},
        )
        if self.console_controller is not None:
            self.console_controller.on_slice_aborted(slot=slot_for_slice(slice_obj.parallel_slot), summary=turn.action.summary)
        self._emit_slice_notification(plan, slice_obj, turn.action)

    def _abort_slice(
        self,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        *,
        summary: str,
        reason_code: str,
        retryable: bool,
    ) -> None:
        action = WorkerAction(
            action_id=make_id("action"),
            action_type="abort",
            summary=summary,
            reason_code=reason_code,
            retryable=retryable,
        )
        turn = ExecutionTurn(
            turn_id=make_id("turn"),
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            worker_id=slice_obj.assigned_worker_id or self.worker_ids[0],
            turn_index=slice_obj.turn_count + 1,
            action=action,
        )
        slice_obj.turn_count += 1
        self._apply_abort(plan, slice_obj, turn)
        self.state.turn_history.append(turn)

    def _finalize_plan_if_ready(self, plan: ExecutionPlan) -> None:
        if any(not slice_obj.is_terminal for slice_obj in plan.slices):
            return
        if all(slice_obj.status == "completed" for slice_obj in plan.slices):
            plan.status = "completed"
            self.state.completed_plan_count += 1
            self.state.consecutive_failed_plans = 0
            self.plan_source.mark_plan_complete(plan, self.state)
        else:
            plan.status = "failed"
            self.state.consecutive_failed_plans += 1
            self.plan_source.mark_plan_failed(plan, self.state)
        plan.touch()

    def _remaining_budget(self, slice_obj: PlanSlice) -> dict[str, int]:
        return {
            "turns_used": slice_obj.turn_count,
            "turns_remaining": max(0, slice_obj.max_turns - slice_obj.turn_count),
            "tool_calls_used": slice_obj.tool_call_count,
            "tool_calls_remaining": max(0, slice_obj.max_tool_calls - slice_obj.tool_call_count),
            "expensive_calls_used": slice_obj.expensive_call_count,
            "expensive_calls_remaining": max(0, slice_obj.max_expensive_calls - slice_obj.expensive_call_count),
        }

    @staticmethod
    def _active_operation_payload(slice_obj: PlanSlice) -> dict[str, Any]:
        return {
            "tool": slice_obj.active_resume_tool or slice_obj.active_operation_tool,
            "ref": slice_obj.active_operation_ref,
            "status": slice_obj.active_operation_status,
            "token": slice_obj.active_resume_token,
            "tool_response_status": slice_obj.last_tool_response_status,
        }

    def _state_summary(self) -> str:
        lines: list[str] = []
        for plan in self.state.plans[-3:]:
            lines.append(f"{plan.plan_id}: {plan.status}")
            for slice_obj in plan.slices:
                line = f"- {slice_obj.slice_id} {slice_obj.status}"
                if slice_obj.last_summary:
                    line += f" | {slice_obj.last_summary[:180]}"
                if slice_obj.last_error:
                    line += f" | error={slice_obj.last_error[:120]}"
                lines.append(line)
        for blocker in self._recent_blockers()[-4:]:
            lines.append(f"! blocker: {blocker[:220]}")
        return "\n".join(lines[-12:])

    def _recent_turn_summaries(self, plan_id: str, slice_id: str) -> list[str]:
        summaries: list[str] = []
        for turn in self.state.turn_history:
            if turn.plan_id != plan_id or turn.slice_id != slice_id:
                continue
            if turn.tool_result is not None:
                summaries.append(f"{turn.action.tool}: {turn.tool_result.summary}")
            elif turn.action.summary:
                summaries.append(f"{turn.action.action_type}: {turn.action.summary}")
        return summaries[-6:]

    def _emit_slice_notification(self, plan: ExecutionPlan, slice_obj: PlanSlice, action: WorkerAction) -> None:
        summary = action.summary or slice_obj.last_summary or slice_obj.last_error
        result = TaskResult(
            task_id=f"{plan.plan_id}:{slice_obj.slice_id}",
            worker_id=slice_obj.assigned_worker_id or self.worker_ids[0],
            status="success" if action.action_type == "final_report" else "error",
            summary=summary,
            artifacts=list(slice_obj.artifacts),
            confidence=action.confidence,
            error=slice_obj.last_error,
            title=slice_obj.title,
            verdict=action.verdict,
            findings=list(action.findings[:3]),
            key_metrics=dict(action.key_metrics),
            next_actions=list(action.next_actions[:3]),
        )
        self.orch.state.results.append(result)
        self.notification_service.send_worker_result(result, self.orch.state.current_cycle)

    def _sanitize_worker_action(self, slice_obj: PlanSlice, action: WorkerAction) -> WorkerAction:
        if not action.reportable_issues:
            return action
        filtered: list[WorkerReportableIssue] = []
        for issue in action.reportable_issues:
            if self._is_contradictory_registry_issue(slice_obj, issue):
                logger.warning(
                    "Suppressing contradictory worker issue for %s: %s",
                    slice_obj.slice_id,
                    issue.summary,
                )
                continue
            filtered.append(issue)
        action.reportable_issues = filtered
        return action

    def _is_contradictory_registry_issue(self, slice_obj: PlanSlice, issue: WorkerReportableIssue) -> bool:
        combined = " ".join(
            part.strip().lower()
            for part in (issue.summary, issue.details, issue.affected_tool)
            if part
        )
        if "tool registry" not in combined and "not found in registry" not in combined:
            return False
        allowlist = set(slice_obj.allowed_tools)
        if issue.affected_tool and issue.affected_tool in allowlist:
            return True
        mentioned = {tool for tool in allowlist if tool.lower() in combined}
        if not mentioned:
            return False
        return bool(self.broker.allowlist() & mentioned)

    async def _terminalize_worker_action_error(
        self,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        *,
        worker_id: str,
        exc: WorkerTerminalActionError,
    ) -> None:
        if exc.parse_error.startswith("tool_prefixed_namespace_forbidden:"):
            incident_summary = "worker_prefixed_tool_name_contract_violation"
            reason_code = "worker_contract_violation"
            summary = f"Worker contract violation: {exc.parse_error}. artifact={exc.artifact_path}"
        else:
            incident_summary = "worker_parse_failure"
            reason_code = "worker_parse_failure"
            summary = f"Worker output could not be parsed: {exc.parse_error}. artifact={exc.artifact_path}"
        self._record_runtime_incident(
            incident_summary,
            {
                "plan_id": plan.plan_id,
                "slice_id": slice_obj.slice_id,
                "worker_id": worker_id,
                "artifact_path": exc.artifact_path,
                "parse_error": exc.parse_error,
                "raw_output": exc.raw_output,
            },
        )
        if hasattr(self.broker, "report_incident"):
            await self.broker.report_incident(
                summary=incident_summary,
                error=exc.parse_error,
                affected_tool="",
                metadata={
                    "plan_id": plan.plan_id,
                    "slice_id": slice_obj.slice_id,
                    "worker_id": worker_id,
                    "artifact_path": exc.artifact_path,
                    "raw_output": exc.raw_output,
                },
                severity="medium",
            )
        self._abort_slice(
            plan,
            slice_obj,
            summary=summary,
            reason_code=reason_code,
            retryable=False,
        )

    async def _broker_call_tool(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        plan_id: str,
        slice_id: str,
        slot: int,
        phase: str,
    ) -> Any:
        parameters = inspect.signature(self.broker.call_tool).parameters
        if {"plan_id", "slice_id", "slot", "phase"}.issubset(parameters):
            return await self.broker.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                plan_id=plan_id,
                slice_id=slice_id,
                slot=slot,
                phase=phase,
            )
        return await self.broker.call_tool(tool_name=tool_name, arguments=arguments)

    def _sync_runtime_summary(self, turn: ExecutionTurn) -> None:
        self.orch.state.current_cycle += 1
        self.orch.state.last_change_at = turn.created_at
        self.orch.state.updated_at = turn.created_at
        if self.console_controller is not None:
            self.console_controller.on_runtime_cycle(
                current_cycle=self.orch.state.current_cycle,
                total_errors=self.orch.state.total_errors,
            )

    @staticmethod
    def _stamp_result_context(result: Any, *, plan_id: str, slice_id: str) -> None:
        result.plan_id = plan_id
        result.slice_id = slice_id

    async def _handle_runtime_error(self, exc: Exception) -> None:
        logger.warning("Brokered execution error: %s", exc)
        self.orch.state.total_errors += 1
        if self.console_controller is not None:
            self.console_controller.on_runtime_error(str(exc), total_errors=self.orch.state.total_errors)
        if isinstance(exc, BrokerServiceError):
            self.state.broker_failure_count += 1
        self._record_runtime_incident("brokered_execution_error", {"error": str(exc), "type": exc.__class__.__name__})
        await self._save_state()

    def _record_runtime_incident(self, summary: str, metadata: dict[str, Any]) -> None:
        path = self.incident_store.record(summary=summary, metadata=metadata, source="brokered_runtime")
        logger.warning("Runtime incident recorded locally: %s", path)

    async def _capture_planner_surface_assumptions(self, plan: ExecutionPlan) -> None:
        blockers = " ".join(self._recent_blockers()).lower()
        if "domain_capability_gap_liquidation_events_missing" not in blockers:
            return
        for slice_obj in plan.slices:
            text = " ".join(
                [
                    slice_obj.title,
                    slice_obj.hypothesis,
                    slice_obj.objective,
                    " ".join(slice_obj.success_criteria),
                    " ".join(slice_obj.evidence_requirements),
                ]
            ).lower()
            if "liquidation" not in text:
                continue
            metadata = {
                "plan_id": plan.plan_id,
                "slice_id": slice_obj.slice_id,
                "slice_title": slice_obj.title,
                "objective": slice_obj.objective,
            }
            self._record_runtime_incident("planner_assumed_missing_domain_surface", metadata)
            if hasattr(self.broker, "report_incident"):
                await self.broker.report_incident(
                    summary="planner_assumed_missing_domain_surface",
                    error="Planner referenced liquidation domain after known capability gap.",
                    affected_tool="",
                    metadata=metadata,
                    severity="medium",
                )

    def _recent_blockers(self) -> list[str]:
        run_incidents_dir = self.incident_store.root_dir / "runs" / self.incident_store.run_id / "incidents"
        fallback_dir = self.incident_store.root_dir / "incidents"
        source_dir = run_incidents_dir if run_incidents_dir.exists() else fallback_dir
        if not source_dir.exists():
            return []
        blockers: list[str] = []
        for path in sorted(source_dir.glob("*.json"))[-12:]:
            try:
                import json

                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            summary = str(payload.get("summary", "") or "").strip()
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            error = str(metadata.get("error", "") or "").strip()
            if summary:
                blockers.append(f"{summary}: {error}" if error else summary)
        return blockers[-8:]

    def _stop_reason_after_round(self, plan: ExecutionPlan, *, did_work: bool) -> StopReason | None:
        if self.orch._drain_mode and not self._has_active_operations(plan):
            return StopReason.GRACEFUL_STOP
        if self.orch.state.total_errors >= int(self.config.max_errors_total or 1):
            return StopReason.MAX_ERRORS
        if self.state.broker_failure_count >= int(self.config.max_broker_failures or 1):
            return StopReason.MCP_UNHEALTHY
        if self.config.plan_source == "planner" and self.state.consecutive_failed_plans >= int(self.config.max_consecutive_failed_plans or 1):
            return StopReason.GOAL_IMPOSSIBLE
        if not did_work and self.state.no_progress_cycles >= int(self.config.max_empty_cycles or 1):
            return StopReason.NO_PROGRESS
        return None

    @staticmethod
    def _has_active_operations(plan: ExecutionPlan) -> bool:
        return any(slice_obj.active_operation_ref for slice_obj in plan.slices if not slice_obj.is_terminal)

    async def _save_state(self) -> None:
        async with self._save_lock:
            self.orch.state.status = self.state.status
            self.state.touch()
            await asyncio.to_thread(self.state_store.save, self.state)
            self.orch.state.updated_at = self.state.updated_at
            self.orch._log_event(OrchestratorEvent.STATE_SAVED)

    async def _finish(self, reason: StopReason, summary: str = "") -> None:
        if self.orch._finish_completed:
            return
        self.orch._finish_completed = True
        self.state.status = "finished"
        self.state.stop_reason = reason.value
        self.orch.state.status = "finished"
        self.orch.state.stop_reason = reason
        await self._save_state()
        report_summary = ""
        run_report = None
        build_reports = getattr(self.orch, "_build_postrun_reports", None)
        if callable(build_reports):
            result = await build_reports()
            if isinstance(result, tuple):
                report_summary, run_report = result
            else:
                report_summary = result or ""
        if self.console_controller is not None:
            self.console_controller.on_runtime_finished(reason=reason.value, total_errors=self.orch.state.total_errors)
        self.notification_service.flush()
        if run_report is not None:
            self.notification_service.send_run_complete(run_report)
        else:
            suffix = f" Reports: {report_summary[:200]}" if report_summary else ""
            self.notification_service.send_lifecycle("finished", f"Reason: {reason.value}. {summary[:200]}{suffix}")
        self.orch._log_event(OrchestratorEvent.FINISHED, f"reason={reason.value} summary={summary[:100]}")
        logger.info("Orchestrator finished: %s. %s %s", reason.value, summary, report_summary)
