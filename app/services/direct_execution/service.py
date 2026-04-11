"""
Async direct execution engine.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from dataclasses import asdict
from typing import Any

from app.compiled_plan_store import CompiledPlanStore
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import DirectAttemptMetadata, ExecutionPlan, ExecutionStateV2, ExecutionTurn, PlanSlice, WorkerAction, make_id
from app.execution_store import ExecutionStateStore
from app.models import OrchestratorEvent, StopReason
from app.plan_sources import CompiledPlanSource, PlanSource, PlannerPlanSource
from app.runtime_console import slot_for_slice
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.budgeting import normalize_plan_budgets
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts
from app.services.direct_execution.managed_invocation import ManagedAdapterInvoker
from app.services.direct_execution.planner import PlannerDecisionCancelled, PlannerDecisionError, PlannerDecisionService
from app.services.direct_execution.slice_readiness import dependency_readiness_blocker, required_output_facts_for_slice
from app.services.direct_execution.tool_catalog import direct_available_tools

logger = logging.getLogger("orchestrator.direct")

_DRAIN_TIMEOUT_SECONDS = 60.0
_SOFT_ABORT_REASON_CODES = {
    "dependency_blocked",
    "infra_contract_blocker",
    "tool_selection_ambiguous",
    "branch_project_contract_blocker",
    "direct_contract_blocker",
    "direct_output_parse_failed",
    "direct_semantic_loop_detected",
    "direct_slice_missing_prerequisite_facts",
    "direct_model_stalled_before_first_action",
    "direct_model_stalled_between_actions",
    "direct_error_loop_detected",
}


class DirectExecutionService:
    def __init__(
        self,
        orch: Any,
        *,
        planner: PlannerDecisionService | None = None,
        direct_executor: DirectSliceExecutor | None = None,
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
        self.process_invoker = ManagedAdapterInvoker(process_registry=orch.process_registry)
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
                invoker=self.process_invoker.invoke_with_retries,
            )
        self.direct_executor = direct_executor or DirectSliceExecutor(
            adapter=orch.worker_adapter,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            direct_config=self.config.direct_execution,
            worker_system_prompt=self.config.worker_system_prompt,
            invoker=self.process_invoker.invoke_with_retries,
        )
        self.fallback_executor = self._build_fallback_executor()
        self.plan_source = plan_source or self._build_plan_source()
        for plan in self.state.plans:
            normalize_plan_budgets(plan)
        self._save_lock = asyncio.Lock()
        self._shutdown_requested = False

    def _build_fallback_executor(self):
        from app.runtime_factory import create_fallback_adapter
        from app.services.direct_execution.fallback_executor import FallbackExecutor

        de = self.config.direct_execution
        fallback_providers = [de.fallback_1, de.fallback_2]

        if not any(name and name.strip() for name in fallback_providers):
            return None

        config = self.config

        def _factory(provider_name: str):
            return create_fallback_adapter(provider_name, config)

        return FallbackExecutor(
            primary_executor=self.direct_executor,
            fallback_providers=fallback_providers,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            direct_config=self.config.direct_execution,
            worker_system_prompt=self.config.worker_system_prompt,
            adapter_factory=_factory,
            invoker=self.process_invoker.invoke_with_retries,
        )

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
            available_tools_provider=direct_available_tools,
            state_summary_provider=self._state_summary,
        )

    async def run(self) -> StopReason:
        self.orch.state.status = "running"
        self.state.status = "running"
        if self.console_controller is not None:
            self.console_controller.on_runtime_started(goal=self.config.goal, plan_source=str(self.config.plan_source or ""))
        await self._save_state()
        try:
            while True:
                try:
                    created_new_plan = False
                    if self.orch._stop_requested:
                        await self._finish(StopReason.GRACEFUL_STOP, "stop_requested")
                        return StopReason.GRACEFUL_STOP
                    if self.orch._drain_mode and self.orch._drain_started_at is not None:
                        drain_elapsed = _time.monotonic() - self.orch._drain_started_at
                        if drain_elapsed > _DRAIN_TIMEOUT_SECONDS:
                            self.orch.request_stop_now()
                            await self._finish(StopReason.GRACEFUL_STOP, "drain_timeout")
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
                        created_new_plan = True
                    did_work = await self._run_plan_round(plan)
                    self._finalize_plan_if_ready(plan)
                    made_progress = did_work or created_new_plan
                    self.state.no_progress_cycles = 0 if made_progress else self.state.no_progress_cycles + 1
                    await self._save_state()
                    stop_reason = self._stop_reason_after_round(plan, did_work=made_progress)
                    if stop_reason is not None:
                        await self._finish(stop_reason, f"plan={plan.plan_id} status={plan.status}")
                        return stop_reason
                    if not did_work:
                        await asyncio.sleep(max(0.05, float(self.config.decision_cycle_sleep_seconds or 0.25)))
                except PlannerDecisionCancelled as exc:
                    if self.orch._stop_requested:
                        await self._finish(StopReason.GRACEFUL_STOP, str(exc) or "shutdown_cancelled_inflight")
                        return StopReason.GRACEFUL_STOP
                    await self._handle_runtime_error(exc)
                except PlannerDecisionError as exc:
                    await self._handle_runtime_error(exc)
                    if self.orch.state.total_errors >= int(self.config.max_errors_total or 1):
                        await self._finish(StopReason.MAX_ERRORS, str(exc))
                        return StopReason.MAX_ERRORS
        finally:
            pass

    def request_shutdown(self, *, immediate: bool) -> None:
        self._shutdown_requested = True
        if immediate:
            self.orch.process_registry.terminate_all(grace_seconds=0.2, force_after=0.5)

    async def _create_next_plan(self) -> ExecutionPlan | None:
        plan = await self.plan_source.next_plan_batch(self.state)
        if plan is None:
            return None
        normalize_plan_budgets(plan)
        plan.status = "running"
        self.state.plans.append(plan)
        self.state.current_plan_id = plan.plan_id
        self._persist_plan_snapshot(plan)
        if self.console_controller is not None:
            self.console_controller.on_plan_created(plan_id=plan.plan_id, plan=plan, all_plans=self.state.plans)
        logger.info("Created direct execution plan %s with %d slices", plan.plan_id, len(plan.slices))
        return plan

    async def _run_plan_round(self, plan: ExecutionPlan) -> bool:
        aborted_before = self._abort_dependency_blocked_slices(plan)
        candidate_slices = [
            item
            for item in sorted(plan.slices, key=lambda slice_obj: (slice_obj.parallel_slot, slice_obj.slice_id))
            if not item.is_terminal
            and not self._is_blocked_checkpoint(item)
            and item.parallel_slot <= self.max_slots
            and self._dependencies_satisfied(plan, item)
        ]
        if not candidate_slices:
            return aborted_before
        semaphore = asyncio.Semaphore(self.max_slots)
        tasks = [asyncio.create_task(self._process_slice(plan, item, semaphore)) for item in candidate_slices]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        did_work = False
        for result in results:
            if isinstance(result, Exception):
                await self._handle_runtime_error(result)
                continue
            did_work = bool(result) or did_work
        aborted_after = self._abort_dependency_blocked_slices(plan)
        return did_work or aborted_before or aborted_after

    async def _process_slice(self, plan: ExecutionPlan, slice_obj: PlanSlice, semaphore: asyncio.Semaphore) -> bool:
        async with semaphore:
            if slice_obj.is_terminal or self._is_blocked_checkpoint(slice_obj):
                return False
            if self.orch._drain_mode:
                return False
            return await self._execute_slice_turn(plan, slice_obj)

    async def _execute_slice_turn(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        blocker = dependency_readiness_blocker(plan, slice_obj)
        if blocker is not None:
            slice_obj.facts.setdefault("direct.missing_prerequisite_facts", blocker.missing_facts)
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary=blocker.summary,
                reason_code=blocker.reason_code,
            )
            return True
        if slice_obj.turn_count >= slice_obj.max_turns:
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary="Direct slice turn budget exhausted before terminal report.",
                reason_code="direct_turn_budget_exhausted",
            )
            return True
        worker_id = self.worker_ids[(slice_obj.parallel_slot - 1) % len(self.worker_ids)]
        slice_obj.assigned_worker_id = worker_id
        slice_obj.status = "running"
        # Display the actual execution provider, not the legacy config worker_id
        display_worker_id = str(getattr(self.config.direct_execution, "provider", "") or worker_id)
        if self.console_controller is not None:
            self.console_controller.on_slice_turn_started(
                slot=slot_for_slice(slice_obj.parallel_slot),
                plan_id=plan.plan_id,
                slice_id=slice_obj.slice_id,
                title=slice_obj.title,
                worker_id=display_worker_id,
                turns_used=slice_obj.turn_count,
                turns_total=slice_obj.max_turns,
                tool_calls_used=slice_obj.tool_call_count,
                tool_calls_total=slice_obj.max_tool_calls,
                summary=slice_obj.last_summary or slice_obj.last_checkpoint_summary,
                operation_ref=slice_obj.active_operation_ref,
                operation_status=slice_obj.active_operation_status,
                plan=plan,
                all_plans=self.state.plans,
            )
        self._persist_plan_snapshot(plan)
        _slot = slot_for_slice(slice_obj.parallel_slot)
        _max_tool_calls = slice_obj.max_tool_calls
        _max_turns = slice_obj.max_turns

        def _on_tool_progress(tool_call_count: int, expensive_call_count: int) -> None:
            if self.console_controller is not None:
                self.console_controller.update_budget(
                    _slot,
                    turns_used=slice_obj.turn_count + 1,
                    tool_calls_used=tool_call_count,
                    tool_calls_total=_max_tool_calls,
                )

        execute_kwargs = dict(
            plan_id=plan.plan_id,
            slice_obj=slice_obj,
            baseline_bootstrap=self.config.research_config,
            known_facts=self._known_facts_for_slice(plan, slice_obj),
            required_output_facts=required_output_facts_for_slice(plan, slice_obj),
            recent_turn_summaries=self._recent_turn_summaries(plan.plan_id, slice_obj.slice_id),
            checkpoint_summary=slice_obj.last_checkpoint_summary,
            on_tool_progress=_on_tool_progress,
        )
        fallback_attempts_data: list[dict[str, Any]] = []
        if self.fallback_executor is not None:
            def _on_provider_switch(provider_name: str, fallback_level: int | None = None) -> None:
                if self.console_controller is not None:
                    self.console_controller.update_worker_id(
                        _slot,
                        worker_id=provider_name,
                        fallback_level=fallback_level,
                    )

            result, fallback_attempts = await self.fallback_executor.execute_with_fallback(
                **execute_kwargs,
                on_provider_switch=_on_provider_switch,
            )
            for attempt in fallback_attempts:
                fallback_attempts_data.append({
                    "provider": attempt.provider,
                    "adapter_name": attempt.adapter_name,
                    "artifact_path": attempt.artifact_path,
                    "duration_ms": attempt.result.duration_ms,
                    "error": attempt.result.error,
                    "attempt_index": attempt.attempt_index,
                })
        else:
            result = await self.direct_executor.execute(**execute_kwargs)
        slice_obj.turn_count += 1
        slice_obj.tool_call_count += int(result.tool_call_count or 0)
        slice_obj.expensive_call_count += int(result.expensive_tool_call_count or 0)
        action = result.action or self._blocked_action_from_failed_direct(result)
        if result.artifact_path and result.artifact_path not in action.artifacts:
            action.artifacts.append(result.artifact_path)
        action.facts.setdefault("execution.kind", "direct")
        action.facts.setdefault("direct.provider", result.provider)
        if result.fallback_provider_index > 0:
            action.facts.setdefault("direct.fallback_used", result.provider)
        action.facts.setdefault("direct.fallback_provider_index", result.fallback_provider_index)
        turn = ExecutionTurn(
            turn_id=make_id("turn"),
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            worker_id=worker_id,
            turn_index=slice_obj.turn_count,
            action=action,
            direct_attempt=DirectAttemptMetadata(
                provider=result.provider,
                artifact_path=result.artifact_path,
                duration_ms=result.duration_ms,
                error=result.error,
                tool_call_count=result.tool_call_count,
                expensive_tool_call_count=result.expensive_tool_call_count,
                parse_retry_count=result.parse_retry_count,
                fallback_attempts=fallback_attempts_data,
            ),
        )
        self.artifact_store.save_turn_action(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=action)
        if action.action_type == "final_report":
            self._apply_final_report(plan, slice_obj, turn)
        elif action.action_type == "checkpoint":
            self._apply_checkpoint(plan, slice_obj, turn)
        elif action.action_type == "abort":
            self._apply_abort(plan, slice_obj, turn)
        else:
            self._checkpoint_blocked(plan, slice_obj, summary=f"Direct model returned unsupported action: {action.action_type}", reason_code="direct_action_invalid")
        self.state.turn_history.append(turn)
        plan.touch()
        self._sync_runtime_summary(turn)
        self._persist_plan_snapshot(plan)
        return True

    def _apply_checkpoint(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        action = turn.action
        if action.status == "complete":
            self._apply_final_report(plan, slice_obj, turn)
            return
        slice_obj.status = "checkpointed"
        slice_obj.last_checkpoint_turn_id = turn.turn_id
        slice_obj.last_checkpoint_status = action.status or "partial"
        slice_obj.last_checkpoint_summary = action.summary
        slice_obj.last_summary = action.summary
        slice_obj.last_error = action.reason_code if action.status == "blocked" else ""
        if self._is_zero_tool_call_stall(slice_obj):
            slice_obj.last_checkpoint_status = "blocked"
            slice_obj.last_error = "direct_model_zero_tool_call_stall"
            stall_summary = (
                f"Slice blocked after {slice_obj.turn_count} direct attempts with zero tool calls. "
                "The model is not making progress with available tools."
            )
            slice_obj.last_checkpoint_summary = stall_summary
            slice_obj.last_summary = stall_summary
        slice_obj.facts.update(action.facts)
        slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
        if self.console_controller is not None:
            self.console_controller.on_slice_checkpoint(slot=slot_for_slice(slice_obj.parallel_slot), summary=slice_obj.last_summary, operation_ref="", operation_status="")

    def _apply_final_report(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        action = turn.action
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=required_output_facts_for_slice(plan, slice_obj),
            inherited_facts=self._known_facts_for_slice(plan, slice_obj),
        )
        if readiness.missing_required_facts:
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary=(
                    f"Direct final_report for slice '{slice_obj.slice_id}' is missing downstream facts: "
                    f"{', '.join(readiness.missing_required_facts)}."
                ),
                reason_code="direct_slice_missing_prerequisite_facts",
            )
            return
        fallback_level = int(action.facts.get("direct.fallback_provider_index", 0))
        slice_obj.status = "completed"
        slice_obj.final_report_turn_id = turn.turn_id
        slice_obj.last_summary = action.summary
        slice_obj.last_error = ""
        slice_obj.last_checkpoint_status = ""
        slice_obj.facts.update(readiness.facts)
        slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
        self.artifact_store.save_report(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=asdict(action) | {"type": action.action_type})
        if self.console_controller is not None:
            self.console_controller.on_slice_completed(slot=slot_for_slice(slice_obj.parallel_slot), summary=action.summary, via="direct", fallback_level=fallback_level)

    def _apply_abort(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        action = turn.action
        if action.reason_code in _SOFT_ABORT_REASON_CODES or action.retryable:
            self._checkpoint_blocked(plan, slice_obj, summary=action.summary, reason_code=action.reason_code or "direct_soft_blocker")
            return
        fallback_level = int(action.facts.get("direct.fallback_provider_index", 0))
        slice_obj.status = "failed"
        slice_obj.last_error = action.reason_code or action.reason or "direct_abort"
        slice_obj.last_summary = action.summary
        slice_obj.facts.update(action.facts)
        slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
        self.artifact_store.save_report(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=asdict(action) | {"type": action.action_type})
        if self.console_controller is not None:
            self.console_controller.on_slice_failed(slot=slot_for_slice(slice_obj.parallel_slot), summary=action.summary, fallback_level=fallback_level)

    def _checkpoint_blocked(self, plan: ExecutionPlan, slice_obj: PlanSlice, *, summary: str, reason_code: str) -> None:
        slice_obj.status = "checkpointed"
        slice_obj.last_checkpoint_status = "blocked"
        slice_obj.last_checkpoint_summary = summary
        slice_obj.last_summary = summary
        slice_obj.last_error = reason_code
        if self.console_controller is not None:
            self.console_controller.on_slice_checkpoint(slot=slot_for_slice(slice_obj.parallel_slot), summary=summary, operation_ref="", operation_status="")
        self._persist_plan_snapshot(plan)

    @staticmethod
    def _blocked_action_from_failed_direct(result: DirectExecutionResult) -> WorkerAction:
        return WorkerAction(
            action_id=make_id("action"),
            action_type="checkpoint",
            status="blocked",
            summary=result.error or "Direct execution failed before producing a terminal action.",
            facts={"direct.error": result.error, "direct.provider": result.provider},
            artifacts=[result.artifact_path] if result.artifact_path else [],
            reason_code="direct_output_parse_failed",
            retryable=False,
        )

    def _abort_dependency_blocked_slices(self, plan: ExecutionPlan) -> bool:
        changed = False
        for slice_obj in plan.slices:
            if slice_obj.is_terminal or self._is_blocked_checkpoint(slice_obj):
                continue
            blockers = [
                dep_id
                for dep_id in slice_obj.depends_on
                if (dep_slice := self._resolve_dependency_slice(plan, dep_id)) is not None
                and (dep_slice.status in {"failed", "aborted"} or self._is_blocked_checkpoint(dep_slice))
            ]
            if not blockers:
                continue
            slice_obj.status = "aborted"
            slice_obj.last_error = "dependency_blocked"
            slice_obj.last_summary = f"Blocked by prerequisite slice(s): {', '.join(blockers)}"
            changed = True
        if changed:
            plan.touch()
            self._persist_plan_snapshot(plan)
        return changed

    def _dependencies_satisfied(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        if not slice_obj.depends_on:
            return True
        return all(
            (dep_slice := self._resolve_dependency_slice(plan, dep_id)) is not None and dep_slice.status == "completed"
            for dep_id in slice_obj.depends_on
        )

    @staticmethod
    def _is_blocked_checkpoint(slice_obj: PlanSlice) -> bool:
        return slice_obj.status == "checkpointed" and slice_obj.last_checkpoint_status == "blocked"

    @staticmethod
    def _is_zero_tool_call_stall(slice_obj: PlanSlice) -> bool:
        return (
            slice_obj.last_checkpoint_status == "partial"
            and slice_obj.turn_count >= 2
            and slice_obj.tool_call_count == 0
        )

    def _finalize_plan_if_ready(self, plan: ExecutionPlan) -> None:
        if plan.is_terminal:
            return
        if any(slice_obj.status == "failed" for slice_obj in plan.slices):
            plan.status = "failed"
            self.state.consecutive_failed_plans += 1
            self.plan_source.mark_plan_failed(plan, self.state)
        elif all(slice_obj.status == "completed" for slice_obj in plan.slices):
            plan.status = "completed"
            self.state.completed_plan_count += 1
            self.state.consecutive_failed_plans = 0
            self.plan_source.mark_plan_complete(plan, self.state)
        elif all(slice_obj.status in {"completed", "aborted"} or self._is_blocked_checkpoint(slice_obj) for slice_obj in plan.slices):
            plan.status = "stopped"
            self.plan_source.mark_plan_failed(plan, self.state)
        else:
            return
        plan.touch()
        self._persist_plan_snapshot(plan)
        if self.console_controller is not None:
            self.console_controller.sync_sequence_progress(plan=plan, all_plans=self.state.plans, active_slice_id=None)

    def _stop_reason_after_round(self, plan: ExecutionPlan, *, did_work: bool) -> StopReason | None:
        if self.orch._stop_requested:
            return StopReason.GRACEFUL_STOP
        if plan.status == "failed" and self.state.consecutive_failed_plans >= int(self.config.max_consecutive_failed_plans or 1):
            return StopReason.MAX_ERRORS
        if not did_work and self.state.no_progress_cycles >= int(self.config.max_empty_cycles or 1):
            return StopReason.NO_PROGRESS
        return None

    async def _finish(self, reason: StopReason, summary: str) -> None:
        if self.console_controller is not None:
            self.console_controller.on_runtime_finished(reason=reason.value, total_errors=self.orch.state.total_errors)
        await self.orch._finish_async(reason, summary)

    async def _save_state(self) -> None:
        self.orch.state.status = self.state.status
        self.state_store.save(self.state)
        self.orch._log_event(OrchestratorEvent.STATE_SAVED)

    def _persist_plan_snapshot(self, plan: ExecutionPlan) -> None:
        self.artifact_store.save_plan(plan)

    async def _handle_runtime_error(self, exc: BaseException) -> None:
        self.orch.state.total_errors += 1
        message = str(exc) or exc.__class__.__name__
        logger.warning("Direct execution error: %s", message)
        self.incident_store.record(summary="direct_execution_error", metadata={"error": message, "type": exc.__class__.__name__}, source="direct_runtime")
        if self.console_controller is not None:
            self.console_controller.on_runtime_error(message, total_errors=self.orch.state.total_errors)
        await self._save_state()

    def _known_facts_for_slice(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> dict[str, Any]:
        facts: dict[str, Any] = dict(slice_obj.facts)
        for dep_id in slice_obj.depends_on:
            dep = self._resolve_dependency_slice(plan, dep_id)
            if dep is None:
                continue
            for key, value in dep.facts.items():
                facts.setdefault(f"{dep_id}.{key}", value)
        for prior in plan.slices:
            if prior.slice_id == slice_obj.slice_id:
                break
            for key, value in prior.facts.items():
                facts.setdefault(f"{prior.slice_id}.{key}", value)
        return facts

    def _resolve_dependency_slice(self, plan: ExecutionPlan, dep_id: str) -> PlanSlice | None:
        local = {item.slice_id: item for item in plan.slices}
        if dep_id in local:
            return local[dep_id]
        # Cross-batch dependencies can point to slices completed in prior plans.
        for prior_plan in reversed(self.state.plans):
            if prior_plan.plan_id == plan.plan_id:
                continue
            for prior_slice in prior_plan.slices:
                if prior_slice.slice_id == dep_id:
                    return prior_slice
        return None

    def _recent_turn_summaries(self, plan_id: str, slice_id: str) -> list[str]:
        summaries: list[str] = []
        for turn in self.state.turn_history:
            if turn.plan_id == plan_id and turn.slice_id == slice_id:
                summaries.append(turn.action.summary or turn.direct_attempt.error or turn.action.action_type)
        return summaries[-6:]

    def _state_summary(self) -> str:
        if not self.state.plans:
            return "No prior direct execution plans."
        lines = []
        for plan in self.state.plans[-5:]:
            lines.append(f"{plan.plan_id}: status={plan.status}")
            for slice_obj in plan.slices:
                if slice_obj.last_summary or slice_obj.last_error:
                    lines.append(f"- {slice_obj.slice_id}: {slice_obj.status} {slice_obj.last_summary or slice_obj.last_error}")
        return "\n".join(lines)

    def _sync_runtime_summary(self, turn: ExecutionTurn) -> None:
        del turn


__all__ = ["DirectExecutionService", "_DRAIN_TIMEOUT_SECONDS"]
