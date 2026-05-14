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
from app.services.direct_execution.acceptance import acceptance_blocker_reason, dependency_unblocked_by
from app.services.direct_execution.acceptance.verifier import AcceptanceVerifier
from app.services.direct_execution.backtests_facts import normalize_backtests_facts
from app.services.direct_execution.blocker_classification import blocker_class_from_slice
from app.services.direct_execution.budgeting import normalize_plan_budgets
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts
from app.services.direct_execution.guardrails import (
    final_report_passes_quality_gate,
    is_prerequisite_block_terminal,
)
from app.services.direct_execution.managed_invocation import ManagedAdapterInvoker
from app.services.direct_execution.planner import PlannerDecisionCancelled, PlannerDecisionError, PlannerDecisionService
from app.services.direct_execution.slice_readiness import dependency_readiness_blocker, downstream_prerequisites_blocker, optional_slice_gate_blocker, required_output_facts_for_slice, upstream_artifact_gate_blocker
from app.services.direct_execution.terminal_application import TerminalActionApplicator
from app.services.mcp_catalog.models import McpCatalogSnapshot

logger = logging.getLogger("orchestrator.direct")

_DRAIN_TIMEOUT_SECONDS = 60.0
def _aggregate_plan_last_error(slices: list[PlanSlice], *, prefer_status: str) -> str:
    """Pick the most meaningful slice-level error for plan-level propagation.

    Prefers errors on slices matching ``prefer_status``; falls back to any
    slice with a non-empty ``last_error``. Used by plan_sources to decide
    whether a plan failed from a transient infra event (continue with next
    batch via fallback) vs. a semantic abort (skip the rest of the sequence).
    """
    for slice_obj in slices:
        if slice_obj.status == prefer_status and slice_obj.last_error:
            return str(slice_obj.last_error)
    for slice_obj in slices:
        if slice_obj.last_error:
            return str(slice_obj.last_error)
    return ""

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
    "feature_data_unavailable",
    "no_features_available",
    "infrastructure_data_unavailable",
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
        self.catalog_snapshot: McpCatalogSnapshot | None = getattr(orch, "mcp_catalog_snapshot", None)
        if self.catalog_snapshot is None:
            raise RuntimeError("mcp_catalog_snapshot_missing")
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
                catalog_snapshot=self.catalog_snapshot,
            )
        self.direct_executor = direct_executor or DirectSliceExecutor(
            adapter=orch.worker_adapter,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            direct_config=self.config.direct_execution,
            worker_system_prompt=self.config.worker_system_prompt,
            invoker=self.process_invoker.invoke_with_retries,
            provider_name=self.config.direct_execution.provider,
            catalog_snapshot=self.catalog_snapshot,
        )
        self.fallback_executor = self._build_fallback_executor()
        self.acceptance_verifier = AcceptanceVerifier(
            direct_config=self.config.direct_execution,
            incident_store=self.incident_store,
        )
        self.plan_source = plan_source or self._build_plan_source()
        self.terminal_actions = TerminalActionApplicator(
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            console_controller=self.console_controller,
            known_facts_for_slice=self._known_facts_for_slice,
            required_output_facts_for_slice=required_output_facts_for_slice,
            persist_plan_snapshot=self._persist_plan_snapshot,
        )
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
            acceptance_verifier=AcceptanceVerifier(
                direct_config=self.config.direct_execution,
                incident_store=self.incident_store,
            ),
        )

    def _build_plan_source(self) -> PlanSource:
        if self.config.plan_source == "compiled_raw":
            return CompiledPlanSource(
                store=CompiledPlanStore(self.config.compiled_plan_dir),
                raw_plan_dir=self.config.raw_plan_dir,
                skip_failures=bool(self.config.compiled_queue_skip_failures),
                notification_service=self.notification_service,
                catalog_snapshot=self.catalog_snapshot,
                incident_store=self.incident_store,
                infra_failure_never_skip_batches=bool(
                    getattr(self.config, "infra_failure_never_skip_batches", True)
                ),
                start_from=getattr(self.config, "start_from", ""),
            )
        assert self.planner is not None
        return PlannerPlanSource(
            planner=self.planner,
            goal=self.config.goal,
            baseline_bootstrap=self.config.research_config,
            max_worker_count=self.max_slots,
            max_plans_per_run=self.config.max_plans_per_run,
            available_tools_provider=self.catalog_snapshot.tool_name_set,
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
                    await self._finalize_plan_if_ready(plan)
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
        # Re-check previously aborted slices whose cross-batch dependencies
        # may now be satisfied by completed slices in prior plans.
        for slice_obj in plan.slices:
            if slice_obj.status != "aborted" or slice_obj.last_error != "dependency_blocked":
                continue
            all_satisfied = all(
                (dep := self._resolve_dependency_slice(plan, dep_id)) is not None
                and dependency_unblocked_by(dep)
                for dep_id in slice_obj.depends_on
            )
            if all_satisfied:
                slice_obj.status = "pending"
                slice_obj.acceptance_state = "pending"
                slice_obj.last_error = ""
                slice_obj.dependency_blocker_slice_id = ""
        self.state.plans.append(plan)
        self.state.current_plan_id = plan.plan_id
        self._persist_plan_snapshot(plan)
        if self.console_controller is not None:
            self.console_controller.on_plan_created(plan_id=plan.plan_id, plan=plan, all_plans=self.state.plans)
        logger.info("Created direct execution plan %s with %d slices", plan.plan_id, len(plan.slices))
        return plan

    async def _run_plan_round(self, plan: ExecutionPlan) -> bool:
        retried = self._retry_blocked_slices(plan)
        aborted_before = self._abort_dependency_blocked_slices(plan, scope_plan_id=plan.plan_id)
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
        aborted_after = self._abort_dependency_blocked_slices(plan, scope_plan_id=plan.plan_id)
        return did_work or bool(retried) or aborted_before or aborted_after

    def _retry_blocked_slices(self, plan: ExecutionPlan) -> list[PlanSlice]:
        max_retries = int(getattr(self.config.direct_execution, "max_blocked_retries", 2) or 0)
        if max_retries <= 0:
            return []
        retried: list[PlanSlice] = []
        for slice_obj in plan.slices:
            if not self._is_blocked_checkpoint(slice_obj):
                continue
            if slice_obj.blocked_retry_count >= max_retries:
                continue
            if slice_obj.turn_count >= slice_obj.max_turns:
                continue
            slice_obj.blocked_retry_count += 1
            slice_obj.status = "pending"
            slice_obj.acceptance_state = "pending"
            slice_obj.last_checkpoint_status = ""
            slice_obj.last_error = ""
            retried.append(slice_obj)
        if retried:
            plan.touch()
            self._persist_plan_snapshot(plan)
        return retried

    async def _process_slice(self, plan: ExecutionPlan, slice_obj: PlanSlice, semaphore: asyncio.Semaphore) -> bool:
        async with semaphore:
            if slice_obj.is_terminal or self._is_blocked_checkpoint(slice_obj):
                return False
            if self.orch._drain_mode:
                return False
            return await self._execute_slice_turn(plan, slice_obj)

    async def _execute_slice_turn(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        missing_tools = [
            str(tool).strip()
            for tool in slice_obj.allowed_tools
            if str(tool).strip() and str(tool).strip() not in self.catalog_snapshot.tool_name_set()
        ]
        if missing_tools:
            self.incident_store.record(
                summary="Slice references tools missing from current live MCP catalog",
                metadata={
                    "plan_id": plan.plan_id,
                    "slice_id": slice_obj.slice_id,
                    "missing_tools": missing_tools,
                    "mcp_catalog_hash": self.catalog_snapshot.schema_hash,
                },
                source="direct_runtime",
                severity="high",
            )
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary=f"Slice blocked: live MCP catalog no longer exposes required tools: {', '.join(missing_tools)}.",
                reason_code="mcp_catalog_tool_missing",
            )
            return True
        blocker = dependency_readiness_blocker(
            plan,
            slice_obj,
            resolve_dependency=lambda dep_id: self._resolve_dependency_slice(plan, dep_id),
        )
        if blocker is not None:
            slice_obj.facts.setdefault("direct.missing_prerequisite_facts", blocker.missing_facts)
            if str(slice_obj.runtime_profile or "").startswith("backtests_"):
                self.incident_store.record(
                    summary="Strict backtests slice blocked by missing prerequisite facts",
                    metadata={
                        "plan_id": plan.plan_id,
                        "slice_id": slice_obj.slice_id,
                        "runtime_profile": slice_obj.runtime_profile,
                        "missing_facts": list(blocker.missing_facts),
                        "blocking_slice_ids": list(blocker.blocking_slice_ids),
                    },
                    source="direct_runtime",
                    severity="medium",
                )
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary=blocker.summary,
                reason_code=blocker.reason_code,
            )
            return True
        # Check optional/gate conditions before executing the slice.
        gate_blocker = optional_slice_gate_blocker(plan, slice_obj)
        if gate_blocker is not None:
            self._checkpoint_blocked(
                plan,
                slice_obj,
                summary=gate_blocker.summary,
                reason_code=gate_blocker.reason_code,
            )
            return True
        # Check upstream artifact gate: if dependencies produced zero runs
        # but this slice requires run_set_non_empty, skip it automatically.
        artifact_blocker = upstream_artifact_gate_blocker(
            plan,
            slice_obj,
            resolve_dependency=lambda dep_id: self._resolve_dependency_slice(plan, dep_id),
        )
        if artifact_blocker is not None:
            slice_obj.status = "completed"
            slice_obj.acceptance_state = "accepted_ready"
            slice_obj.verdict = "SKIP"
            slice_obj.last_summary = artifact_blocker.summary
            slice_obj.last_error = ""
            slice_obj.facts["direct.skipped_by_upstream_zero_artifacts"] = True
            slice_obj.facts["dependency.blocking_slice_ids"] = list(artifact_blocker.blocking_slice_ids)
            if self.console_controller is not None:
                self.console_controller.on_slice_completed(
                    slot=slot_for_slice(slice_obj.parallel_slot),
                    summary=artifact_blocker.summary,
                    via="auto_skip",
                    fallback_level=0,
                )
            self.incident_store.record(
                summary="Slice auto-skipped: upstream produced zero backtest artifacts",
                metadata={
                    "plan_id": plan.plan_id,
                    "slice_id": slice_obj.slice_id,
                    "slice_title": slice_obj.title,
                    "blocking_slice_ids": list(artifact_blocker.blocking_slice_ids),
                },
                source="direct_runtime",
                severity="low",
            )
            self._persist_plan_snapshot(plan)
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

        _checkpoint_summary = slice_obj.last_checkpoint_summary
        if slice_obj.blocked_retry_count > 0:
            _checkpoint_summary = (
                f"[RETRY #{slice_obj.blocked_retry_count}] "
                "Previous attempts were blocked because the model returned results without "
                "using any MCP tools. You MUST call tools to produce real evidence."
            )

        execute_kwargs = dict(
            plan_id=plan.plan_id,
            slice_obj=slice_obj,
            baseline_bootstrap=self.config.research_config,
            known_facts=self._known_facts_for_slice(plan, slice_obj),
            required_output_facts=required_output_facts_for_slice(plan, slice_obj),
            recent_turn_summaries=self._recent_turn_summaries(plan.plan_id, slice_obj.slice_id),
            checkpoint_summary=_checkpoint_summary,
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
        action = result.action or self.terminal_actions.blocked_action_from_failed_direct(result)
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
                mcp_catalog_hash=self.catalog_snapshot.schema_hash,
                duration_ms=result.duration_ms,
                error=result.error,
                tool_call_count=result.tool_call_count,
                expensive_tool_call_count=result.expensive_tool_call_count,
                parse_retry_count=result.parse_retry_count,
                fallback_attempts=fallback_attempts_data,
            ),
        )
        if action.action_type == "final_report":
            verified = await self._verify_acceptance_before_final_report(
                plan=plan,
                slice_obj=slice_obj,
                turn=turn,
                transcript=list(result.transcript or []),
                known_facts=dict(execute_kwargs["known_facts"]),
                required_output_facts=list(execute_kwargs["required_output_facts"]),
                existing_acceptance_result=dict(result.acceptance_result or {}),
            )
            self.artifact_store.save_turn_action(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=action)
            if not verified:
                self.state.turn_history.append(turn)
                plan.touch()
                self._sync_runtime_summary(turn)
                self._persist_plan_snapshot(plan)
                return True
            self.terminal_actions.apply_final_report(plan, slice_obj, turn)
        elif action.action_type == "checkpoint":
            self.artifact_store.save_turn_action(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=action)
            self.terminal_actions.apply_checkpoint(plan, slice_obj, turn)
        elif action.action_type == "abort":
            self.artifact_store.save_turn_action(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=action)
            self.terminal_actions.apply_abort(
                plan,
                slice_obj,
                turn,
                soft_abort_reason_codes=_SOFT_ABORT_REASON_CODES,
            )
        else:
            self.artifact_store.save_turn_action(plan_id=plan.plan_id, slice_id=slice_obj.slice_id, turn_id=turn.turn_id, payload=action)
            self._checkpoint_blocked(plan, slice_obj, summary=f"Direct model returned unsupported action: {action.action_type}", reason_code="direct_action_invalid")
        self.state.turn_history.append(turn)
        plan.touch()
        self._sync_runtime_summary(turn)
        self._persist_plan_snapshot(plan)
        return True

    async def _verify_acceptance_before_final_report(
        self,
        *,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        turn: ExecutionTurn,
        transcript: list[dict[str, Any]],
        known_facts: dict[str, Any],
        required_output_facts: list[str],
        existing_acceptance_result: dict[str, Any] | None = None,
    ) -> bool:
        action = turn.action
        passes_gate, gate_reason = final_report_passes_quality_gate(
            tool_call_count=int(turn.direct_attempt.tool_call_count or 0),
            action=action,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )
        if not passes_gate:
            return True
        if isinstance(existing_acceptance_result, dict) and existing_acceptance_result.get("status"):
            proof = dict(existing_acceptance_result)
            passed = str(proof.get("status") or "").strip().lower() == "pass"
            blockers = [str(item) for item in proof.get("blocking_reasons", [])] if isinstance(proof.get("blocking_reasons"), list) else []
            route = str(proof.get("route") or "")
        else:
            result = await self.acceptance_verifier.verify(
                plan=plan,
                slice_obj=slice_obj,
                action=action,
                transcript=transcript,
                known_facts=known_facts,
                required_output_facts=required_output_facts,
            )
            proof = result.to_dict()
            passed = result.passed
            blockers = list(result.blocking_reasons)
            route = result.route
        slice_obj.acceptance_proof = proof
        slice_obj.acceptance_blockers = list(blockers)
        action.facts["acceptance.status"] = str(proof.get("status") or "")
        action.facts["acceptance.failed_predicates"] = list(blockers)
        proof_path = self.artifact_store.save_acceptance_proof(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload=proof,
        )
        if str(proof_path) not in action.artifacts:
            action.artifacts.append(str(proof_path))
        if passed:
            return True
        self.incident_store.record(
            summary="Direct final_report rejected by acceptance verifier",
            metadata={
                "plan_id": plan.plan_id,
                "slice_id": slice_obj.slice_id,
                "slice_title": slice_obj.title,
                "contract_kind": proof.get("contract", {}).get("kind") if isinstance(proof.get("contract"), dict) else "",
                "blocking_reasons": list(blockers),
                "route": route,
                "provider": turn.direct_attempt.provider,
            },
            source="direct_acceptance",
            severity="medium",
        )
        self.terminal_actions.checkpoint_blocked(
            plan,
            slice_obj,
            summary=(
                f"Direct final_report for slice '{slice_obj.slice_id}' failed acceptance verifier: "
                f"{', '.join(blockers[:5]) or 'acceptance_verifier_failed'}."
            ),
            reason_code="direct_acceptance_verifier_failed",
        )
        slice_obj.acceptance_proof = proof
        slice_obj.acceptance_blockers = list(blockers)
        slice_obj.facts["acceptance.status"] = str(proof.get("status") or "")
        slice_obj.facts["acceptance.failed_predicates"] = list(blockers)
        return False

    def _checkpoint_blocked(self, plan: ExecutionPlan, slice_obj: PlanSlice, *, summary: str, reason_code: str) -> None:
        self.terminal_actions.checkpoint_blocked(
            plan,
            slice_obj,
            summary=summary,
            reason_code=reason_code,
        )

    def _apply_checkpoint(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        self.terminal_actions.apply_checkpoint(plan, slice_obj, turn)

    def _apply_final_report(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        self.terminal_actions.apply_final_report(plan, slice_obj, turn)

    def _apply_abort(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        self.terminal_actions.apply_abort(
            plan,
            slice_obj,
            turn,
            soft_abort_reason_codes=_SOFT_ABORT_REASON_CODES,
        )

    def _abort_dependency_blocked_slices(self, plan: ExecutionPlan, *, scope_plan_id: str | None = None) -> bool:
        max_retries = int(getattr(self.config.direct_execution, "max_blocked_retries", 3) or 0)
        plan_slice_ids = {s.slice_id for s in plan.slices}
        changed = False
        for slice_obj in plan.slices:
            if slice_obj.is_terminal or self._is_blocked_checkpoint(slice_obj):
                continue
            blockers = []
            for dep_id in slice_obj.depends_on:
                # When scoped, skip cross-plan dependencies — they may resolve later
                dep_slice = self._resolve_dependency_slice(plan, dep_id)
                if dep_slice is None:
                    continue
                if dep_slice.status == "completed" and not dependency_unblocked_by(dep_slice):
                    blockers.append(dep_id)
                    continue
                if scope_plan_id and dep_id not in plan_slice_ids:
                    continue
                if dep_slice.status in {"failed", "aborted"} or (
                    self._is_blocked_checkpoint(dep_slice)
                    and dep_slice.blocked_retry_count >= max_retries
                ):
                    blockers.append(dep_id)
            if not blockers:
                continue
            blocker_slice = self._resolve_dependency_slice(plan, blockers[0])
            blocker_reason_code = ""
            if blocker_slice is not None:
                blocker_reason_code = (
                    str(getattr(blocker_slice, "last_error", "") or "")
                    or acceptance_blocker_reason(blocker_slice)
                )
            blocker_class = blocker_class_from_slice(blocker_slice) if blocker_slice is not None else "unknown"
            slice_obj.status = "aborted"
            slice_obj.acceptance_state = "blocked"
            slice_obj.last_error = "dependency_blocked"
            slice_obj.last_summary = f"Blocked by prerequisite slice(s): {', '.join(blockers)}"
            slice_obj.dependency_blocker_slice_id = blockers[0]
            slice_obj.dependency_blocker_reason_code = blocker_reason_code
            slice_obj.dependency_blocker_class = blocker_class
            slice_obj.facts["dependency.blocking_slice_ids"] = list(blockers)
            if blocker_reason_code:
                slice_obj.facts["dependency.blocker_reason_code"] = blocker_reason_code
            if blocker_class:
                slice_obj.facts["dependency.blocker_class"] = blocker_class
            changed = True
        if changed:
            plan.touch()
            self._persist_plan_snapshot(plan)
        return changed

    def _dependencies_satisfied(self, plan: ExecutionPlan, slice_obj: PlanSlice) -> bool:
        if not slice_obj.depends_on:
            return True
        return all(
            (dep_slice := self._resolve_dependency_slice(plan, dep_id)) is not None and dependency_unblocked_by(dep_slice)
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

    async def _finalize_plan_if_ready(self, plan: ExecutionPlan) -> None:
        if plan.is_terminal:
            return
        max_retries = int(getattr(self.config.direct_execution, "max_blocked_retries", 3) or 0)
        if any(slice_obj.status == "failed" for slice_obj in plan.slices):
            plan.status = "failed"
            plan.last_error = _aggregate_plan_last_error(plan.slices, prefer_status="failed")
            self.state.consecutive_failed_plans += 1
            self.plan_source.mark_plan_failed(plan, self.state)
        elif all(slice_obj.status == "completed" and dependency_unblocked_by(slice_obj) for slice_obj in plan.slices):
            plan.status = "completed"
            plan.last_error = ""
            self.state.completed_plan_count += 1
            self.state.consecutive_failed_plans = 0
            self.plan_source.mark_plan_complete(plan, self.state)
        elif all(
            slice_obj.status in {"completed", "aborted"}
            or (
                self._is_blocked_checkpoint(slice_obj)
                and slice_obj.blocked_retry_count >= max_retries
            )
            for slice_obj in plan.slices
        ):
            # Don't stop if any aborted slices have cross-plan dependencies
            # that might be satisfied by future batches.
            plan_slice_ids = {s.slice_id for s in plan.slices}
            has_deferred = any(
                slice_obj.status == "aborted"
                and slice_obj.last_error == "dependency_blocked"
                and any(
                    dep_id not in plan_slice_ids
                    for dep_id in slice_obj.depends_on
                )
                for slice_obj in plan.slices
            )
            if has_deferred:
                return
            plan.status = "stopped"
            plan.last_error = _aggregate_plan_last_error(plan.slices, prefer_status="aborted") or next(
                (
                    acceptance_blocker_reason(slice_obj)
                    for slice_obj in plan.slices
                    if slice_obj.status == "completed" and not dependency_unblocked_by(slice_obj)
                ),
                "",
            )
            self.plan_source.mark_plan_failed(plan, self.state)
        else:
            acceptance_blocker = self._uniform_acceptance_blocker(plan)
            if not acceptance_blocker:
                return
            plan.status = "stopped"
            plan.last_error = acceptance_blocker
            self.plan_source.mark_plan_failed(plan, self.state)
        plan.touch()
        self._persist_plan_snapshot(plan)
        if self.console_controller is not None:
            self.console_controller.sync_sequence_progress(plan=plan, all_plans=self.state.plans, active_slice_id=None)
        await self._maybe_send_sequence_report(plan)

    def _stop_reason_after_round(self, plan: ExecutionPlan, *, did_work: bool) -> StopReason | None:
        if self.orch._stop_requested:
            return StopReason.GRACEFUL_STOP
        if plan.status == "failed" and self.state.consecutive_failed_plans >= int(self.config.max_consecutive_failed_plans or 1):
            return StopReason.MAX_ERRORS
        if not did_work and self.state.no_progress_cycles >= int(self.config.max_empty_cycles or 1):
            return StopReason.NO_PROGRESS
        return None

    def _uniform_acceptance_blocker(self, plan: ExecutionPlan) -> str:
        blockers: list[str] = []
        for slice_obj in plan.slices:
            if slice_obj.is_terminal or self._is_blocked_checkpoint(slice_obj):
                continue
            if not slice_obj.depends_on:
                return ""
            dep_reasons: list[str] = []
            for dep_id in slice_obj.depends_on:
                dep_slice = self._resolve_dependency_slice(plan, dep_id)
                if dep_slice is None:
                    return ""
                if dep_slice.status != "completed" or dependency_unblocked_by(dep_slice):
                    return ""
                dep_reason = acceptance_blocker_reason(dep_slice)
                if not dep_reason:
                    return ""
                dep_reasons.append(dep_reason)
            blockers.append(dep_reasons[0])
        if blockers and len(set(blockers)) == 1:
            return blockers[0]
        return ""

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
        downstream_needs = self._downstream_prereq_facts_expected_from(plan, slice_obj)
        if downstream_needs and "direct.missing_downstream_prerequisites" not in facts:
            facts["direct.missing_downstream_prerequisites"] = downstream_needs
        return normalize_backtests_facts(facts)

    def _downstream_prereq_facts_expected_from(
        self,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
    ) -> list[str]:
        """Facts downstream slices declare as prerequisites that this slice is
        expected to produce (i.e. not already guaranteed by some other ancestor
        of those downstream slices)."""
        by_id = {item.slice_id: item for item in plan.slices}
        own_outputs = {
            str(item).strip()
            for item in (slice_obj.required_output_facts or [])
            if str(item).strip()
        }
        expected: list[str] = []
        for candidate in plan.slices:
            if slice_obj.slice_id not in (candidate.depends_on or []):
                continue
            prereqs = [
                str(item).strip()
                for item in (candidate.required_prerequisite_facts or [])
                if str(item).strip()
            ]
            if not prereqs:
                continue
            other_ancestor_outputs: set[str] = set()
            for dep_id in candidate.depends_on or []:
                if dep_id == slice_obj.slice_id or dep_id not in by_id:
                    continue
                dep = by_id[dep_id]
                for fact in dep.required_output_facts or []:
                    normalized = str(fact or "").strip()
                    if normalized:
                        other_ancestor_outputs.add(normalized)
            for fact in prereqs:
                if fact in own_outputs or fact in other_ancestor_outputs:
                    continue
                if fact not in expected:
                    expected.append(fact)
        return expected

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

    async def _maybe_send_sequence_report(self, plan: ExecutionPlan) -> None:
        """Send a Telegram report with LLM narrative when a compiled sequence finishes."""
        ns = getattr(self, "notification_service", None)
        if ns is None:
            return
        from app.services.direct_execution.sequence_reporter import build_and_send_sequence_report
        await build_and_send_sequence_report(
            plan=plan, state=self.state, config=self.config,
            plan_source=self.plan_source, notification_service=ns,
        )

__all__ = ["DirectExecutionService", "_DRAIN_TIMEOUT_SECONDS"]
