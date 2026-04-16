"""
Terminal action application helpers for direct execution slices.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from app.execution_models import ExecutionPlan, ExecutionTurn, PlanSlice, WorkerAction, make_id
from app.runtime_console import slot_for_slice
from app.services.direct_execution.acceptance import acceptance_blocker_reason, accepted_completion_state
from app.services.direct_execution.executor import DirectExecutionResult
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts
from app.services.direct_execution.guardrails import final_report_passes_quality_gate, is_prerequisite_block_terminal
from app.services.direct_execution.slice_readiness import downstream_prerequisites_blocker


class TerminalActionApplicator:
    def __init__(
        self,
        *,
        artifact_store: Any,
        incident_store: Any,
        console_controller: Any,
        known_facts_for_slice: Callable[[ExecutionPlan, PlanSlice], dict[str, Any]],
        required_output_facts_for_slice: Callable[[ExecutionPlan, PlanSlice], list[str]],
        persist_plan_snapshot: Callable[[ExecutionPlan], None],
    ) -> None:
        self.artifact_store = artifact_store
        self.incident_store = incident_store
        self.console_controller = console_controller
        self.known_facts_for_slice = known_facts_for_slice
        self.required_output_facts_for_slice = required_output_facts_for_slice
        self.persist_plan_snapshot = persist_plan_snapshot

    def apply_checkpoint(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        action = turn.action
        if action.status == "complete":
            self.apply_final_report(plan, slice_obj, turn)
            return
        slice_obj.status = "checkpointed"
        slice_obj.acceptance_state = "pending"
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
            self.console_controller.on_slice_checkpoint(
                slot=slot_for_slice(slice_obj.parallel_slot),
                summary=slice_obj.last_summary,
                operation_ref="",
                operation_status="",
            )

    def apply_final_report(self, plan: ExecutionPlan, slice_obj: PlanSlice, turn: ExecutionTurn) -> None:
        action = turn.action
        known_facts = self.known_facts_for_slice(plan, slice_obj)
        required_output_facts = self.required_output_facts_for_slice(plan, slice_obj)
        if is_prerequisite_block_terminal(action, known_facts):
            fallback_level = int(action.facts.get("direct.fallback_provider_index", 0))
            action.facts["direct.skipped_by_prerequisite"] = True
            slice_obj.status = "completed"
            slice_obj.acceptance_state = "reported_terminal"
            slice_obj.final_report_turn_id = turn.turn_id
            slice_obj.verdict = "SKIP"
            slice_obj.last_summary = action.summary or "Skipped: prerequisite slice declared downstream work not required"
            slice_obj.last_error = acceptance_blocker_reason(slice_obj)
            slice_obj.last_checkpoint_status = ""
            self._clear_dependency_blocker(slice_obj)
            slice_obj.facts.update(action.facts)
            slice_obj.facts["direct.skipped_by_prerequisite"] = True
            slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
            self.artifact_store.save_report(
                plan_id=plan.plan_id,
                slice_id=slice_obj.slice_id,
                turn_id=turn.turn_id,
                payload=asdict(action) | {"type": action.action_type},
            )
            if self.console_controller is not None:
                self.console_controller.on_slice_completed(
                    slot=slot_for_slice(slice_obj.parallel_slot),
                    summary=slice_obj.last_summary,
                    via="direct",
                    fallback_level=fallback_level,
                )
            return
        passes_gate, gate_reason = final_report_passes_quality_gate(
            tool_call_count=int(turn.direct_attempt.tool_call_count or 0),
            action=action,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )
        if not passes_gate:
            self.incident_store.record(
                summary="Direct final_report rejected by quality gate",
                metadata={
                    "plan_id": plan.plan_id,
                    "slice_id": slice_obj.slice_id,
                    "slice_title": slice_obj.title,
                    "reason": gate_reason,
                    "provider": turn.direct_attempt.provider,
                    "tool_call_count": turn.direct_attempt.tool_call_count,
                    "runtime_profile": slice_obj.runtime_profile,
                    "evidence_refs": list(action.evidence_refs or []),
                },
                source="direct_runtime",
                severity="medium",
            )
            self.checkpoint_blocked(
                plan,
                slice_obj,
                summary=(
                    f"Direct final_report for slice '{slice_obj.slice_id}' failed evidence quality gate: "
                    f"{gate_reason}."
                ),
                reason_code="direct_final_report_quality_gate_failed",
            )
            return
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )
        if readiness.missing_required_facts:
            self.checkpoint_blocked(
                plan,
                slice_obj,
                summary=(
                    f"Direct final_report for slice '{slice_obj.slice_id}' is missing downstream facts: "
                    f"{', '.join(readiness.missing_required_facts)}."
                ),
                reason_code="direct_slice_missing_prerequisite_facts",
            )
            return
        downstream_blocker = downstream_prerequisites_blocker(
            plan,
            slice_obj,
            resolve_dependency=lambda dep_id: self._resolve_dependency_slice(plan, dep_id),
            hydrated_facts=dict(readiness.facts),
        )
        if downstream_blocker is not None:
            slice_obj.facts.setdefault("direct.missing_downstream_prerequisites", downstream_blocker.missing_facts)
            self.incident_store.record(
                summary="Slice blocked by downstream prerequisite gap",
                metadata={
                    "plan_id": plan.plan_id,
                    "slice_id": slice_obj.slice_id,
                    "slice_title": slice_obj.title,
                    "runtime_profile": slice_obj.runtime_profile,
                    "missing_facts": downstream_blocker.missing_facts,
                    "downstream_slice_ids": downstream_blocker.blocking_slice_ids,
                },
                source="direct_runtime",
                severity="medium",
            )
            self.checkpoint_blocked(
                plan,
                slice_obj,
                summary=(
                    f"Direct final_report for slice '{slice_obj.slice_id}' passed quality gate, "
                    f"but downstream slices need facts not yet available: "
                    f"{', '.join(downstream_blocker.missing_facts)}. "
                    f"Produce these facts before completing."
                ),
                reason_code="direct_slice_missing_downstream_prerequisites",
            )
            return
        fallback_level = int(action.facts.get("direct.fallback_provider_index", 0))
        action.facts = dict(readiness.facts)
        action.evidence_refs = list(readiness.evidence_refs)
        slice_obj.status = "completed"
        slice_obj.verdict = str(action.verdict or "").strip()
        slice_obj.acceptance_state = accepted_completion_state(
            slice_obj=slice_obj,
            verdict=slice_obj.verdict,
        )
        slice_obj.final_report_turn_id = turn.turn_id
        slice_obj.last_summary = action.summary
        slice_obj.last_error = acceptance_blocker_reason(slice_obj)
        slice_obj.last_checkpoint_status = ""
        self._clear_dependency_blocker(slice_obj)
        slice_obj.facts.update(readiness.facts)
        slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
        self.artifact_store.save_report(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload=asdict(action) | {"type": action.action_type},
        )
        if self.console_controller is not None:
            self.console_controller.on_slice_completed(
                slot=slot_for_slice(slice_obj.parallel_slot),
                summary=action.summary,
                via="direct",
                fallback_level=fallback_level,
            )

    def apply_abort(
        self,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        turn: ExecutionTurn,
        *,
        soft_abort_reason_codes: set[str],
    ) -> None:
        action = turn.action
        if action.reason_code in soft_abort_reason_codes or action.retryable:
            self.checkpoint_blocked(
                plan,
                slice_obj,
                summary=action.summary,
                reason_code=action.reason_code or "direct_soft_blocker",
            )
            return
        fallback_level = int(action.facts.get("direct.fallback_provider_index", 0))
        slice_obj.status = "failed"
        slice_obj.acceptance_state = "blocked"
        slice_obj.last_error = action.reason_code or action.reason or "direct_abort"
        slice_obj.last_summary = action.summary
        slice_obj.verdict = str(action.verdict or "").strip()
        self._clear_dependency_blocker(slice_obj)
        slice_obj.facts.update(action.facts)
        slice_obj.artifacts.extend(item for item in action.artifacts if item not in slice_obj.artifacts)
        self.artifact_store.save_report(
            plan_id=plan.plan_id,
            slice_id=slice_obj.slice_id,
            turn_id=turn.turn_id,
            payload=asdict(action) | {"type": action.action_type},
        )
        if self.console_controller is not None:
            self.console_controller.on_slice_failed(
                slot=slot_for_slice(slice_obj.parallel_slot),
                summary=action.summary,
                fallback_level=fallback_level,
            )

    def checkpoint_blocked(
        self,
        plan: ExecutionPlan,
        slice_obj: PlanSlice,
        *,
        summary: str,
        reason_code: str,
    ) -> None:
        slice_obj.status = "checkpointed"
        slice_obj.acceptance_state = "blocked"
        slice_obj.last_checkpoint_status = "blocked"
        slice_obj.last_checkpoint_summary = summary
        slice_obj.last_summary = summary
        slice_obj.last_error = reason_code
        self._clear_dependency_blocker(slice_obj)
        if self.console_controller is not None:
            self.console_controller.on_slice_checkpoint(
                slot=slot_for_slice(slice_obj.parallel_slot),
                summary=summary,
                operation_ref="",
                operation_status="",
            )
        self.persist_plan_snapshot(plan)

    @staticmethod
    def blocked_action_from_failed_direct(result: DirectExecutionResult) -> WorkerAction:
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

    @staticmethod
    def _resolve_dependency_slice(plan: ExecutionPlan, dep_id: str) -> PlanSlice | None:
        local = {item.slice_id: item for item in plan.slices}
        return local.get(dep_id)

    @staticmethod
    def _clear_dependency_blocker(slice_obj: PlanSlice) -> None:
        slice_obj.dependency_blocker_slice_id = ""
        slice_obj.dependency_blocker_reason_code = ""
        slice_obj.dependency_blocker_class = ""

    @staticmethod
    def _is_zero_tool_call_stall(slice_obj: PlanSlice) -> bool:
        return (
            slice_obj.last_checkpoint_status == "partial"
            and slice_obj.turn_count >= 2
            and slice_obj.tool_call_count == 0
        )


__all__ = ["TerminalActionApplicator"]
