"""
Worker decision service for brokered slice turns.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any

from app.adapters.base import BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice, WorkerAction, make_id
from app.execution_parsing import StructuredOutputError, parse_worker_action_output
from app.plan_prompts import build_brokered_worker_correction_prompt, build_brokered_worker_prompt
from app.services.brokered_execution.invocation import AdapterInvocationError, invoke_adapter_with_retries

logger = logging.getLogger("orchestrator.worker_decision")


class WorkerDecisionError(RuntimeError):
    """Raised when the worker fails to produce a valid action."""


class WorkerTerminalActionError(WorkerDecisionError):
    """Raised when the current slice turn should terminalize after invalid worker output."""

    def __init__(
        self,
        message: str,
        *,
        artifact_path: str,
        parse_error: str,
        raw_output: str,
    ) -> None:
        super().__init__(message)
        self.artifact_path = artifact_path
        self.parse_error = parse_error
        self.raw_output = raw_output


class WorkerContractViolationError(WorkerTerminalActionError):
    """Raised when worker output violates the public broker contract."""


class WorkerParseFailureError(WorkerTerminalActionError):
    """Raised when worker output cannot be parsed into a structured broker action."""


class WorkerDecisionService:
    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        artifact_store: ExecutionArtifactStore,
        timeout_seconds: int,
        retry_attempts: int,
        retry_backoff_seconds: float,
        worker_system_prompt: str = "",
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.worker_system_prompt = worker_system_prompt

    async def choose_action(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        baseline_bootstrap: dict[str, Any],
        known_facts: dict[str, Any],
        recent_turn_summaries: list[str],
        latest_tool_summary: str,
        remaining_budget: dict[str, int],
        checkpoint_summary: str,
        active_operation: dict[str, Any],
    ) -> WorkerAction:
        prompt = build_brokered_worker_prompt(
            plan_id=plan_id,
            slice_payload=asdict(slice_obj),
            worker_system_prompt=self.worker_system_prompt,
            baseline_bootstrap=baseline_bootstrap,
            known_facts=known_facts,
            recent_turn_summaries=recent_turn_summaries,
            latest_tool_summary=latest_tool_summary,
            remaining_budget=remaining_budget,
            checkpoint_summary=checkpoint_summary,
            active_operation=active_operation,
        )
        try:
            response = await invoke_adapter_with_retries(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.retry_attempts,
                base_backoff_seconds=self.retry_backoff_seconds,
                system_prompt=self.worker_system_prompt,
            )
        except AdapterInvocationError as exc:
            raise WorkerDecisionError(str(exc)) from exc
        if not response.success:
            if response.timed_out:
                failure_path = self._save_parse_failure(
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    allowed_tools=list(slice_obj.allowed_tools),
                    prompt=prompt,
                    response=response,
                    error="worker_invoke_timeout",
                )
                raise WorkerParseFailureError(
                    f"worker_invoke_timeout [artifact={failure_path}]",
                    artifact_path=str(failure_path),
                    parse_error="worker_invoke_timeout",
                    raw_output=response.raw_output,
                )
            raise WorkerDecisionError(response.error or "worker_invoke_failed")
        return await self._parse_with_correction(
            plan_id=plan_id,
            slice_obj=slice_obj,
            prompt=prompt,
            response=response,
            latest_tool_summary=latest_tool_summary,
            baseline_bootstrap=baseline_bootstrap,
            active_operation=active_operation,
        )

    async def _parse_with_correction(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        prompt: str,
        response: Any,
        latest_tool_summary: str,
        baseline_bootstrap: dict[str, Any],
        active_operation: dict[str, Any],
    ) -> WorkerAction:
        allowlist = set(slice_obj.allowed_tools)
        corrective_retry_used = False
        try:
            action = parse_worker_action_output(response.raw_output, allowlist=allowlist)
        except StructuredOutputError as exc:
            first_failure = self._save_parse_failure(
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                allowed_tools=list(slice_obj.allowed_tools),
                prompt=prompt,
                response=response,
                error=str(exc),
            )
            if not self._eligible_for_corrective_retry(exc):
                raise WorkerParseFailureError(
                    f"{exc} [artifact={first_failure}]",
                    artifact_path=str(first_failure),
                    parse_error=str(exc),
                    raw_output=response.raw_output,
                ) from exc
            logger.warning(
                "Worker contract violation for %s/%s: %s. Retrying once with corrective prompt (artifact=%s)",
                plan_id,
                slice_obj.slice_id,
                exc,
                first_failure,
            )
            corrective_prompt = build_brokered_worker_correction_prompt(
                previous_prompt=prompt,
                raw_output=response.raw_output,
                allowed_tools=list(slice_obj.allowed_tools),
                parse_error=str(exc),
            )
            try:
                corrective = await invoke_adapter_with_retries(
                    adapter=self.adapter,
                    prompt=corrective_prompt,
                    timeout_seconds=self.timeout_seconds,
                    max_attempts=1,
                    base_backoff_seconds=self.retry_backoff_seconds,
                    system_prompt=self.worker_system_prompt,
                )
            except AdapterInvocationError as retry_exc:
                raise self._terminal_parse_error(
                    parse_error=str(exc),
                    message=f"{exc} [artifact={first_failure}] corrective_retry_failed:{retry_exc}",
                    artifact_path=str(first_failure),
                    raw_output=response.raw_output,
                ) from retry_exc
            if not corrective.success:
                raise self._terminal_parse_error(
                    parse_error=str(exc),
                    message=f"{exc} [artifact={first_failure}] corrective_retry_failed:{corrective.error or 'worker_invoke_failed'}",
                    artifact_path=str(first_failure),
                    raw_output=response.raw_output,
                )
            try:
                action = parse_worker_action_output(corrective.raw_output, allowlist=allowlist)
                corrective_retry_used = True
            except StructuredOutputError as corrective_exc:
                second_failure = self._save_parse_failure(
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    allowed_tools=list(slice_obj.allowed_tools),
                    prompt=corrective_prompt,
                    response=corrective,
                    error=str(corrective_exc),
                )
                logger.warning(
                    "Worker corrective retry failed for %s/%s: %s (artifact=%s)",
                    plan_id,
                    slice_obj.slice_id,
                    corrective_exc,
                    second_failure,
                )
                raise self._terminal_parse_error(
                    parse_error=str(corrective_exc),
                    message=f"{corrective_exc} [artifact={second_failure}]",
                    artifact_path=str(second_failure),
                    raw_output=corrective.raw_output,
                ) from corrective_exc
        semantic_error = self._semantic_error_for_action(
            action=action,
            slice_obj=slice_obj,
            latest_tool_summary=latest_tool_summary,
            baseline_bootstrap=baseline_bootstrap,
            active_operation=active_operation,
        )
        if semantic_error is None:
            return action
        if corrective_retry_used:
            raise WorkerContractViolationError(
                f"{semantic_error} [artifact=semantic_validation]",
                artifact_path="semantic_validation",
                parse_error=semantic_error,
                raw_output=response.raw_output,
            )
        logger.warning(
            "Worker semantic contract drift for %s/%s: %s. Retrying once with corrective prompt.",
            plan_id,
            slice_obj.slice_id,
            semantic_error,
        )
        corrective_prompt = build_brokered_worker_correction_prompt(
            previous_prompt=prompt,
            raw_output=response.raw_output,
            allowed_tools=list(slice_obj.allowed_tools),
            parse_error=semantic_error,
        )
        try:
            corrective = await invoke_adapter_with_retries(
                adapter=self.adapter,
                prompt=corrective_prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=1,
                base_backoff_seconds=self.retry_backoff_seconds,
                system_prompt=self.worker_system_prompt,
            )
        except AdapterInvocationError as retry_exc:
            raise WorkerContractViolationError(
                f"{semantic_error} corrective_retry_failed:{retry_exc}",
                artifact_path="semantic_validation",
                parse_error=semantic_error,
                raw_output=response.raw_output,
            ) from retry_exc
        if not corrective.success:
            raise WorkerContractViolationError(
                f"{semantic_error} corrective_retry_failed:{corrective.error or 'worker_invoke_failed'}",
                artifact_path="semantic_validation",
                parse_error=semantic_error,
                raw_output=corrective.raw_output,
            )
        try:
            corrected_action = parse_worker_action_output(corrective.raw_output, allowlist=allowlist)
        except StructuredOutputError as corrective_exc:
            second_failure = self._save_parse_failure(
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                allowed_tools=list(slice_obj.allowed_tools),
                prompt=corrective_prompt,
                response=corrective,
                error=str(corrective_exc),
            )
            raise self._terminal_parse_error(
                parse_error=str(corrective_exc),
                message=f"{corrective_exc} [artifact={second_failure}]",
                artifact_path=str(second_failure),
                raw_output=corrective.raw_output,
            ) from corrective_exc
        semantic_error = self._semantic_error_for_action(
            action=corrected_action,
            slice_obj=slice_obj,
            latest_tool_summary=latest_tool_summary,
            baseline_bootstrap=baseline_bootstrap,
            active_operation=active_operation,
        )
        if semantic_error is not None:
            raise WorkerContractViolationError(
                semantic_error,
                artifact_path="semantic_validation",
                parse_error=semantic_error,
                raw_output=corrective.raw_output,
            )
        return corrected_action

    @staticmethod
    def _semantic_error_for_action(
        *,
        action: WorkerAction,
        slice_obj: PlanSlice,
        latest_tool_summary: str,
        baseline_bootstrap: dict[str, Any],
        active_operation: dict[str, Any],
    ) -> str | None:
        if action.action_type == "tool_call" and action.tool == "features_custom":
            requested_action = str(action.arguments.get("action", "") or "").strip().lower()
            if requested_action == "create":
                return "features_custom_create_forbidden"
        if action.action_type == "tool_call" and action.tool == "research_search":
            baseline_snapshot_id = str(baseline_bootstrap.get("baseline_snapshot_id", "") or "").strip()
            project_id = str(action.arguments.get("project_id", "") or "").strip()
            if project_id and baseline_snapshot_id and project_id == baseline_snapshot_id:
                return "research_search_snapshot_id_as_project_id_forbidden"
        if action.action_type == "tool_call" and action.tool == "features_analytics":
            requested_action = str(action.arguments.get("action", "") or "").strip().lower()
            if requested_action in {"analytics", "heatmap", "render", "portability"}:
                selectors = ("feature_name", "feature", "column_name", "column", "name")
                if not any(str(action.arguments.get(key, "") or "").strip() for key in selectors):
                    return "features_analytics_feature_selector_required"
                if "not ready" in latest_tool_summary.lower() and "analytics" in latest_tool_summary.lower():
                    return "features_analytics_not_ready_retry_forbidden"
        if action.action_type == "tool_call" and action.tool == "events_sync":
            family = str(action.arguments.get("family", "") or "").strip().lower()
            scope = str(action.arguments.get("scope", "") or "").strip().lower()
            if not family:
                return "events_sync_family_required"
            if not scope:
                return "events_sync_scope_required"
            active_tool = str(active_operation.get("tool", "") or "").strip()
            active_status = str(active_operation.get("status", "") or "").strip().lower()
            if active_tool == "events_sync" and active_status in {"queued", "pending", "running", "started", "running_compute", "in_progress"}:
                return "duplicate_events_sync_while_active_operation_running"
        if action.action_type == "tool_call" and action.tool == "datasets_preview":
            dataset_id = str(action.arguments.get("dataset_id", "") or "").strip()
            view = str(action.arguments.get("view", "") or "").strip().lower()
            if not dataset_id:
                return "datasets_preview_dataset_id_required"
            if view not in {"rows", "chart"}:
                return "datasets_preview_view_required"
        if action.action_type == "tool_call" and action.tool in set(slice_obj.allowed_tools):
            return None
        texts = [latest_tool_summary]
        if action.action_type != "tool_call":
            texts.extend([action.summary, action.reason])
            texts.extend(action.pending_questions)
        texts.extend(issue.summary for issue in action.reportable_issues)
        texts.extend(issue.details for issue in action.reportable_issues)
        combined = " ".join(part.strip().lower() for part in texts if part)
        if (
            "tool registry" in combined
            or "not found in registry" in combined
            or "only standard development tools" in combined
            or "not available in this session" in combined
        ):
            allowlist = [tool for tool in slice_obj.allowed_tools if tool and tool.lower() in combined]
            if allowlist and "tool not found in registry" not in latest_tool_summary.lower():
                return f"contradictory_tool_registry_claim:{','.join(sorted(allowlist))}"
        if "tool registry" not in combined and "not found in registry" not in combined:
            return None
        if "tool not found in registry" in latest_tool_summary.lower():
            return None
        allowlist = [tool for tool in slice_obj.allowed_tools if tool and tool.lower() in combined]
        if not allowlist:
            return None
        return f"contradictory_tool_registry_claim:{','.join(sorted(allowlist))}"

    def _save_parse_failure(
        self,
        *,
        plan_id: str,
        slice_id: str,
        allowed_tools: list[str],
        prompt: str,
        response: Any,
        error: str,
    ) -> str:
        failure_path = self.artifact_store.save_worker_parse_failure(
            plan_id=plan_id,
            slice_id=slice_id,
            payload={
                "failure_id": make_id("worker_parse_failure"),
                "error": error,
                "plan_id": plan_id,
                "slice_id": slice_id,
                "allowed_tools": list(allowed_tools),
                "prompt": prompt,
                "raw_output": response.raw_output,
                "response_error": response.error,
                "timed_out": response.timed_out,
                "exit_code": response.exit_code,
            },
        )
        logger.warning(
            "Worker action parse failed for %s/%s: %s (artifact=%s)",
            plan_id,
            slice_id,
            error,
            failure_path,
        )
        return str(failure_path)

    @staticmethod
    def _eligible_for_corrective_retry(exc: StructuredOutputError) -> bool:
        text = str(exc)
        return (
            text.startswith("tool_not_in_allowlist:")
            or text.startswith("tool_prefixed_namespace_forbidden:")
            or text in {"json_object_not_found", "balanced_json_object_not_found", "worker_action_type_invalid"}
        )

    @staticmethod
    def _terminal_parse_error(
        *,
        parse_error: str,
        message: str,
        artifact_path: str,
        raw_output: str,
    ) -> WorkerTerminalActionError:
        if parse_error.startswith("tool_not_in_allowlist:") or parse_error.startswith("tool_prefixed_namespace_forbidden:"):
            return WorkerContractViolationError(
                message,
                artifact_path=artifact_path,
                parse_error=parse_error,
                raw_output=raw_output,
            )
        return WorkerParseFailureError(
            message,
            artifact_path=artifact_path,
            parse_error=parse_error,
            raw_output=raw_output,
        )
