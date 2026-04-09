"""
Structured planner service for brokered execution.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.adapters.base import BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionPlan
from app.execution_parsing import StructuredOutputError, parse_execution_plan_output
from app.plan_prompts import build_brokered_plan_creation_prompt
from app.services.brokered_execution.budgeting import normalize_plan_budgets
from app.services.brokered_execution.invocation import AdapterInvocationError, invoke_adapter_with_retries


class PlannerDecisionError(RuntimeError):
    """Raised when the planner cannot produce a valid execution plan."""


class PlannerDecisionService:
    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        artifact_store: ExecutionArtifactStore,
        timeout_seconds: int,
        retry_attempts: int,
        retry_backoff_seconds: float,
        operator_directives: str = "",
        observer: Any | None = None,
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.operator_directives = operator_directives
        self.observer = observer

    async def create_plan(
        self,
        *,
        goal: str,
        baseline_bootstrap: dict[str, Any],
        available_tools: list[str],
        worker_count: int,
        plan_version: int,
        previous_state_summary: str,
        previous_blockers: list[str] | None = None,
    ) -> ExecutionPlan:
        prompt = build_brokered_plan_creation_prompt(
            goal=goal,
            operator_directives=self.operator_directives,
            baseline_bootstrap=baseline_bootstrap,
            plan_version=plan_version,
            worker_count=worker_count,
            available_tools=available_tools,
            previous_state_summary=previous_state_summary,
            previous_blockers=previous_blockers,
        )
        try:
            response = await invoke_adapter_with_retries(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.retry_attempts,
                base_backoff_seconds=self.retry_backoff_seconds,
                on_attempt_start=self._on_attempt_start,
                on_attempt_retry=self._on_attempt_retry,
                on_attempt_finish=self._on_attempt_finish,
            )
        except AdapterInvocationError as exc:
            raise PlannerDecisionError(str(exc)) from exc
        if not response.success:
            raise PlannerDecisionError(response.error or "planner_invoke_failed")
        try:
            plan = parse_execution_plan_output(response.raw_output)
        except StructuredOutputError as exc:
            raise PlannerDecisionError(str(exc)) from exc
        plan = normalize_plan_budgets(plan)
        unknown_tools = sorted(
            {
                tool_name
                for slice_obj in plan.slices
                for tool_name in slice_obj.allowed_tools
                if tool_name not in set(available_tools)
            }
        )
        if unknown_tools:
            raise PlannerDecisionError(f"planner_used_unknown_tools:{','.join(unknown_tools)}")
        self.artifact_store.save_plan(plan)
        return plan

    def _on_attempt_start(self, *, attempt: int, max_attempts: int) -> None:
        if self.observer is not None:
            self.observer.on_planner_started(
                adapter_name=self.adapter.name(),
                attempt=attempt,
                max_attempts=max_attempts,
            )

    def _on_attempt_retry(self, *, error: str, attempt: int, max_attempts: int) -> None:
        if self.observer is not None:
            self.observer.on_planner_retry(error=error, attempt=attempt, max_attempts=max_attempts)

    def _on_attempt_finish(self, *, success: bool, error: str, attempt: int, max_attempts: int) -> None:
        del attempt, max_attempts
        if self.observer is not None:
            self.observer.on_planner_finished(success=success, error=error)

    def save_plan_snapshot(self, plan: ExecutionPlan) -> None:
        self.artifact_store.save_plan(plan)

    @staticmethod
    def plan_payload(plan: ExecutionPlan) -> dict[str, Any]:
        return asdict(plan)
