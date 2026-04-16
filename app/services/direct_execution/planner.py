"""
Structured planner service for direct execution.

Produces a SemanticRawPlan via LLM, then feeds it through the same
deterministic compiler (compile_semantic_raw_plan) used by the converter
pipeline — guaranteeing identical tool expansion, budget assignment,
batching, and reconciliation.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.adapters.base import BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionPlan
from app.planner_semantic_parsing import parse_planner_semantic_output
from app.planner_semantic_prompts import build_planner_semantic_prompt
from app.raw_plan_compiler import compile_semantic_raw_plan
from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.direct_execution.invocation import AdapterInvocationCancelled, AdapterInvocationError, invoke_adapter_with_retries


class PlannerDecisionError(RuntimeError):
    """Raised when the planner cannot produce a valid execution plan."""


class PlannerDecisionCancelled(PlannerDecisionError):
    """Raised when planner invocation is cancelled by runtime shutdown."""


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
        invoker: Any = invoke_adapter_with_retries,
        catalog_snapshot: McpCatalogSnapshot | None = None,
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.operator_directives = operator_directives
        self.observer = observer
        self.invoker = invoker
        self._pending_plans: list[ExecutionPlan] = []
        self.catalog_snapshot = catalog_snapshot

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
        # Return a queued plan from a previous multi-batch compilation.
        if self._pending_plans:
            plan = self._pending_plans.pop(0)
            self.artifact_store.save_plan(plan)
            return plan

        prompt = build_planner_semantic_prompt(
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
            response = await self.invoker(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.retry_attempts,
                base_backoff_seconds=self.retry_backoff_seconds,
                on_attempt_start=self._on_attempt_start,
                on_attempt_retry=self._on_attempt_retry,
                on_attempt_finish=self._on_attempt_finish,
                process_owner="planner",
            )
        except AdapterInvocationCancelled as exc:
            raise PlannerDecisionCancelled(str(exc)) from exc
        except AdapterInvocationError as exc:
            raise PlannerDecisionError(str(exc)) from exc
        if not response.success:
            raise PlannerDecisionError(response.error or "planner_invoke_failed")
        try:
            document, semantic_plan = parse_planner_semantic_output(
                response.raw_output,
                goal=goal,
                baseline_bootstrap=baseline_bootstrap,
            )
        except Exception as exc:
            raise PlannerDecisionError(str(exc)) from exc
        sequence = compile_semantic_raw_plan(
            document,
            semantic_plan,
            semantic_method="planner_llm",
            plan_source_kind="planner",
            catalog_snapshot=self.catalog_snapshot,
        )
        if not sequence.plans:
            raise PlannerDecisionError("planner_compilation_produced_no_plans")
        # Queue all plans beyond the first for subsequent calls.
        for plan in sequence.plans[1:]:
            self._pending_plans.append(plan)
        plan = sequence.plans[0]
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
