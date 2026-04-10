"""
Fallback executor that chains primary → fallback_1 → fallback_2 providers
when a direct slice fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.adapters.base import BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor

logger = logging.getLogger("orchestrator.direct.fallback")


@dataclass
class FallbackAttempt:
    """Record of a single fallback attempt."""

    provider: str
    adapter_name: str
    result: DirectExecutionResult
    artifact_path: str
    attempt_index: int  # 1 = fallback_1, 2 = fallback_2


class FallbackExecutor:
    """Wraps a primary DirectSliceExecutor and retries on configured fallback adapters."""

    def __init__(
        self,
        *,
        primary_executor: DirectSliceExecutor,
        fallback_providers: list[str],
        artifact_store: ExecutionArtifactStore,
        incident_store: LocalIncidentStore,
        direct_config: Any,
        worker_system_prompt: str,
        adapter_factory: Callable[[str], BaseAdapter | None],
        invoker: Any | None = None,
    ) -> None:
        self.primary_executor = primary_executor
        self.fallback_providers = [
            name for name in fallback_providers if name and name.strip()
        ]
        self.artifact_store = artifact_store
        self.incident_store = incident_store
        self.direct_config = direct_config
        self.worker_system_prompt = worker_system_prompt
        self.adapter_factory = adapter_factory
        self.invoker = invoker
        self._adapter_cache: dict[str, BaseAdapter] = {}

    def _get_or_create_adapter(self, provider_name: str) -> BaseAdapter | None:
        if provider_name in self._adapter_cache:
            return self._adapter_cache[provider_name]
        adapter = self.adapter_factory(provider_name)
        if adapter is not None:
            self._adapter_cache[provider_name] = adapter
        return adapter

    def _make_fallback_executor(self, adapter: BaseAdapter) -> DirectSliceExecutor:
        if self.invoker is not None:
            return DirectSliceExecutor(
                adapter=adapter,
                artifact_store=self.artifact_store,
                incident_store=self.incident_store,
                direct_config=self.direct_config,
                worker_system_prompt=self.worker_system_prompt,
                invoker=self.invoker,
            )
        from app.services.direct_execution.invocation import invoke_adapter_with_retries

        return DirectSliceExecutor(
            adapter=adapter,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            direct_config=self.direct_config,
            worker_system_prompt=self.worker_system_prompt,
            invoker=invoke_adapter_with_retries,
        )

    @staticmethod
    def _is_success(result: DirectExecutionResult) -> bool:
        """Return True only when the result is a genuine final_report success."""
        action = result.action
        if action is None:
            return False
        return action.action_type == "final_report"

    @staticmethod
    def _build_fallback_prompt_section(
        *,
        failed_provider: str,
        error: str,
        raw_output_excerpt: str,
        attempt_index: int,
    ) -> str:
        section = (
            "## Fallback Context\n\n"
            f"The previous execution attempt on provider '{failed_provider}' (attempt #{attempt_index}) failed.\n"
        )
        if error:
            section += f"**Error:** {error[:800]}\n\n"
        if raw_output_excerpt:
            section += f"**Partial output:**\n```\n{raw_output_excerpt[:1500]}\n```\n\n"
        section += (
            "Please attempt the same task using your available tools. "
            "Learn from the previous failure and avoid repeating the same mistakes.\n"
        )
        return section

    async def execute_with_fallback(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        baseline_bootstrap: dict[str, Any],
        known_facts: dict[str, Any],
        required_output_facts: list[str],
        recent_turn_summaries: list[str],
        checkpoint_summary: str,
        on_tool_progress: Any = None,
        on_provider_switch: Any = None,
    ) -> tuple[DirectExecutionResult, list[FallbackAttempt]]:
        """Execute the slice on the primary provider, then fall back on failure."""

        # --- Primary attempt ---
        primary_result = await self.primary_executor.execute(
            plan_id=plan_id,
            slice_obj=slice_obj,
            baseline_bootstrap=baseline_bootstrap,
            known_facts=known_facts,
            required_output_facts=required_output_facts,
            recent_turn_summaries=recent_turn_summaries,
            checkpoint_summary=checkpoint_summary,
            on_tool_progress=on_tool_progress,
        )

        if self._is_success(primary_result) or not self.fallback_providers:
            return primary_result, []

        # --- Fallback chain ---
        attempts: list[FallbackAttempt] = []
        last_result = primary_result
        last_error = primary_result.error or "unknown_error"
        last_output = primary_result.raw_output

        for idx, provider_name in enumerate(self.fallback_providers, start=1):
            adapter = self._get_or_create_adapter(provider_name)
            if adapter is None:
                logger.warning("Fallback provider %r not available, skipping", provider_name)
                continue

            logger.info(
                "Attempting fallback #%d with provider %r for slice %s",
                idx, provider_name, slice_obj.slice_id,
            )

            if on_provider_switch is not None:
                try:
                    on_provider_switch(provider_name)
                except Exception:
                    pass

            extra_section = self._build_fallback_prompt_section(
                failed_provider=last_result.provider or "primary",
                error=last_error,
                raw_output_excerpt=last_output[:2000],
                attempt_index=idx - 1,
            )

            executor = self._make_fallback_executor(adapter)

            try:
                result = await executor.execute(
                    plan_id=plan_id,
                    slice_obj=slice_obj,
                    baseline_bootstrap=baseline_bootstrap,
                    known_facts=known_facts,
                    required_output_facts=required_output_facts,
                    recent_turn_summaries=recent_turn_summaries,
                    checkpoint_summary=checkpoint_summary,
                    extra_prompt_section=extra_section,
                    on_tool_progress=on_tool_progress,
                )
            except Exception as exc:
                logger.warning("Fallback executor exception for %r: %s", provider_name, exc)
                result = DirectExecutionResult(
                    action=None,
                    artifact_path="",
                    raw_output="",
                    error=str(exc),
                    provider=provider_name,
                    fallback_provider_index=idx,
                )

            # Override provider name to reflect the actual fallback
            result = DirectExecutionResult(
                action=result.action,
                artifact_path=result.artifact_path,
                raw_output=result.raw_output,
                error=result.error,
                provider=provider_name,
                duration_ms=result.duration_ms,
                tool_call_count=result.tool_call_count,
                expensive_tool_call_count=result.expensive_tool_call_count,
                parse_retry_count=result.parse_retry_count,
                fallback_provider_index=idx,
            )

            artifact_path = self.artifact_store.save_direct_fallback_attempt(
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                payload={
                    "attempt_id": f"fallback_{idx}_{slice_obj.slice_id}_{slice_obj.turn_count + 1}",
                    "provider": provider_name,
                    "status": "completed" if result.action is not None else "failed",
                    "error": result.error,
                    "raw_output_length": len(result.raw_output),
                    "duration_ms": result.duration_ms,
                    "tool_call_count": result.tool_call_count,
                    "fallback_index": idx,
                },
            )

            attempts.append(
                FallbackAttempt(
                    provider=provider_name,
                    adapter_name=adapter.name(),
                    result=result,
                    artifact_path=str(artifact_path),
                    attempt_index=idx,
                )
            )

            if self._is_success(result):
                logger.info(
                    "Fallback #%d (%s) succeeded for slice %s",
                    idx, provider_name, slice_obj.slice_id,
                )
                return result, attempts

            last_result = result
            last_error = result.error or "fallback_failed"
            last_output = result.raw_output

            logger.warning(
                "Fallback #%d (%s) also failed for slice %s: %s",
                idx, provider_name, slice_obj.slice_id, result.error[:200],
            )

        # All providers failed — return last fallback result
        return last_result, attempts


__all__ = ["FallbackAttempt", "FallbackExecutor"]
