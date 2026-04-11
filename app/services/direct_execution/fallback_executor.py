"""
Fallback executor that chains primary → fallback_1 → fallback_2 providers
when a direct slice fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from app.adapters.base import BaseAdapter
from app.adapters.qwen_worker_cli import QwenWorkerCli
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor
from app.services.direct_execution.guardrails import (
    attempt_verdict_repair,
    build_contract_repair_prompt,
    checkpoint_complete_passes_gate,
    final_report_passes_quality_gate,
    normalize_incomplete_reason,
    should_attempt_contract_repair,
)

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

    def _make_fallback_executor(self, adapter: BaseAdapter, *, provider_name: str) -> DirectSliceExecutor:
        if self.invoker is not None:
            return DirectSliceExecutor(
                adapter=adapter,
                artifact_store=self.artifact_store,
                incident_store=self.incident_store,
                direct_config=self.direct_config,
                worker_system_prompt=self.worker_system_prompt,
                invoker=self.invoker,
                provider_name=provider_name,
            )
        from app.services.direct_execution.invocation import invoke_adapter_with_retries

        return DirectSliceExecutor(
            adapter=adapter,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            direct_config=self.direct_config,
            worker_system_prompt=self.worker_system_prompt,
            invoker=invoke_adapter_with_retries,
            provider_name=provider_name,
        )

    @staticmethod
    def _is_success(
        result: DirectExecutionResult,
        *,
        slice_obj: PlanSlice,
        required_output_facts: list[str],
        inherited_facts: dict[str, Any],
    ) -> bool:
        action = result.action
        if action is None:
            return False
        if action.action_type == "final_report":
            passes, reason = final_report_passes_quality_gate(
                tool_call_count=result.tool_call_count,
                action=action,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=inherited_facts,
            )
            if not passes:
                logger.warning(
                    "final_report quality gate FAILED for slice %s: %s "
                    "(verdict=%s confidence=%.2f tool_calls=%d evidence_refs=%d)",
                    slice_obj.slice_id,
                    reason,
                    action.verdict,
                    float(action.confidence or 0),
                    result.tool_call_count,
                    len(action.evidence_refs or []),
                )
            return passes
        return checkpoint_complete_passes_gate(
            slice_obj=slice_obj,
            action=action,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
        )

    @staticmethod
    def _maybe_apply_verdict_repair(
        *,
        result: DirectExecutionResult,
        slice_obj: PlanSlice,
        required_output_facts: list[str],
        inherited_facts: dict[str, Any],
    ) -> DirectExecutionResult:
        if result.action is None or result.action.action_type != "final_report":
            return result
        repaired = attempt_verdict_repair(
            action=result.action,
            tool_call_count=result.tool_call_count,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
            provider=result.provider or "",
        )
        if repaired is None:
            return result
        logger.info(
            "Verdict repaired for slice %s: %s -> WATCHLIST (confidence=%.2f, tool_calls=%d, evidence_refs=%d)",
            slice_obj.slice_id,
            result.action.verdict,
            float(result.action.confidence or 0),
            result.tool_call_count,
            len(result.action.evidence_refs or []),
        )
        return DirectExecutionResult(
            action=repaired,
            artifact_path=result.artifact_path,
            raw_output=result.raw_output,
            error=result.error,
            provider=result.provider,
            duration_ms=result.duration_ms,
            tool_call_count=result.tool_call_count,
            expensive_tool_call_count=result.expensive_tool_call_count,
            parse_retry_count=result.parse_retry_count,
            fallback_provider_index=result.fallback_provider_index,
        )

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

        # Verdict repair: if the model said INCOMPLETE/PARTIAL but all evidence
        # gates actually pass, fix the verdict to avoid a needless fallback.
        primary_result = self._maybe_apply_verdict_repair(
            result=primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )

        if self._is_success(
            primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        ) or not self.fallback_providers:
            return primary_result, []

        # --- Fallback chain ---
        attempts: list[FallbackAttempt] = []
        last_result = primary_result
        last_error = normalize_incomplete_reason(primary_result)
        last_output = primary_result.raw_output
        remaining_repairs = max(0, int(getattr(self.direct_config, "parse_repair_attempts", 1) or 0))

        primary_result = await self._maybe_repair_result(
            executor=self.primary_executor,
            provider_name=str(primary_result.provider or getattr(self.direct_config, "provider", "") or "primary"),
            prior_result=primary_result,
            slice_obj=slice_obj,
            plan_id=plan_id,
            baseline_bootstrap=baseline_bootstrap,
            known_facts=known_facts,
            required_output_facts=required_output_facts,
            recent_turn_summaries=recent_turn_summaries,
            checkpoint_summary=checkpoint_summary,
            attempts_remaining=remaining_repairs,
            attempt_label="repair_primary",
            on_tool_progress=on_tool_progress,
        )
        if self._is_success(
            primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        ):
            return primary_result, []
        last_result = primary_result
        last_error = normalize_incomplete_reason(primary_result)
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
                    on_provider_switch(provider_name, idx)
                except TypeError:
                    try:
                        on_provider_switch(provider_name)
                    except Exception:
                        pass
                except Exception:
                    pass

            extra_section = self._build_fallback_prompt_section(
                failed_provider=last_result.provider or "primary",
                error=last_error,
                raw_output_excerpt=last_output[:2000],
                attempt_index=idx - 1,
            )

            try:
                executor = self._make_fallback_executor(adapter, provider_name=provider_name)
            except TypeError:
                executor = self._make_fallback_executor(adapter)  # test doubles may not accept provider_name
            adapter_invoke_kwargs: dict[str, Any] = {}
            extra_sections: list[str] = [extra_section]
            if provider_name == "qwen_cli" and isinstance(adapter, QwenWorkerCli) and bool(getattr(self.direct_config, "qwen_tool_registry_preflight", True)):
                registry = adapter.preflight_tool_registry(required_tools=slice_obj.allowed_tools)
                if not registry["available"]:
                    # Do NOT disable tool use — let Qwen try with its own MCP config.
                    # Disabling tools guarantees zero_tool_calls → quality gate rejection.
                    logger.warning(
                        "Qwen tool registry preflight failed: %s (proceeding with tool_use=True)",
                        registry.get("reason", "unknown"),
                    )
                    extra_sections.append(
                        f"## Note\nQwen tool registry preflight incomplete: {registry.get('reason', '')}. "
                        "Proceed using available MCP tools."
                    )

            try:
                result = await executor.execute(
                    plan_id=plan_id,
                    slice_obj=slice_obj,
                    baseline_bootstrap=baseline_bootstrap,
                    known_facts=known_facts,
                    required_output_facts=required_output_facts,
                    recent_turn_summaries=recent_turn_summaries,
                    checkpoint_summary=checkpoint_summary,
                    extra_prompt_section="\n\n".join(section for section in extra_sections if section.strip()),
                    on_tool_progress=on_tool_progress,
                    attempt_label=f"fallback_{idx}",
                    adapter_invoke_kwargs=adapter_invoke_kwargs,
                )
            except Exception as exc:
                logger.warning("Fallback executor exception for %r: %s", provider_name, exc)
                result = DirectExecutionResult(
                    action=None,
                    artifact_path="",
                    raw_output="",
                    error=str(exc) or f"{provider_name}_fallback_exception",
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

            failure_reason = normalize_incomplete_reason(result)
            result = await self._maybe_repair_result(
                executor=executor,
                provider_name=provider_name,
                prior_result=result,
                slice_obj=slice_obj,
                plan_id=plan_id,
                baseline_bootstrap=baseline_bootstrap,
                known_facts=known_facts,
                required_output_facts=required_output_facts,
                recent_turn_summaries=recent_turn_summaries,
                checkpoint_summary=checkpoint_summary,
                attempts_remaining=remaining_repairs,
                attempt_label=f"repair_fallback_{idx}",
                on_tool_progress=on_tool_progress,
                adapter_invoke_kwargs=adapter_invoke_kwargs,
            )
            failure_reason = normalize_incomplete_reason(result)

            artifact_path = self.artifact_store.save_direct_fallback_attempt(
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                payload={
                    "attempt_id": f"fallback_summary_{idx}_{slice_obj.slice_id}_{slice_obj.turn_count + 1}",
                    "provider": provider_name,
                    "status": "completed" if result.action is not None else "failed",
                    "error": failure_reason,
                    "parse_error": result.error if "direct_output_parse_failed" in str(result.error or "") else "",
                    "raw_output_length": len(result.raw_output),
                    "raw_output_excerpt": result.raw_output[:2000],
                    "duration_ms": result.duration_ms,
                    "tool_call_count": result.tool_call_count,
                    "fallback_index": idx,
                    "adapter_name": adapter.name(),
                    "terminal_action_type": result.action.action_type if result.action is not None else "",
                    "terminal_action_status": result.action.status if result.action is not None else "",
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

            if self._is_success(
                result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            ):
                logger.info(
                    "Fallback #%d (%s) succeeded for slice %s",
                    idx, provider_name, slice_obj.slice_id,
                )
                return result, attempts

            last_result = result
            last_error = failure_reason
            last_output = result.raw_output

            logger.warning(
                "Fallback #%d (%s) also failed for slice %s: %s",
                idx, provider_name, slice_obj.slice_id, failure_reason[:200],
            )

        # All providers failed — return last fallback result
        return last_result, attempts

    async def _maybe_repair_result(
        self,
        *,
        executor: DirectSliceExecutor,
        provider_name: str,
        prior_result: DirectExecutionResult,
        slice_obj: PlanSlice,
        plan_id: str,
        baseline_bootstrap: dict[str, Any],
        known_facts: dict[str, Any],
        required_output_facts: list[str],
        recent_turn_summaries: list[str],
        checkpoint_summary: str,
        attempts_remaining: int,
        attempt_label: str,
        on_tool_progress: Any = None,
        adapter_invoke_kwargs: dict[str, Any] | None = None,
    ) -> DirectExecutionResult:
        if not bool(getattr(self.direct_config, "contract_guardrails_enabled", True)):
            return prior_result
        if not should_attempt_contract_repair(
            provider_name=provider_name,
            result=prior_result,
            attempts_remaining=attempts_remaining,
        ):
            return prior_result
        allowed_tools = sorted({str(item).strip() for item in slice_obj.allowed_tools if str(item).strip()})
        failure_reason = normalize_incomplete_reason(prior_result)
        forbid_tool_calls = (
            "tool_not_in_allowlist" in failure_reason.lower()
            or bool(prior_result.action and prior_result.action.facts.get("direct.invalid_terminal_tool_call"))
        )
        repair_prompt = build_contract_repair_prompt(
            provider_name=provider_name,
            allowed_tools=allowed_tools,
            failure_reason=failure_reason,
            raw_output_excerpt=prior_result.raw_output[:2000],
            required_output_facts=required_output_facts,
            forbid_tool_calls=forbid_tool_calls,
        )
        repaired = await executor.execute(
            plan_id=plan_id,
            slice_obj=slice_obj,
            baseline_bootstrap=baseline_bootstrap,
            known_facts=known_facts,
            required_output_facts=required_output_facts,
            recent_turn_summaries=recent_turn_summaries,
            checkpoint_summary=checkpoint_summary,
            extra_prompt_section=repair_prompt,
            on_tool_progress=on_tool_progress,
            attempt_label=attempt_label,
            adapter_invoke_kwargs=adapter_invoke_kwargs,
        )
        return DirectExecutionResult(
            action=repaired.action,
            artifact_path=repaired.artifact_path,
            raw_output=repaired.raw_output,
            error=normalize_incomplete_reason(repaired),
            provider=provider_name,
            duration_ms=repaired.duration_ms,
            tool_call_count=repaired.tool_call_count,
            expensive_tool_call_count=repaired.expensive_tool_call_count,
            parse_retry_count=repaired.parse_retry_count + 1,
            fallback_provider_index=repaired.fallback_provider_index or prior_result.fallback_provider_index,
        )


__all__ = ["FallbackAttempt", "FallbackExecutor"]
