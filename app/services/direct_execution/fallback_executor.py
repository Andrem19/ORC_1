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
from app.execution_models import PlanSlice, WorkerAction
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.acceptance import (
    slice_requires_strict_acceptance,
    verdict_acceptance_blocker_reason,
    verdict_is_accepted,
)
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor
from app.services.direct_execution.fallback_failures import build_actionable_failure_checkpoint, is_provider_rate_limit
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts
from app.services.direct_execution.abort_validation import (
    abort_claims_empty_results,
    build_transcript_correction_prompt,
    transcript_has_successful_tool_data,
)
from app.services.direct_execution.guardrails import (
    REPAIRABLE_PROVIDER_NAMES,
    attempt_verdict_repair,
    build_contract_repair_prompt,
    checkpoint_complete_passes_gate,
    final_report_passes_quality_gate,
    normalize_incomplete_reason,
    should_attempt_contract_repair,
)
from app.services.direct_execution.guardrails import _RESEARCH_SETUP_TOOLS
from app.services.direct_execution.issue_classification import classify_issue_text

logger = logging.getLogger("orchestrator.direct.fallback")

_REPAIR_SKIP_INFRA_SIGNALS = frozenset({
    "qwen_mcp_tools_unavailable",
    "direct_empty_allowed_tools",
})
_REPAIR_ALLOWED_NEW_TOOL_CALL_PROVIDERS = frozenset({
    "glm_cli",
})
_SOFT_ABORT_REASON_CODES = frozenset({
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
})


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
        acceptance_verifier: Any | None = None,
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
        self.acceptance_verifier = acceptance_verifier
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
                catalog_snapshot=self.primary_executor.catalog_snapshot,
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
            catalog_snapshot=self.primary_executor.catalog_snapshot,
        )

    async def _is_success(
        self,
        result: DirectExecutionResult,
        *,
        slice_obj: PlanSlice,
        required_output_facts: list[str],
        inherited_facts: dict[str, Any],
    ) -> bool:
        action = result.action
        if action is None:
            return False
        if action.action_type == "abort":
            reason_code = str(action.reason_code or "").strip().lower()
            return bool(action.retryable) or reason_code in _SOFT_ABORT_REASON_CODES
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
                return False
            verdict_ok = verdict_is_accepted(slice_obj, action.verdict)
            if self.acceptance_verifier is not None:
                acceptance = await self.acceptance_verifier.verify(
                    plan=None,  # verifier does not depend on plan internals
                    slice_obj=slice_obj,
                    action=action,
                    transcript=list(result.transcript or []),
                    known_facts=inherited_facts,
                    required_output_facts=required_output_facts,
                )
                result.acceptance_result.update(acceptance.to_dict())
                if acceptance.passed:
                    if not verdict_ok:
                        logger.info(
                            "final_report acceptance proof PASSED for slice %s — "
                            "overriding non-accepted verdict %s",
                            slice_obj.slice_id,
                            action.verdict,
                        )
                    return True
                if verdict_ok:
                    logger.warning(
                        "final_report acceptance verifier FAILED for slice %s: %s",
                        slice_obj.slice_id,
                        ",".join(acceptance.blocking_reasons[:5]) or acceptance.status,
                    )
                    return False
            elif verdict_ok:
                return True
            logger.warning(
                "final_report acceptance gate FAILED for slice %s: %s "
                "(verdict=%s confidence=%.2f tool_calls=%d evidence_refs=%d)",
                slice_obj.slice_id,
                verdict_acceptance_blocker_reason(slice_obj, action.verdict) or "terminal_report_not_accepted",
                action.verdict,
                float(action.confidence or 0),
                result.tool_call_count,
                len(action.evidence_refs or []),
            )
            return False
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
            transcript=list(result.transcript or []),
            acceptance_result=dict(result.acceptance_result or {}),
        )

    @staticmethod
    def _maybe_apply_fact_contract_repair(
        *,
        result: DirectExecutionResult,
        slice_obj: PlanSlice,
        required_output_facts: list[str],
        inherited_facts: dict[str, Any],
    ) -> DirectExecutionResult:
        if result.action is None or result.action.action_type != "final_report":
            return result
        readiness = hydrate_final_report_facts(
            slice_obj=slice_obj,
            action=result.action,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
        )
        if readiness.missing_required_facts:
            return result
        if readiness.facts == dict(result.action.facts or {}) and readiness.evidence_refs == list(result.action.evidence_refs or []):
            return result
        result.action.facts = dict(readiness.facts)
        result.action.evidence_refs = list(readiness.evidence_refs)
        return result

    @staticmethod
    def _with_quality_gate_error(
        *,
        result: DirectExecutionResult,
        slice_obj: PlanSlice,
        required_output_facts: list[str],
        inherited_facts: dict[str, Any],
    ) -> DirectExecutionResult:
        if result.action is None or result.action.action_type != "final_report":
            return result
        passes, reason = final_report_passes_quality_gate(
            tool_call_count=result.tool_call_count,
            action=result.action,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=inherited_facts,
        )
        if passes or not reason:
            return result
        return DirectExecutionResult(
            action=result.action,
            artifact_path=result.artifact_path,
            raw_output=result.raw_output,
            error=reason,
            provider=result.provider,
            duration_ms=result.duration_ms,
            tool_call_count=result.tool_call_count,
            expensive_tool_call_count=result.expensive_tool_call_count,
            parse_retry_count=result.parse_retry_count,
            fallback_provider_index=result.fallback_provider_index,
            transcript=list(result.transcript or []),
            acceptance_result=dict(result.acceptance_result or {}),
        )

    @staticmethod
    def _with_acceptance_gate_error(
        *,
        result: DirectExecutionResult,
        slice_obj: PlanSlice,
    ) -> DirectExecutionResult:
        if result.action is None or result.action.action_type != "final_report":
            return result
        if verdict_is_accepted(slice_obj, result.action.verdict):
            return result
        blocker_reason = verdict_acceptance_blocker_reason(slice_obj, result.action.verdict)
        if not blocker_reason:
            return result
        return DirectExecutionResult(
            action=result.action,
            artifact_path=result.artifact_path,
            raw_output=result.raw_output,
            error="evidence_complete_but_verdict_not_accepted",
            provider=result.provider,
            duration_ms=result.duration_ms,
            tool_call_count=result.tool_call_count,
            expensive_tool_call_count=result.expensive_tool_call_count,
            parse_retry_count=result.parse_retry_count,
            fallback_provider_index=result.fallback_provider_index,
            transcript=list(result.transcript or []),
            acceptance_result=dict(result.acceptance_result or {}),
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
        if "zero_tool_calls" in (error or "").lower():
            section += (
                "## CRITICAL: Previous attempt returned ZERO tool calls.\n"
                "The model produced a final_report without calling any MCP tools.\n"
                "You MUST call at least one tool BEFORE returning any JSON result.\n"
                "Steps:\n"
                "1. Read the slice objective above.\n"
                "2. Pick the FIRST approved tool that is relevant.\n"
                "3. Call it with appropriate arguments.\n"
                "4. Base your final_report on the actual tool result.\n\n"
            )
        if "evidence_complete_but_verdict_not_accepted" in (error or "").lower():
            section += (
                "## CRITICAL: Previous attempt gathered enough evidence but returned a non-accepted verdict.\n"
                "For dependency-critical slices, use `COMPLETE` when the evidence fully proves every required fact.\n"
                "Do not default to `WATCHLIST` after the facts are already closed.\n\n"
            )
        section += (
            "Please attempt the same task using your available tools. "
            "Learn from the previous failure and avoid repeating the same mistakes.\n"
        )
        return section

    @staticmethod
    def _build_registry_prompt_section(*, registry: dict[str, Any]) -> str:
        mapping = registry.get("canonical_to_visible") if isinstance(registry.get("canonical_to_visible"), dict) else {}
        if not mapping:
            return ""
        lines = [
            "## Exact MCP tool names",
            "Qwen native preflight already confirmed these exact visible MCP tools.",
            "Use the exact visible name for tool calls. Do not claim dev_space1 tools are unavailable.",
        ]
        for canonical_name, visible_name in sorted(mapping.items()):
            lines.append(f"- `{canonical_name}` -> `{visible_name}`")
        return "\n".join(lines)

    @staticmethod
    def _build_prior_attempt_context(
        *,
        provider_name: str,
        result: DirectExecutionResult,
    ) -> str:
        lines = [
            "## Prior attempt context",
            f"Previous provider: {provider_name}",
            f"Previous failure: {normalize_incomplete_reason(result)[:800]}",
        ]
        raw_output = str(result.raw_output or "").strip()
        if raw_output:
            lines.extend(
                [
                    "Previous raw output excerpt:",
                    "```",
                    raw_output[:2000],
                    "```",
                ]
            )
        # Transcript-validated correction: when the model aborted claiming
        # empty results but the transcript shows successful tool data, inject
        # a strong correction so the retry model does not repeat the same
        # hallucination.
        action = result.action
        transcript = list(result.transcript or [])
        if action is not None and transcript:
            reason_code = str(getattr(action, "reason_code", "") or "")
            summary = str(getattr(action, "summary", "") or "")
            if abort_claims_empty_results(reason_code, summary, raw_output):
                if transcript_has_successful_tool_data(transcript):
                    correction = build_transcript_correction_prompt(transcript)
                    if correction.strip():
                        lines.append("")
                        lines.append(correction)
        return "\n".join(lines)

    def _record_qwen_namespace_incident(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        registry: dict[str, Any],
        raw_output: str,
    ) -> None:
        self.incident_store.record(
            summary="Qwen fallback claimed dev_space1 tools unavailable despite visible registry",
            metadata={
                "plan_id": plan_id,
                "slice_id": slice_obj.slice_id,
                "provider": "qwen_cli",
                "allowed_tools": list(slice_obj.allowed_tools or []),
                "visible_tools": list(registry.get("visible_tools") or []),
                "exact_visible_tools": list(registry.get("exact_visible_tools") or []),
                "canonical_to_visible": dict(registry.get("canonical_to_visible") or {}),
                "raw_output_excerpt": str(raw_output or "")[:1500],
            },
            source="direct_fallback",
            severity="medium",
        )

    def _record_provider_limit_incident(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        provider_name: str,
        failure_reason: str,
    ) -> None:
        self.incident_store.record(
            summary="Fallback chain hit provider rate limit",
            metadata={
                "plan_id": plan_id,
                "slice_id": slice_obj.slice_id,
                "provider": provider_name,
                "failure_reason": failure_reason[:500],
            },
            source="direct_fallback",
            severity="medium",
        )

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
        primary_result = self._maybe_apply_fact_contract_repair(
            result=primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )
        primary_result = self._with_quality_gate_error(
            result=primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        )
        primary_result = self._with_acceptance_gate_error(
            result=primary_result,
            slice_obj=slice_obj,
        )

        if await self._is_success(
            primary_result,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        ) or not self.fallback_providers:
            return primary_result, []

        primary_route = self._result_route(primary_result)
        if primary_route == "hard_block_infra":
            return primary_result, []

        primary_retry_budget = max(0, int(getattr(self.direct_config, "primary_retry_budget", 1) or 0))
        if primary_route == "fallback_allowed" and primary_retry_budget > 0 and not self._should_skip_repair(primary_result):
            primary_retry_prompt = self._build_prior_attempt_context(
                provider_name=str(primary_result.provider or getattr(self.direct_config, "provider", "") or "primary"),
                result=primary_result,
            )
            retry_result = await self.primary_executor.execute(
                plan_id=plan_id,
                slice_obj=slice_obj,
                baseline_bootstrap=baseline_bootstrap,
                known_facts=known_facts,
                required_output_facts=required_output_facts,
                recent_turn_summaries=recent_turn_summaries,
                checkpoint_summary=checkpoint_summary,
                extra_prompt_section=primary_retry_prompt,
                on_tool_progress=on_tool_progress,
                attempt_label="primary_retry",
            )
            retry_result = self._maybe_apply_verdict_repair(
                result=retry_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            retry_result = self._maybe_apply_fact_contract_repair(
                result=retry_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            retry_result = self._with_quality_gate_error(
                result=retry_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            retry_result = self._with_acceptance_gate_error(
                result=retry_result,
                slice_obj=slice_obj,
            )
            if await self._is_success(
                retry_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            ):
                return retry_result, []
            primary_result = retry_result
            primary_route = self._result_route(primary_result)
            if primary_route == "hard_block_infra":
                return primary_result, []

        # --- Fallback chain ---
        attempts: list[FallbackAttempt] = []
        last_result = primary_result
        last_error = normalize_incomplete_reason(primary_result)
        last_output = primary_result.raw_output
        remaining_repairs = max(0, int(getattr(self.direct_config, "parse_repair_attempts", 1) or 0))

        if primary_route == "repair_only":
            if not self._should_skip_repair(primary_result):
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
                primary_result = self._maybe_apply_fact_contract_repair(
                    result=primary_result,
                    slice_obj=slice_obj,
                    required_output_facts=required_output_facts,
                    inherited_facts=known_facts,
                )
                primary_result = self._with_quality_gate_error(
                    result=primary_result,
                    slice_obj=slice_obj,
                    required_output_facts=required_output_facts,
                    inherited_facts=known_facts,
                )
                primary_result = self._with_acceptance_gate_error(
                    result=primary_result,
                    slice_obj=slice_obj,
                )
                if await self._is_success(
                    primary_result,
                    slice_obj=slice_obj,
                    required_output_facts=required_output_facts,
                    inherited_facts=known_facts,
                ):
                    return primary_result, []
            return primary_result, []

        if not self._should_skip_repair(primary_result):
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
            primary_result = self._maybe_apply_fact_contract_repair(
                result=primary_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            primary_result = self._with_quality_gate_error(
                result=primary_result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            primary_result = self._with_acceptance_gate_error(
                result=primary_result,
                slice_obj=slice_obj,
            )
            if await self._is_success(
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

            # Pass allowed MCP tools to claude_cli so it can build --allowedTools flag
            if provider_name == "claude_cli":
                from app.runtime_factory import get_allowed_mcp_tools
                allowed_mcp_tools = get_allowed_mcp_tools(slice_obj.allowed_tools)
                if allowed_mcp_tools:
                    adapter_invoke_kwargs["allowed_mcp_tools"] = allowed_mcp_tools
                    logger.debug("Passing %d allowed MCP tools to claude_cli fallback", len(allowed_mcp_tools))
                else:
                    logger.warning(
                        "No allowed MCP tools for slice %s; claude_cli fallback will have zero tool access "
                        "(consider skipping this provider in config)",
                        slice_obj.slice_id,
                    )
            qwen_registry: dict[str, Any] = {}
            if provider_name == "qwen_cli" and isinstance(adapter, QwenWorkerCli) and bool(getattr(self.direct_config, "qwen_tool_registry_preflight", True)):
                probe_timeout = int(getattr(self.direct_config, "qwen_preflight_timeout_seconds", 60) or 60)
                try:
                    qwen_registry = adapter.preflight_tool_registry(
                        required_tools=slice_obj.allowed_tools,
                        timeout=probe_timeout,
                    )
                except TypeError:
                    # Test doubles and older adapters may not accept the timeout kwarg yet.
                    qwen_registry = adapter.preflight_tool_registry(required_tools=slice_obj.allowed_tools)
                registry_section = self._build_registry_prompt_section(registry=qwen_registry)
                if registry_section:
                    extra_sections.append(registry_section)
                if not qwen_registry["available"]:
                    # Do NOT disable tool use — let Qwen try with its own MCP config.
                    # Disabling tools guarantees zero_tool_calls → quality gate rejection.
                    missing = qwen_registry.get("missing_required_tools") or []
                    visible = qwen_registry.get("visible_tools") or []
                    logger.warning(
                        "Qwen tool registry preflight failed: %s (proceeding with tool_use=True) "
                        "missing_required=%s visible_count=%d",
                        qwen_registry.get("reason", "unknown"),
                        missing,
                        len(visible),
                    )
                    extra_sections.append(
                        f"## Note\nQwen tool registry preflight incomplete: {qwen_registry.get('reason', '')}. "
                        "Proceed using available MCP tools."
                    )
            approved = [str(item).strip() for item in (slice_obj.allowed_tools or []) if str(item).strip()]
            if approved:
                approved_section = (
                    "## Approved tools for this slice\n"
                    "You MUST use these MCP tools: "
                    + ", ".join(sorted(approved))
                    + "\nDo not invent tool names."
                )
                if qwen_registry.get("canonical_to_visible"):
                    approved_section += "\nIf Qwen preflight listed an exact visible name for a tool, use that exact visible name in the tool call."
                else:
                    approved_section += "\nIf a tool is not visible in your MCP registry, proceed with the closest available research tool and record the gap in evidence."
                extra_sections.append(approved_section)

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
                    transcript=[],
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
                transcript=list(result.transcript or []),
                acceptance_result=dict(result.acceptance_result or {}),
            )
            result = self._maybe_apply_fact_contract_repair(
                result=result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            result = self._with_quality_gate_error(
                result=result,
                slice_obj=slice_obj,
                required_output_facts=required_output_facts,
                inherited_facts=known_facts,
            )
            result = self._with_acceptance_gate_error(
                result=result,
                slice_obj=slice_obj,
            )
            result_route = self._result_route(result)

            failure_reason = normalize_incomplete_reason(result)
            if (
                provider_name == "qwen_cli"
                and "dev_space1_tools_unavailable" in failure_reason.lower()
                and qwen_registry.get("canonical_to_visible")
            ):
                result_route = "repair_only"
                self._record_qwen_namespace_incident(
                    plan_id=plan_id,
                    slice_obj=slice_obj,
                    registry=qwen_registry,
                    raw_output=result.raw_output,
                )
            if result_route == "hard_block_infra":
                artifact_path = self.artifact_store.save_direct_fallback_attempt(
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    payload={
                        "attempt_id": f"fallback_summary_{idx}_{slice_obj.slice_id}_{slice_obj.turn_count + 1}",
                        "provider": provider_name,
                        "status": "failed",
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
                return result, attempts
            if not self._should_skip_repair(result):
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
                    repair_context_section=self._build_registry_prompt_section(registry=qwen_registry),
                    registry_verified=bool(qwen_registry.get("canonical_to_visible")),
                )
                result = self._maybe_apply_fact_contract_repair(
                    result=result,
                    slice_obj=slice_obj,
                    required_output_facts=required_output_facts,
                    inherited_facts=known_facts,
                )
                result = self._with_quality_gate_error(
                    result=result,
                    slice_obj=slice_obj,
                    required_output_facts=required_output_facts,
                    inherited_facts=known_facts,
                )
                result = self._with_acceptance_gate_error(
                    result=result,
                    slice_obj=slice_obj,
                )
                failure_reason = normalize_incomplete_reason(result)
                result_route = self._result_route(result)

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

            if await self._is_success(
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
            if result_route in {"repair_only", "hard_block_infra"}:
                return result, attempts

            last_result = result
            last_error = failure_reason
            last_output = result.raw_output

            logger.warning(
                "Fallback #%d (%s) also failed for slice %s: %s",
                idx, provider_name, slice_obj.slice_id, failure_reason[:200],
            )
            if result.tool_call_count == 0:
                logger.warning(
                    "Fallback #%d (%s) produced zero tool calls for slice %s (verdict=%s)",
                    idx, provider_name, slice_obj.slice_id,
                    getattr(result.action, "verdict", "?") if result.action else "?",
                )
            if is_provider_rate_limit(failure_reason):
                self._record_provider_limit_incident(
                    plan_id=plan_id,
                    slice_obj=slice_obj,
                    provider_name=provider_name,
                    failure_reason=failure_reason,
                )

        # All providers failed — return the last result, annotated with the
        # most actionable failed attempt when the tail failure is only provider
        # infrastructure (for example a rate limit).
        return build_actionable_failure_checkpoint(
            last_result=last_result,
            attempts=attempts,
            slice_obj=slice_obj,
            required_output_facts=required_output_facts,
            inherited_facts=known_facts,
        ), attempts

    def _should_skip_repair(self, result: DirectExecutionResult) -> bool:
        if not bool(getattr(self.direct_config, "fallback_skip_repair_on_infra_signal", True)):
            return False
        return normalize_incomplete_reason(result) in _REPAIR_SKIP_INFRA_SIGNALS

    @staticmethod
    def _result_route(result: DirectExecutionResult) -> str:
        acceptance = result.acceptance_result if isinstance(result.acceptance_result, dict) else {}
        route = str(acceptance.get("route") or "").strip()
        if route in {"repair_only", "hard_block_infra", "fallback_allowed"}:
            return route
        action = result.action
        classification = classify_issue_text(
            result.error,
            result.raw_output,
            normalize_incomplete_reason(result),
            getattr(action, "reason_code", ""),
            getattr(action, "summary", ""),
        )
        if classification == "contract_misuse":
            return "repair_only"
        if classification == "infra_unavailable":
            return "hard_block_infra"
        return "fallback_allowed"

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
        repair_context_section: str = "",
        registry_verified: bool = False,
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
        is_missing_facts = "missing_required_facts" in failure_reason.lower()
        strict_acceptance_required = (
            (
                "evidence_complete_but_verdict_not_accepted" in failure_reason.lower()
                or "auto_salvage_stub_rejected" in failure_reason.lower()
            )
            and slice_requires_strict_acceptance(slice_obj)
        )
        is_research_setup = _RESEARCH_SETUP_TOOLS.issubset(set(allowed_tools))
        forbid_tool_calls = (
            "tool_not_in_allowlist" in failure_reason.lower()
            or (is_missing_facts and not is_research_setup)
            or bool(prior_result.action and prior_result.action.facts.get("direct.invalid_terminal_tool_call"))
            or strict_acceptance_required
        )
        allow_new_tool_calls = (
            provider_name in _REPAIR_ALLOWED_NEW_TOOL_CALL_PROVIDERS
            and not forbid_tool_calls
            and not is_missing_facts
        ) or (
            is_research_setup
            and is_missing_facts
            and provider_name in REPAIRABLE_PROVIDER_NAMES
        )
        repair_tool_budget = max(0, int(getattr(self.direct_config, "repair_tool_call_budget", 3) or 0))
        repair_prompt = build_contract_repair_prompt(
            provider_name=provider_name,
            allowed_tools=allowed_tools,
            failure_reason=failure_reason,
            raw_output_excerpt=prior_result.raw_output[:2000],
            required_output_facts=required_output_facts,
            forbid_tool_calls=forbid_tool_calls,
            registry_verified=registry_verified,
            allow_new_tool_calls=allow_new_tool_calls,
            repair_tool_call_budget=repair_tool_budget if allow_new_tool_calls else 0,
            is_research_setup_repair=is_research_setup and is_missing_facts,
            strict_acceptance_required=strict_acceptance_required,
        )
        if repair_context_section.strip():
            repair_prompt = repair_context_section.strip() + "\n\n" + repair_prompt
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
        repaired_action = repaired.action
        repaired_tool_call_count = int(repaired.tool_call_count or 0)
        repaired_expensive_count = int(repaired.expensive_tool_call_count or 0)
        repaired_transcript = list(repaired.transcript or [])

        # Repair prompts often forbid new tool calls and only request a fixed
        # terminal JSON from already collected evidence. Preserve prior proven
        # tool telemetry so quality-gate checks evaluate real work instead of
        # failing with zero_tool_calls on a rewrite-only repair response.
        if (
            repaired_action is not None
            and repaired_action.action_type == "final_report"
            and repaired_tool_call_count <= 0
            and int(prior_result.tool_call_count or 0) > 0
        ):
            repaired_tool_call_count = int(prior_result.tool_call_count or 0)
            if repaired_expensive_count <= 0:
                repaired_expensive_count = int(prior_result.expensive_tool_call_count or 0)
            if not repaired_transcript and prior_result.transcript:
                repaired_transcript = list(prior_result.transcript or [])
            prior_action = prior_result.action
            if (
                not list(repaired_action.evidence_refs or [])
                and prior_action is not None
                and list(prior_action.evidence_refs or [])
            ):
                repaired_action.evidence_refs = list(prior_action.evidence_refs or [])
            repaired_action.facts.setdefault("direct.repair_reused_prior_tool_trace", True)
            repaired_action.facts.setdefault("direct.repair_reused_prior_tool_count", repaired_tool_call_count)

        return DirectExecutionResult(
            action=repaired_action,
            artifact_path=repaired.artifact_path,
            raw_output=repaired.raw_output,
            error=normalize_incomplete_reason(repaired),
            provider=provider_name,
            duration_ms=repaired.duration_ms,
            tool_call_count=repaired_tool_call_count,
            expensive_tool_call_count=repaired_expensive_count,
            parse_retry_count=repaired.parse_retry_count + 1,
            fallback_provider_index=repaired.fallback_provider_index or prior_result.fallback_provider_index,
            transcript=repaired_transcript,
            acceptance_result=dict(repaired.acceptance_result or {}),
        )

__all__ = ["FallbackAttempt", "FallbackExecutor"]
