"""
LM Studio direct OpenAI-tool loop backed by dev_space1 MCP calls.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.adapters.base import AdapterResponse
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.backtests_protocol import (
    augment_allowed_tools_for_backtests,
    build_backtests_first_action_guide,
    build_backtests_zero_tool_nudge,
    is_backtests_context,
)
from app.services.direct_execution.feature_contract_runtime import (
    build_feature_contract_construction_final_report,
    build_feature_contract_exploration_final_report,
    build_feature_contract_identifier_report,
    build_feature_profitability_filter_final_report,
    feature_contract_exploration_missing_tools,
    feature_contract_exploration_next_call,
    repair_feature_analytics_identifier_from_transcript,
)
from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool
from app.services.direct_execution.lmstudio_finalization import (
    build_generic_transcript_salvage_report,
    build_profile_final_report,
)
from app.services.direct_execution.issue_classification import classify_issue_payload
from app.services.direct_execution.mcp_client import DirectMcpClient, _to_jsonable
from app.services.direct_execution.runtime_profiles import resolve_runtime_slice_metadata_with_prerequisites
from app.services.direct_execution.research_handles import (
    ResearchHandleState,
    repair_handle_arguments,
    seed_handle_state_from_facts,
    update_handle_state,
)
from app.services.direct_execution.semantic_progress import (
    build_auto_final_report,
    build_semantic_loop_abort,
    build_watchdog_checkpoint,
    compact_tool_result_message,
    derive_research_write_facts,
    should_auto_finalize_research_slice,
    tool_call_signature,
)
from app.services.direct_execution.lmstudio_connection import LMStudioConnectionPool, _is_cloud_provider
from app.services.direct_execution.temperature_config import get_adaptive_temperature
from app.services.direct_execution.no_op_coercion import coerce_no_op_terminal_write
from app.services.direct_execution.tool_preflight import preflight_direct_tool_call
from app.services.direct_execution.transcript_facts import derive_facts_from_transcript
from app.services.mcp_catalog.classifier import is_expensive_tool, is_expensive_tool_call
from app.services.mcp_catalog.models import McpCatalogSnapshot

logger = logging.getLogger("orchestrator.direct.lmstudio")

_STRING_JSON_FIELDS = frozenset({"record", "spec", "payload"})


def _coerce_nested_json_strings(value: Any, depth: int = 0) -> tuple[Any, bool]:
    """Recursively coerce string values that look like JSON or Python literals.

    MiniMax often emits nested values like ``metadata.shortlist_families`` as
    Python repr strings (``"['a', 'b']"``) instead of actual lists.  This
    function walks dicts and lists, attempting to parse any string that starts
    with ``[`` or ``{`` first as JSON, then as a Python literal.
    """
    if depth > 10:
        return value, False
    if isinstance(value, dict):
        changed = False
        result: dict[str, Any] = {}
        for k, v in value.items():
            coerced, did_change = _coerce_nested_json_strings(v, depth + 1)
            result[k] = coerced
            if did_change:
                changed = True
        return result, changed
    if isinstance(value, list):
        changed = False
        items: list[Any] = []
        for item in value:
            coerced, did_change = _coerce_nested_json_strings(item, depth + 1)
            items.append(coerced)
            if did_change:
                changed = True
        return items, changed
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped[0] in "[{":
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    result, _ = _coerce_nested_json_strings(parsed, depth + 1)
                    return result, True
            except (json.JSONDecodeError, ValueError):
                pass
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, (dict, list)):
                    result, _ = _coerce_nested_json_strings(parsed, depth + 1)
                    return result, True
            except (ValueError, SyntaxError):
                pass
    return value, False


def _coerce_string_json_fields(arguments: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce known nested fields from JSON string to dict/list.

    Some models (e.g. MiniMax) serialize nested objects like ``record`` as JSON
    strings instead of actual objects.  The MCP server rejects string-encoded
    payloads (``"record must be valid JSON when provided as a string"``), so we
    detect and deserialize them before the call is dispatched.

    After top-level coercion, recursively walks the resulting dicts/lists and
    coerces any nested string values that look like JSON or Python literals
    (e.g. ``"['a', 'b']"`` → ``['a', 'b']``).
    """
    notes: list[str] = []
    for field_name in _STRING_JSON_FIELDS:
        value = arguments.get(field_name)
        if not isinstance(value, str) or not value.strip():
            continue
        stripped = value.strip()
        if not stripped.startswith(("{", "[")):
            continue
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, (dict, list)):
                arguments[field_name] = parsed
                notes.append(f"coerced {field_name} from JSON string to {type(parsed).__name__}")
        except (json.JSONDecodeError, ValueError):
            pass
    # Recursively coerce nested string values that look like JSON/Python literals
    for field_name in _STRING_JSON_FIELDS:
        value = arguments.get(field_name)
        if isinstance(value, (dict, list)):
            coerced, changed = _coerce_nested_json_strings(value)
            if changed:
                arguments[field_name] = coerced
                notes.append(f"coerced nested JSON strings in {field_name}")
    return arguments, notes


@dataclass
class LmStudioToolLoopResult:
    response: AdapterResponse
    transcript: list[dict[str, Any]] = field(default_factory=list)
    tool_call_count: int = 0
    expensive_tool_call_count: int = 0


class LmStudioToolLoop:
    def __init__(
        self,
        *,
        adapter: LmStudioWorkerApi,
        mcp_client: DirectMcpClient,
        incident_store: LocalIncidentStore,
        catalog_snapshot: McpCatalogSnapshot | None = None,
        allowed_tools: set[str],
        slice_title: str,
        slice_objective: str = "",
        success_criteria: list[str],
        policy_tags: list[str] | None = None,
        required_prerequisite_facts: list[str] | None = None,
        required_output_facts: list[str],
        runtime_profile: str = "",
        finalization_mode: str = "",
        max_tool_calls: int,
        max_expensive_tool_calls: int,
        safe_exclude_tools: set[str],
        first_action_timeout_seconds: int,
        stalled_action_timeout_seconds: int,
        zero_tool_retries: int = 2,
        first_turn_tool_choice: str = "required",
        baseline_bootstrap: dict[str, Any] | None = None,
        known_facts: dict[str, Any] | None = None,
    ) -> None:
        self.adapter = adapter
        self.mcp_client = mcp_client
        self.incident_store = incident_store
        self.catalog_snapshot = catalog_snapshot
        self.slice_title = slice_title
        self.slice_objective = str(slice_objective or "")
        self.success_criteria = success_criteria
        self.policy_tags = [str(item).strip() for item in list(policy_tags or []) if str(item).strip()]
        allowed_tools = augment_allowed_tools_for_backtests(
            allowed_tools={str(item).strip() for item in allowed_tools if str(item).strip()},
            catalog_snapshot=catalog_snapshot,
            runtime_profile=runtime_profile,
            title=self.slice_title,
            objective=self.slice_objective,
            success_criteria=list(success_criteria or []),
            policy_tags=self.policy_tags,
        )
        self.allowed_tools = allowed_tools
        (
            resolved_profile,
            resolved_required_facts,
            resolved_prerequisites,
            resolved_finalization_mode,
        ) = resolve_runtime_slice_metadata_with_prerequisites(
            runtime_profile=runtime_profile,
            required_output_facts=list(required_output_facts),
            required_prerequisite_facts=list(required_prerequisite_facts or []),
            finalization_mode=finalization_mode,
            allowed_tools=sorted(allowed_tools),
            catalog_snapshot=catalog_snapshot,
            title=self.slice_title,
            objective=self.slice_objective,
            success_criteria=list(success_criteria or []),
            policy_tags=self.policy_tags,
        )
        self.required_output_facts = list(resolved_required_facts)
        self.required_prerequisite_facts = list(resolved_prerequisites)
        self.runtime_profile = str(resolved_profile or "").strip()
        self.finalization_mode = str(resolved_finalization_mode or "").strip()
        self.max_tool_calls = max(1, int(max_tool_calls or 1))
        self.max_expensive_tool_calls = max(0, int(max_expensive_tool_calls or 0))
        self.safe_exclude_tools = safe_exclude_tools
        self.first_action_timeout_seconds = max(5, int(first_action_timeout_seconds or 5))
        self.stalled_action_timeout_seconds = max(5, int(stalled_action_timeout_seconds or 5))

        # Tool-call reliability: nudge retries and first-turn tool_choice
        self._zero_tool_retries = max(0, int(zero_tool_retries))
        self._first_turn_tool_choice = str(first_turn_tool_choice or "auto").strip()
        self._baseline_bootstrap = dict(baseline_bootstrap or {})
        self._known_facts = dict(known_facts or {})

        # Connection pool for LM Studio API with retry logic
        self._connection_pool = LMStudioConnectionPool(
            base_url=self.adapter.base_url,
            api_key=self.adapter.api_key or "",
            model=self.adapter.model or "",
            timeout=max(self.first_action_timeout_seconds, self.stalled_action_timeout_seconds),
        )

        # Counter for consecutive checkpoints to prevent infinite loops
        self._consecutive_checkpoint_count = 0
        self._max_consecutive_checkpoints = 3
        self._exploration_streak = 0
        self._shortlist_nudge_sent = False
        self._research_setup_list_streak = 0
        self._research_setup_project_nudge_sent = False
        self._mixed_domain_read_streak = 0
        self._mixed_domain_nudge_sent = False
        self._mixed_domain_finalize_nudge_sent = False
        self._pending_system_nudge = ""

        # Adaptive temperature for weak providers (cloud providers are not weak)
        _temp_provider = "minimax" if _is_cloud_provider(self.adapter.base_url) else "lmstudio"
        self._get_adaptive_temp = lambda temp: get_adaptive_temperature(_temp_provider, temp)

        # Cloud providers need longer warm-up timeout (network latency)
        self._is_cloud = _is_cloud_provider(self.adapter.base_url)
        self._warmup_timeout = 15 if self._is_cloud else 5

    async def invoke(
        self,
        *,
        prompt: str,
        timeout_seconds: int,
        plan_id: str,
        slice_id: str,
        on_progress: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> LmStudioToolLoopResult:
        started = time.monotonic()
        try:
            # Pre-flight: verify LM Studio is reachable before investing in
            # schema fetching and prompt construction.
            warm_ok = await asyncio.wait_for(
                asyncio.to_thread(self._connection_pool.warm_up, self._warmup_timeout),
                timeout=self._warmup_timeout + 3,
            )
            if on_progress is not None:
                on_progress("preflight", {"healthy": warm_ok})
            if not warm_ok:
                logger.warning("Provider pre-flight health check failed — aborting slice early")
                return LmStudioToolLoopResult(
                    response=AdapterResponse(
                        success=True,
                        raw_output=build_watchdog_checkpoint(
                            summary=f"Provider API not reachable for slice '{self.slice_title or slice_id}'.",
                            reason_code="direct_lmstudio_preflight_failed",
                        ),
                        duration_seconds=time.monotonic() - started,
                    ),
                    transcript=[],
                    tool_call_count=0,
                    expensive_tool_call_count=0,
                )
            tools = self._openai_tool_schemas()
            if on_progress is not None:
                on_progress("tool_schema_ready", {"tool_count": len(tools)})
                on_progress(
                    "schema_fetch",
                    {"elapsed_ms": int((time.monotonic() - started) * 1000), "tool_count": len(tools), "fallback": False, "source": "startup_snapshot"},
                )
            messages: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are a direct execution agent. "
                        + self._build_first_action_guide()
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            transcript: list[dict[str, Any]] = []
            tool_call_count = 0
            tool_attempt_count = 0
            expensive_count = 0
            last_signature = ""
            repeat_count = 0
            had_contract_issue = False
            contract_error_streak = 0
            handle_state = ResearchHandleState()
            seed_handle_state_from_facts(handle_state, self._known_facts)
            zero_tool_nudge_count = 0
            _required_fallback_to_auto = False
            deadline = time.monotonic() + max(1, int(timeout_seconds or 1))
            timing_marks: dict[str, int] = {"schema_fetch": int((time.monotonic() - started) * 1000)}
            while time.monotonic() < deadline:
                idle_budget = self.first_action_timeout_seconds if tool_call_count == 0 else self.stalled_action_timeout_seconds
                # Use "required" on the first turn to force tool use
                if tool_attempt_count == 0 and not _required_fallback_to_auto:
                    tc = self._first_turn_tool_choice
                else:
                    tc = "auto"
                response = await self._chat(messages=messages, tools=tools, timeout=max(1, min(int(deadline - time.monotonic()), idle_budget)), tool_choice=tc)
                # If "required" caused an API error, retry with "auto"
                if tc == "required" and response.get("error") and tool_call_count == 0:
                    err_text = str(response.get("error", "")).lower()
                    if "tool_choice" in err_text or "required" in err_text or "not supported" in err_text:
                        logger.warning("LM Studio does not support tool_choice=required, falling back to auto")
                        _required_fallback_to_auto = True
                        response = await self._chat(messages=messages, tools=tools, timeout=max(1, min(int(deadline - time.monotonic()), idle_budget)), tool_choice="auto")
                transcript.append({"kind": "assistant_response", "payload": response})
                if "first_model_token" not in timing_marks:
                    timing_marks["first_model_token"] = int((time.monotonic() - started) * 1000)
                    if on_progress is not None:
                        on_progress("first_model_token", {"elapsed_ms": timing_marks["first_model_token"]})
                if on_progress is not None:
                    on_progress("assistant_response", {"transcript_len": len(transcript)})
                if response.get("error"):
                    salvage_raw = self._maybe_auto_finalize_after_stall(transcript)
                    if salvage_raw is not None:
                        # Salvage payloads are transcript summaries, not real completions.
                        # Force tool_call_count=0 so the quality gate rejects them and the
                        # orchestrator routes the slice to the fallback chain.
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=salvage_raw,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=0,
                            expensive_tool_call_count=expensive_count,
                        )
                    reason_code = "direct_model_stalled_before_first_action" if tool_call_count == 0 else "direct_model_stalled_between_actions"
                    summary = (
                        f"Direct model stalled before producing the first action for slice '{self.slice_title or slice_id}'."
                        if tool_call_count == 0
                        else f"Direct model stalled before progressing slice '{self.slice_title or slice_id}' after prior actions."
                    )
                    return LmStudioToolLoopResult(
                        response=AdapterResponse(
                            success=True,
                            raw_output=build_watchdog_checkpoint(summary=summary, reason_code=reason_code),
                            duration_seconds=time.monotonic() - started,
                            metadata={"timings_ms": timing_marks},
                        ),
                        transcript=transcript,
                        tool_call_count=tool_call_count,
                        expensive_tool_call_count=expensive_count,
                    )
                message = ((response.get("choices") or [{}])[0].get("message") or {})
                tool_calls = message.get("tool_calls") or []
                content = str(message.get("content") or "")
                if not tool_calls:
                    # Nudge retry: give the model another chance to call tools.
                    # Only nudge if the model has NEVER called a tool in this session.
                    if tool_attempt_count == 0 and zero_tool_nudge_count < self._zero_tool_retries and time.monotonic() < deadline:
                        zero_tool_nudge_count += 1
                        nudge_msg = self._build_zero_tool_nudge_message()
                        logger.info(
                            "Zero tool calls nudge retry %d/%d for slice '%s'",
                            zero_tool_nudge_count, self._zero_tool_retries,
                            self.slice_title or slice_id,
                        )
                        transcript.append({
                            "kind": "zero_tool_nudge",
                            "attempt": zero_tool_nudge_count,
                            "original_content": content[:500],
                        })
                        messages.append({"role": "assistant", "content": content})
                        messages.append({"role": "user", "content": nudge_msg})
                        continue
                    return LmStudioToolLoopResult(
                        response=AdapterResponse(success=bool(content.strip()), raw_output=content, error="" if content.strip() else "empty_lmstudio_direct_output", duration_seconds=time.monotonic() - started),
                        transcript=transcript,
                        tool_call_count=tool_call_count,
                        expensive_tool_call_count=expensive_count,
                    )
                messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                for call in tool_calls:
                    tool_name = str(((call.get("function") or {}).get("name")) or "").strip()
                    call_id = str(call.get("id") or f"call_{tool_call_count + 1}")
                    args_text = str(((call.get("function") or {}).get("arguments")) or "{}")
                    try:
                        arguments = json.loads(args_text) if args_text.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {}
                    effective_arguments = arguments
                    repair_notes: list[str] = []
                    tool_attempt_count += 1
                    coerced_arguments, coerce_notes = _coerce_string_json_fields(dict(effective_arguments))
                    if coerce_notes:
                        effective_arguments = coerced_arguments
                        repair_notes.extend(coerce_notes)
                    repaired_arguments, handle_notes = repair_handle_arguments(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        state=handle_state,
                        catalog_snapshot=self.catalog_snapshot,
                        runtime_profile=self.runtime_profile,
                    )
                    if handle_notes:
                        effective_arguments = repaired_arguments
                        repair_notes.extend(handle_notes)
                    effective_arguments, setup_notes = self._maybe_rewrite_research_setup_open_to_create(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        handle_state=handle_state,
                    )
                    if setup_notes:
                        repair_notes.extend(setup_notes)
                    repaired_feature_arguments, feature_notes = repair_feature_analytics_identifier_from_transcript(
                        transcript=transcript,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                    )
                    if feature_notes:
                        effective_arguments = repaired_feature_arguments
                        repair_notes.extend(feature_notes)
                    try:
                        preflight = preflight_direct_tool_call(
                            tool_name,
                            effective_arguments,
                            catalog_snapshot=self.catalog_snapshot,
                            confirmed_handles={
                                "project_id": handle_state.confirmed_project_id,
                                "job_id": handle_state.confirmed_job_id or handle_state.confirmed_experiment_job_id,
                                "run_id": handle_state.confirmed_run_id,
                                "snapshot_id": handle_state.confirmed_snapshot_id,
                                "operation_id": handle_state.confirmed_operation_id,
                            },
                            transcript=transcript,
                            runtime_profile=self.runtime_profile,
                            baseline_bootstrap=self._baseline_bootstrap,
                        )
                    except TypeError:
                        preflight = preflight_direct_tool_call(
                            tool_name,
                            effective_arguments,
                            catalog_snapshot=self.catalog_snapshot,
                        )
                    effective_arguments = preflight.arguments
                    if preflight.repair_notes:
                        repair_notes.extend(preflight.repair_notes)
                    coerced_args, coercion_note = coerce_no_op_terminal_write(
                        runtime_profile=self.runtime_profile,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        transcript=transcript,
                        project_id=handle_state.confirmed_project_id,
                        allowed_tools=self.allowed_tools,
                    )
                    if coerced_args is not None:
                        effective_arguments = coerced_args
                        if coercion_note:
                            repair_notes.append(coercion_note)
                        self.incident_store.record(
                            summary="Direct runtime coerced no-op tool call into canonical terminal-write payload",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "runtime_profile": self.runtime_profile,
                                "tool_name": tool_name,
                                "original_arguments": arguments,
                                "coerced_arguments": effective_arguments,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                    if tool_name not in self.allowed_tools or tool_name in self.safe_exclude_tools:
                        result_payload = {"error": f"direct_tool_not_allowed:{tool_name}", "allowed_tools": sorted(self.allowed_tools)}
                    elif preflight.local_payload is not None:
                        result_payload = preflight.local_payload
                        details = result_payload.get("details") if isinstance(result_payload.get("details"), dict) else {}
                        reason_code = str(details.get("reason_code") or "").strip()
                        if reason_code in {
                            "suspicious_durable_handle",
                            "backtests_plan_required_before_start",
                            "duplicate_baseline_start_blocked",
                            "backtests_start_blocked_after_handle_misuse",
                            "prompt_allowed_tool_contract_mismatch",
                        }:
                            self.incident_store.record(
                                summary="Direct local preflight blocked unsafe tool call",
                                metadata={
                                    "plan_id": plan_id,
                                    "slice_id": slice_id,
                                    "slice_title": self.slice_title,
                                    "tool_name": tool_name,
                                    "arguments": effective_arguments,
                                    "handle_field": str(details.get("field_name") or ""),
                                    "handle_value": str(details.get("handle_value") or ""),
                                    "reason_code": reason_code,
                                    "runtime_profile": self.runtime_profile,
                                    "allowed_tools": sorted(self.allowed_tools),
                                    "transcript_len": len(transcript),
                                },
                                source="direct_runtime",
                                severity="medium",
                            )
                    elif tool_call_count >= self.max_tool_calls:
                        salvage_raw = self._build_budget_salvage_report(transcript, salvage_reason="budget")
                        if salvage_raw is not None:
                            self.incident_store.record(
                                summary="Direct slice auto-finalized from research transcript after budget exhaustion",
                                metadata={
                                    "plan_id": plan_id,
                                    "slice_id": slice_id,
                                    "slice_title": self.slice_title,
                                    "tool_name": tool_name,
                                    "max_tool_calls": self.max_tool_calls,
                                    "used_tool_calls": tool_call_count,
                                },
                                source="direct_runtime",
                                severity="low",
                            )
                            # Salvage is telemetry after budget exhaustion, not a real
                            # completion — zero out tool_call_count so the quality gate
                            # rejects it and fallback providers get a chance.
                            return LmStudioToolLoopResult(
                                response=AdapterResponse(
                                    success=True,
                                    raw_output=salvage_raw,
                                    duration_seconds=time.monotonic() - started,
                                ),
                                transcript=transcript,
                                tool_call_count=0,
                                expensive_tool_call_count=expensive_count,
                            )
                        result_payload = {"error": "direct_tool_budget_exhausted", "max_tool_calls": self.max_tool_calls}
                        self.incident_store.record(
                            summary="Direct tool budget exhausted",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "tool_name": tool_name,
                                "max_tool_calls": self.max_tool_calls,
                                "used_tool_calls": tool_call_count,
                            },
                            source="direct_runtime",
                            severity="medium",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=build_watchdog_checkpoint(
                                    summary=(
                                        f"Slice blocked after tool budget exhausted ({tool_call_count}/{self.max_tool_calls}). "
                                        f"Last requested tool: {tool_name}."
                                    ),
                                    reason_code="direct_tool_budget_exhausted",
                                    facts={
                                        "last_error_tool": tool_name,
                                        "tool_call_count": tool_call_count,
                                        "max_tool_calls": self.max_tool_calls,
                                    },
                                ),
                                duration_seconds=time.monotonic() - started,
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    elif self.catalog_snapshot is not None and is_expensive_tool_call(self.catalog_snapshot, tool_name, effective_arguments) and expensive_count >= self.max_expensive_tool_calls:
                        expensive_salvage_raw = self._build_budget_salvage_report(transcript, salvage_reason="expensive_budget")
                        if expensive_salvage_raw is not None:
                            self.incident_store.record(
                                summary="Direct slice auto-finalized after expensive budget exhaustion",
                                metadata={
                                    "plan_id": plan_id,
                                    "slice_id": slice_id,
                                    "slice_title": self.slice_title,
                                    "tool_name": tool_name,
                                    "max_expensive_tool_calls": self.max_expensive_tool_calls,
                                    "used_expensive_tool_calls": expensive_count,
                                },
                                source="direct_runtime",
                                severity="low",
                            )
                            # Expensive-budget salvage is telemetry; zero out the count so
                            # the quality gate rejects it and fallback providers run.
                            return LmStudioToolLoopResult(
                                response=AdapterResponse(
                                    success=True,
                                    raw_output=expensive_salvage_raw,
                                    duration_seconds=time.monotonic() - started,
                                ),
                                transcript=transcript,
                                tool_call_count=0,
                                expensive_tool_call_count=expensive_count,
                            )
                        result_payload = {"error": "direct_expensive_tool_budget_exhausted", "max_expensive_tool_calls": self.max_expensive_tool_calls}
                        self.incident_store.record(
                            summary="Direct expensive tool budget exhausted",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "tool_name": tool_name,
                                "max_expensive_tool_calls": self.max_expensive_tool_calls,
                                "used_expensive_tool_calls": expensive_count,
                            },
                            source="direct_runtime",
                            severity="medium",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=build_watchdog_checkpoint(
                                    summary=(
                                        "Slice blocked after expensive-tool budget exhausted "
                                        f"({expensive_count}/{self.max_expensive_tool_calls}). "
                                        f"Last requested tool: {tool_name}."
                                    ),
                                    reason_code="direct_expensive_tool_budget_exhausted",
                                    facts={
                                        "last_error_tool": tool_name,
                                        "expensive_tool_call_count": expensive_count,
                                        "max_expensive_tool_calls": self.max_expensive_tool_calls,
                                    },
                                ),
                                duration_seconds=time.monotonic() - started,
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    else:
                        if preflight.charge_budget:
                            if "first_tool_call" not in timing_marks:
                                timing_marks["first_tool_call"] = int((time.monotonic() - started) * 1000)
                                if on_progress is not None:
                                    on_progress("first_tool_call", {"elapsed_ms": timing_marks["first_tool_call"], "tool_name": tool_name})
                            tool_call_count += 1
                            if self.catalog_snapshot is not None and is_expensive_tool_call(self.catalog_snapshot, tool_name, effective_arguments):
                                expensive_count += 1
                        result_payload = await self._call_tool(tool_name=tool_name, arguments=effective_arguments, plan_id=plan_id, slice_id=slice_id)
                    feature_contract_identifier_report = build_feature_contract_identifier_report(
                        transcript=transcript,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        allowed_tools=self.allowed_tools,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                        required_output_facts=self.required_output_facts,
                    )
                    if feature_contract_identifier_report is not None:
                        self.incident_store.record(
                            summary="Direct feature-contract slice auto-finalized after predictable identifier misuse",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "arguments": effective_arguments,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=feature_contract_identifier_report,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    had_contract_issue = had_contract_issue or _is_contract_issue_payload(result_payload)
                    _local_err = (
                        result_payload.get("ok") is False
                        and result_payload.get("error_class") == "agent_contract_misuse"
                    )
                    _mcp_err = _is_contract_issue_payload(result_payload)
                    contract_error_streak = contract_error_streak + 1 if (_local_err or _mcp_err) else 0
                    if contract_error_streak >= 3:
                        self.incident_store.record(
                            summary=(
                                f"Direct error loop: {contract_error_streak} consecutive contract/misuse errors, "
                                f"last tool: {tool_name}"
                            ),
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "tool_name": tool_name,
                                "streak": contract_error_streak,
                            },
                            source="direct_runtime",
                            severity="medium",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=build_watchdog_checkpoint(
                                    summary=(
                                        f"Slice blocked after {contract_error_streak} consecutive contract errors. "
                                        f"Last tool: {tool_name}. "
                                        "Follow tool contract hints to fix call parameters before retrying."
                                    ),
                                    reason_code="direct_error_loop_detected",
                                    facts={"last_error_tool": tool_name, "error_streak": contract_error_streak},
                                ),
                                duration_seconds=time.monotonic() - started,
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    signature = tool_call_signature(tool_name, effective_arguments)
                    if signature and signature == last_signature:
                        repeat_count += 1
                    else:
                        repeat_count = 1 if signature else 0
                        last_signature = signature
                    transcript.append(
                        {
                            "kind": "tool_result",
                            "tool": tool_name,
                            "arguments": effective_arguments,
                            "original_arguments": arguments,
                            "repair_notes": repair_notes,
                            "payload": result_payload,
                        }
                    )
                    if on_progress is not None:
                        on_progress(
                            "tool_result",
                            {
                                "transcript_len": len(transcript),
                                "tool_name": tool_name,
                                "tool_call_count": tool_call_count,
                                "expensive_tool_call_count": expensive_count,
                                "transcript": list(transcript),
                            },
                        )
                    if not result_payload.get("error"):
                        update_handle_state(
                            tool_name=tool_name,
                            arguments=effective_arguments,
                            result_payload=result_payload,
                            state=handle_state,
                            runtime_profile=self.runtime_profile,
                        )
                    research_setup_guard_result = self._maybe_handle_research_setup_project_selection(
                        transcript=transcript,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        handle_state=handle_state,
                        plan_id=plan_id,
                        slice_id=slice_id,
                        started=started,
                        tool_call_count=tool_call_count,
                        expensive_tool_call_count=expensive_count,
                    )
                    if research_setup_guard_result is not None:
                        return research_setup_guard_result
                    auto_finalize = should_auto_finalize_research_slice(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        success_criteria=self.success_criteria,
                        allowed_tools=self.allowed_tools,
                        required_output_facts=self.required_output_facts,
                        prior_contract_issue=had_contract_issue,
                        runtime_profile=self.runtime_profile,
                        finalization_mode=self.finalization_mode,
                    )
                    if auto_finalize:
                        self.incident_store.record(
                            summary="Direct slice auto-finalized after fact-based write",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "arguments": effective_arguments,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                                response=AdapterResponse(
                                    success=True,
                                    raw_output=build_auto_final_report(
                                        arguments=effective_arguments,
                                        result_payload=result_payload,
                                        success_criteria=self.success_criteria,
                                        runtime_profile=self.runtime_profile,
                                        tool_name=tool_name,
                                    ),
                                    duration_seconds=time.monotonic() - started,
                                    metadata={"timings_ms": timing_marks},
                                ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    shortlist_guard_result = self._maybe_handle_shortlist_exploration(
                        transcript=transcript,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        plan_id=plan_id,
                        slice_id=slice_id,
                        started=started,
                        tool_call_count=tool_call_count,
                        expensive_tool_call_count=expensive_count,
                    )
                    if shortlist_guard_result is not None:
                        return shortlist_guard_result
                    mixed_domain_guard_result = self._maybe_handle_mixed_domain_exploration(
                        transcript=transcript,
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        plan_id=plan_id,
                        slice_id=slice_id,
                        started=started,
                        tool_call_count=tool_call_count,
                        expensive_tool_call_count=expensive_count,
                    )
                    if mixed_domain_guard_result is not None:
                        return mixed_domain_guard_result
                    feature_contract_exploration_report = build_feature_contract_exploration_final_report(
                        transcript=transcript,
                        tool_name=tool_name,
                        result_payload=result_payload,
                        allowed_tools=self.allowed_tools,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                        required_output_facts=self.required_output_facts,
                    )
                    if feature_contract_exploration_report is not None:
                        self.incident_store.record(
                            summary="Direct feature-contract exploration slice auto-finalized after sufficient live probes",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "tool_call_count": tool_call_count,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=feature_contract_exploration_report,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    feature_contract_construction_report = build_feature_contract_construction_final_report(
                        transcript=transcript,
                        tool_name=tool_name,
                        result_payload=result_payload,
                        allowed_tools=self.allowed_tools,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                        required_output_facts=self.required_output_facts,
                    )
                    if feature_contract_construction_report is not None:
                        self.incident_store.record(
                            summary="Direct feature-contract construction slice auto-finalized after sufficient live probes",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "tool_call_count": tool_call_count,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=feature_contract_construction_report,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    feature_profitability_filter_report = build_feature_profitability_filter_final_report(
                        transcript=transcript,
                        tool_name=tool_name,
                        result_payload=result_payload,
                        allowed_tools=self.allowed_tools,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                        required_output_facts=self.required_output_facts,
                    )
                    if feature_profitability_filter_report is not None:
                        self.incident_store.record(
                            summary="Direct feature-profitability filter slice auto-finalized after sufficient live probes",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "tool_call_count": tool_call_count,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=feature_profitability_filter_report,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    next_exploration_call = feature_contract_exploration_next_call(
                        transcript=transcript,
                        allowed_tools=self.allowed_tools,
                        slice_title=self.slice_title,
                        slice_objective=self.slice_objective,
                        success_criteria=self.success_criteria,
                        policy_tags=self.policy_tags,
                    )
                    if (
                        next_exploration_call is not None
                        and tool_name in {"features_catalog", "events", "datasets", "research_memory"}
                        and not _is_contract_issue_payload(result_payload)
                    ):
                        missing_tools = feature_contract_exploration_missing_tools(
                            transcript=transcript,
                            allowed_tools=self.allowed_tools,
                            slice_title=self.slice_title,
                            slice_objective=self.slice_objective,
                            success_criteria=self.success_criteria,
                            policy_tags=self.policy_tags,
                        )
                        self._pending_system_nudge = (
                            "Feature-contract exploration protocol: do not finalize yet. "
                            f"Coverage is still missing {', '.join(missing_tools)}. "
                            f"Call {next_exploration_call['tool']} next with arguments {next_exploration_call['arguments']}."
                        )
                    if self._should_auto_finalize_profile_probe(
                        tool_name=tool_name,
                        result_payload=result_payload,
                        tool_call_count=tool_call_count,
                    ):
                        self.incident_store.record(
                            summary="Direct slice auto-finalized after repeated profile probe reads",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "tool_call_count": tool_call_count,
                            },
                            source="direct_runtime",
                            severity="low",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=build_profile_final_report(
                                    transcript=transcript,
                                    success_criteria=self.success_criteria,
                                    required_output_facts=self.required_output_facts,
                                    runtime_profile=self.runtime_profile,
                                ) or build_watchdog_checkpoint(
                                    summary="Strict evidence gate rejected profile-based auto-finalization.",
                                    reason_code="direct_contract_blocker",
                                ),
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    loop_threshold = 5 if signature.startswith("ro:") else 3
                    if repeat_count >= loop_threshold and signature:
                        summary = (
                            f"Direct model entered a semantic loop repeating the same "
                            f"{'read-only' if signature.startswith('ro:') else 'mutating'} call for "
                            f"slice '{self.slice_title or slice_id}'."
                        )
                        self.incident_store.record(
                            summary="Direct semantic loop detected",
                            metadata={
                                "plan_id": plan_id,
                                "slice_id": slice_id,
                                "slice_title": self.slice_title,
                                "tool_name": tool_name,
                                "repeat_count": repeat_count,
                                "signature": signature,
                                "arguments": effective_arguments,
                            },
                            source="direct_runtime",
                            severity="medium",
                        )
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=build_semantic_loop_abort(summary=summary),
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": compact_tool_result_message(
                                tool_name=tool_name,
                                arguments=effective_arguments,
                                result_payload=result_payload,
                                success_criteria=self.success_criteria,
                                runtime_profile=self.runtime_profile,
                                allowed_tools=self.allowed_tools,
                            )[:4000],
                        }
                    )
                    nudge = self._maybe_build_nudge_message(tool_call_count)
                    if nudge:
                        messages.append({"role": "user", "content": nudge})
            return LmStudioToolLoopResult(
                response=AdapterResponse(
                    success=True,
                    raw_output=build_watchdog_checkpoint(
                        summary=(
                            f"Direct model stalled before producing the first action for slice '{self.slice_title or slice_id}'."
                            if tool_call_count == 0
                            else f"Direct model stalled after prior actions for slice '{self.slice_title or slice_id}'."
                        ),
                        reason_code="direct_model_stalled_before_first_action" if tool_call_count == 0 else "direct_model_stalled_between_actions",
                    ),
                    timed_out=True,
                    duration_seconds=time.monotonic() - started,
                    metadata={"timings_ms": timing_marks},
                ),
                transcript=transcript,
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_count,
            )
        finally:
            await self.mcp_client.close()

    def _maybe_build_nudge_message(self, tool_call_count: int) -> str | None:
        if self._pending_system_nudge:
            message = self._pending_system_nudge
            self._pending_system_nudge = ""
            return message
        threshold = max(1, int(self.max_tool_calls * 0.6))
        if tool_call_count < threshold:
            return None
        remaining = self.max_tool_calls - tool_call_count
        return (
            f"Budget alert: {tool_call_count}/{self.max_tool_calls} tool calls used, "
            f"{remaining} remaining. "
            "You have enough evidence. Return your final_report now with "
            "verdict=WATCHLIST or COMPLETE. Do not make more tool calls unless critical."
        )

    def _is_research_setup_slice(self) -> bool:
        return {"research_project", "research_map", "research_memory"}.issubset(self.allowed_tools)

    def _research_setup_create_template(self) -> dict[str, Any]:
        title = self.slice_title.strip() or "Cycle Research Project"
        objective = self.slice_objective.strip() or "Open a dedicated cycle research project and record the setup invariants."
        return {
            "tool": "research_project",
            "arguments": {
                "action": "create",
                "project": {
                    "name": title[:80],
                    "goal": objective[:240],
                },
            },
        }

    def _maybe_rewrite_research_setup_open_to_create(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        handle_state: ResearchHandleState,
    ) -> tuple[dict[str, Any], list[str]]:
        if tool_name != "research_project" or not self._is_research_setup_slice():
            return arguments, []
        action = str(arguments.get("action") or "").strip().lower()
        project_id = str(arguments.get("project_id") or "").strip()
        if action != "open" or handle_state.confirmed_project_id:
            return arguments, []
        if self._research_setup_list_streak < 1 or not _looks_like_transient_project_id(project_id):
            return arguments, []
        replacement = self._research_setup_create_template()["arguments"]
        return (
            replacement,
            [
                "rewrote ambiguous research_project.open into research_project.create for clean-run setup",
                f"discarded transient project_id '{project_id}'",
            ],
        )

    def _maybe_handle_research_setup_project_selection(
        self,
        *,
        transcript: list[dict[str, Any]],
        tool_name: str,
        arguments: dict[str, Any],
        result_payload: dict[str, Any],
        handle_state: ResearchHandleState,
        plan_id: str,
        slice_id: str,
        started: float,
        tool_call_count: int,
        expensive_tool_call_count: int,
    ) -> LmStudioToolLoopResult | None:
        if not self._is_research_setup_slice():
            return None
        if result_payload.get("error") or result_payload.get("ok") is False:
            return None
        if tool_name != "research_project":
            return None
        action = str(arguments.get("action") or "").strip().lower()
        if action in {"create", "open", "current"} and handle_state.confirmed_project_id:
            self._research_setup_list_streak = 0
            self._research_setup_project_nudge_sent = False
            return None
        if action != "list":
            return None
        if handle_state.confirmed_project_id:
            self._research_setup_list_streak = 0
            self._research_setup_project_nudge_sent = False
            return None
        self._research_setup_list_streak += 1
        next_call = self._research_setup_create_template()
        if self._research_setup_list_streak >= 2:
            self.incident_store.record(
                summary="Research setup slice repeated project discovery without opening a concrete project",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "slice_title": self.slice_title,
                    "list_streak": self._research_setup_list_streak,
                    "next_call": next_call,
                },
                source="direct_runtime",
                severity="medium",
            )
            return LmStudioToolLoopResult(
                response=AdapterResponse(
                    success=True,
                    raw_output=build_watchdog_checkpoint(
                        summary=(
                            "Research setup stayed in project discovery without creating or explicitly opening the cycle project. "
                            "Use research_project.create or research_project.open next."
                        ),
                        reason_code="direct_setup_project_selection_ambiguous",
                        facts={
                            "project_list_streak": self._research_setup_list_streak,
                            "next_call_example": next_call,
                        },
                    ),
                    duration_seconds=time.monotonic() - started,
                ),
                transcript=transcript,
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_tool_call_count,
            )
        if not self._research_setup_project_nudge_sent:
            self._research_setup_project_nudge_sent = True
            self._pending_system_nudge = (
                "Research setup protocol reminder: the project list is discovery only and this clean run still has no "
                "confirmed cycle project. Next step: call research_project(action='create', project={name, goal}) or "
                "research_project(action='open', project_id='...') with an explicit persistent project id. "
                "Do not use server_session_id as project_id."
            )
        return None

    def _maybe_handle_shortlist_exploration(
        self,
        *,
        transcript: list[dict[str, Any]],
        tool_name: str,
        arguments: dict[str, Any],
        result_payload: dict[str, Any],
        plan_id: str,
        slice_id: str,
        started: float,
        tool_call_count: int,
        expensive_tool_call_count: int,
    ) -> LmStudioToolLoopResult | None:
        if self.runtime_profile != "research_shortlist":
            return None
        if result_payload.get("error") or result_payload.get("ok") is False:
            return None
        facts_before = derive_facts_from_transcript(transcript[:-1], runtime_profile=self.runtime_profile)
        facts_after = derive_facts_from_transcript(transcript, runtime_profile=self.runtime_profile)
        missing_before = self._missing_shortlist_facts(facts_before)
        missing_after = self._missing_shortlist_facts(facts_after)
        write_facts = derive_research_write_facts(
            arguments=arguments,
            result_payload=result_payload,
            runtime_profile=self.runtime_profile,
            tool_name=tool_name,
        )
        if self._is_terminal_shortlist_write(write_facts):
            self._exploration_streak = 0
            return None
        if len(missing_after) < len(missing_before):
            self._exploration_streak = 0
            return None
        self._exploration_streak += 1
        if self._exploration_streak >= 4:
            project_id = str(facts_after.get("research.project_id") or facts_before.get("research.project_id") or "project-id").strip()
            next_call = self._shortlist_write_template(project_id=project_id)
            self.incident_store.record(
                summary="Direct shortlist slice blocked after repeated exploration without terminal write",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "tool_name": tool_name,
                    "exploration_streak": self._exploration_streak,
                    "missing_required_facts": missing_after,
                    "next_call": next_call,
                },
                source="direct_runtime",
                severity="medium",
            )
            return LmStudioToolLoopResult(
                response=AdapterResponse(
                    success=True,
                    raw_output=build_watchdog_checkpoint(
                        summary=(
                            "Slice stayed in exploration without the required shortlist milestone write. "
                            f"Missing facts: {', '.join(missing_after) or 'none'}."
                        ),
                        reason_code="direct_missing_terminal_write",
                        facts={
                            "missing_required_facts": missing_after,
                            "next_call_example": next_call,
                            "exploration_streak": self._exploration_streak,
                        },
                    ),
                    duration_seconds=time.monotonic() - started,
                ),
                transcript=transcript,
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_tool_call_count,
            )
        if self._exploration_streak >= 2 and not self._shortlist_nudge_sent:
            self._shortlist_nudge_sent = True
            next_call = self._shortlist_write_template(
                project_id=str(facts_after.get("research.project_id") or facts_before.get("research.project_id") or "project-id").strip()
            )
            self._pending_system_nudge = (
                "Research shortlist protocol reminder: read/map calls do not complete this slice. "
                f"Next step: call {next_call['tool']}(action='create', kind='milestone', ...) with metadata.shortlist_families, "
                "metadata.novelty_justification_present=true, and content.candidates=[{family, why_new, relative_to}]."
            )
        return None

    def _missing_shortlist_facts(self, facts: dict[str, Any]) -> list[str]:
        return [key for key in self.required_output_facts if _is_missing_fact(facts.get(key))]

    def _is_terminal_shortlist_write(self, facts: dict[str, Any]) -> bool:
        return not self._missing_shortlist_facts(facts)

    def _shortlist_write_template(self, *, project_id: str) -> dict[str, Any]:
        tool_name = "research_memory" if "research_memory" in self.allowed_tools else "research_record" if "research_record" in self.allowed_tools else "research_memory"
        return {
            "tool": tool_name,
            "arguments": {
                "action": "create",
                "kind": "milestone",
                "project_id": project_id or "project-id",
                "record": {
                    "title": "Wave 1 novel signal shortlist",
                    "summary": "Recorded the first-wave shortlist with novelty justification versus the base space and history v1-v12.",
                    "metadata": {
                        "stage": "hypothesis_formation",
                        "outcome": "shortlist_recorded",
                        "shortlist_families": ["funding dislocation", "expiry proximity"],
                        "novelty_justification_present": True,
                    },
                    "content": {
                        "candidates": [
                            {
                                "family": "funding dislocation",
                                "why_new": "Explain why this family is new relative to the base space and v1-v12.",
                                "relative_to": ["base", "v1-v12"],
                            }
                        ]
                    },
                },
            },
        }

    def _should_auto_finalize_profile_probe(
        self,
        *,
        tool_name: str,
        result_payload: dict[str, Any],
        tool_call_count: int,
    ) -> bool:
        del tool_name
        if self.runtime_profile != "catalog_contract_probe":
            return False
        if tool_call_count < 3:
            return False
        if result_payload.get("error"):
            return False
        if result_payload.get("ok") is False:
            return False
        return True

    def _build_first_action_guide(self) -> str:
        tools = ", ".join(sorted(self.allowed_tools)[:8]) or "no tools"
        first_criterion = str(self.success_criteria[0] or "").strip() if self.success_criteria else ""
        if self._is_backtests_slice():
            return build_backtests_first_action_guide(
                allowed_tools=self.allowed_tools,
                baseline_bootstrap=self._baseline_bootstrap,
                first_criterion=first_criterion,
            )
        if self._is_backtests_analysis_context_slice():
            preferred_tool = self._mixed_domain_domain_tools()[0] if self._mixed_domain_domain_tools() else "backtests_analysis"
            parts = [
                "PROTOCOL:",
                "Step 1: If candidate context is missing, do at most one targeted research_memory.search.",
                f"Step 2: Switch immediately to {preferred_tool} for live stability/integration/cannibalization evidence.",
                "Step 3: Avoid repeated research_memory searches once the shortlist context is known.",
                "Step 4: Return exactly one final_report JSON with evidence_refs referencing the live non-research results.",
            ]
            if first_criterion:
                parts.append(f"Prioritize evidence for: {first_criterion}.")
            parts.extend([
                "",
                f"IMPORTANT: Use {preferred_tool} rather than looping on research_memory.",
            ])
            return " ".join(parts)
        parts = [
            "PROTOCOL:",
            f"Step 1: Call one tool from [{tools}].",
            "Step 2: Read the tool result.",
            "Step 3: Return exactly one final_report JSON with evidence_refs referencing the tool results.",
        ]
        if first_criterion:
            parts.append(f"Prioritize evidence for: {first_criterion}.")
        first_tool = sorted(self.allowed_tools)[0] if self.allowed_tools else "tool"
        parts.extend([
            "",
            f'Example: 1) Call {first_tool}({{...}}) -> get data. '
            f'2) Return: {{"type":"final_report","summary":"...","verdict":"WATCHLIST",'
            f'"findings":["..."],"facts":{{}},"evidence_refs":["transcript:1:{first_tool}"],"confidence":0.7}}',
            "",
            "Use exact evidence_refs from successful tool results when possible. "
            "Accepted refs include transcript tool-result refs and concrete ids like node_* or note_*.",
            "",
            f"IMPORTANT: You MUST call at least one tool from [{tools}]. "
            "Returning text without calling a tool is a failure.",
        ])
        if self._is_mixed_domain_contract_slice():
            domain_tools = ", ".join(self._domain_contract_tools()[:4]) or "non-research domain tools"
            parts.extend([
                "",
                "Mixed-domain protocol:",
                "Use research_memory only to recover shortlist/project context.",
                f"After at most 2 research reads, switch to one of [{domain_tools}].",
                "Repeated research_memory.search calls do not satisfy feature/data contract evidence.",
            ])
        return " ".join(parts)

    def _build_zero_tool_nudge_message(self) -> str:
        if self._is_backtests_slice():
            return build_backtests_zero_tool_nudge(
                allowed_tools=self.allowed_tools,
                baseline_bootstrap=self._baseline_bootstrap,
            )
        nudge_tools = ", ".join(sorted(self.allowed_tools)[:8])
        return (
            f"You did not call any tool. You MUST call one of [{nudge_tools}] now. "
            "Call a tool first, then return your final_report JSON."
        )

    def _maybe_auto_finalize_after_stall(self, transcript: list[dict[str, Any]]) -> str | None:
        if feature_contract_exploration_missing_tools(
            transcript=transcript,
            allowed_tools=self.allowed_tools,
            slice_title=self.slice_title,
            slice_objective=self.slice_objective,
            success_criteria=self.success_criteria,
            policy_tags=self.policy_tags,
        ):
            return None
        profiled = build_profile_final_report(
            transcript=transcript,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
            runtime_profile=self.runtime_profile,
        )
        if profiled is not None:
            return profiled
        return build_generic_transcript_salvage_report(
            transcript=transcript,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
            runtime_profile=self.runtime_profile,
            slice_title=self.slice_title,
            salvage_reason="stall",
        )

    def _build_budget_salvage_report(self, transcript: list[dict[str, Any]], *, salvage_reason: str = "budget") -> str | None:
        if feature_contract_exploration_missing_tools(
            transcript=transcript,
            allowed_tools=self.allowed_tools,
            slice_title=self.slice_title,
            slice_objective=self.slice_objective,
            success_criteria=self.success_criteria,
            policy_tags=self.policy_tags,
        ):
            return None
        profiled = build_profile_final_report(
            transcript=transcript,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
            runtime_profile=self.runtime_profile,
        )
        if profiled is not None:
            return profiled
        return build_generic_transcript_salvage_report(
            transcript=transcript,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
            runtime_profile=self.runtime_profile,
            slice_title=self.slice_title,
            salvage_reason=salvage_reason,
            minimum_successful_tools=self._minimum_successful_budget_salvage_tools(),
        )

    def _minimum_successful_budget_salvage_tools(self) -> int:
        if self.runtime_profile == "research_memory" and len(self.allowed_tools) > 1:
            return 1
        return 2

    def _is_mixed_domain_contract_slice(self) -> bool:
        tool_set = {str(item).strip() for item in self.allowed_tools if str(item).strip()}
        has_research = "research_memory" in tool_set
        has_domain_contract_tool = bool(
            tool_set
            & {
                "features_catalog",
                "events",
                "datasets",
                "features_custom",
                "features_dataset",
                "features_analytics",
                "models_dataset",
            }
        )
        return self.runtime_profile == "generic_mutation" and has_research and has_domain_contract_tool

    def _is_backtests_analysis_context_slice(self) -> bool:
        tool_set = {str(item).strip() for item in self.allowed_tools if str(item).strip()}
        if "research_memory" not in tool_set:
            return False
        if not (tool_set & {"backtests_conditions", "backtests_analysis", "backtests_runs"}):
            return False
        haystack = " ".join(
            str(item).strip().lower()
            for item in (
                self.slice_title,
                self.slice_objective,
                *self.success_criteria,
                *self.policy_tags,
            )
            if str(item).strip()
        )
        markers = (
            "stability",
            "condition analysis",
            "integration",
            "cannibal",
            "ownership",
            "new-entry proof",
            "analysis",
        )
        return any(marker in haystack for marker in markers)

    def _is_backtests_slice(self) -> bool:
        return is_backtests_context(
            runtime_profile=self.runtime_profile,
            title=self.slice_title,
            objective=self.slice_objective,
            success_criteria=list(self.success_criteria or []),
            policy_tags=list(self.policy_tags or []),
            allowed_tools=self.allowed_tools,
        )

    def _format_backtests_plan_call(self) -> str:
        snapshot_id = str(
            self._baseline_bootstrap.get("baseline_snapshot_id")
            or self._baseline_bootstrap.get("snapshot_id")
            or "active-signal-v1"
        )
        version = self._baseline_bootstrap.get("baseline_version", 1)
        symbol = str(self._baseline_bootstrap.get("symbol", "BTCUSDT") or "BTCUSDT")
        anchor_timeframe = str(self._baseline_bootstrap.get("anchor_timeframe", "1h") or "1h")
        execution_timeframe = str(self._baseline_bootstrap.get("execution_timeframe", "5m") or "5m")
        return (
            "backtests_plan("
            f"snapshot_id='{snapshot_id}', version={version}, symbol='{symbol}', "
            f"anchor_timeframe='{anchor_timeframe}', execution_timeframe='{execution_timeframe}')"
        )

    def _domain_contract_tools(self) -> list[str]:
        preferred = (
            "features_catalog",
            "events",
            "datasets",
            "features_custom",
            "features_dataset",
            "features_analytics",
            "models_dataset",
        )
        return [name for name in preferred if name in self.allowed_tools]

    def _backtests_domain_tools(self) -> list[str]:
        preferred = (
            "backtests_plan",
            "backtests_runs",
            "backtests_conditions",
            "backtests_analysis",
        )
        return [name for name in preferred if name in self.allowed_tools]

    def _mixed_domain_domain_tools(self) -> list[str]:
        if self._is_mixed_domain_contract_slice():
            return self._domain_contract_tools()
        if self._is_backtests_analysis_context_slice() or self._is_backtests_slice():
            return self._backtests_domain_tools()
        return []

    def _feature_contract_next_call(self) -> dict[str, Any]:
        if "features_catalog" in self.allowed_tools:
            return {"tool": "features_catalog", "arguments": {"scope": "available"}}
        if "events" in self.allowed_tools:
            return {"tool": "events", "arguments": {"view": "catalog"}}
        if "datasets" in self.allowed_tools:
            return {"tool": "datasets", "arguments": {"view": "catalog"}}
        if "features_custom" in self.allowed_tools:
            return {"tool": "features_custom", "arguments": {"action": "inspect", "view": "contract"}}
        if "models_dataset" in self.allowed_tools:
            return {"tool": "models_dataset", "arguments": {"action": "contract"}}
        if "features_dataset" in self.allowed_tools:
            return {"tool": "features_dataset", "arguments": {"action": "inspect", "view": "columns"}}
        domain_tools = self._domain_contract_tools()
        fallback_tool = domain_tools[0] if domain_tools else "tool"
        return {"tool": fallback_tool, "arguments": {}}

    def _backtests_analysis_next_call(self) -> dict[str, Any]:
        preferred = self._backtests_domain_tools()
        fallback_tool = preferred[0] if preferred else "backtests_analysis"
        return {"tool": fallback_tool, "arguments": {}}

    def _successful_mixed_domain_domain_tool_count(self, transcript: list[dict[str, Any]]) -> int:
        count = 0
        domain_tools = set(self._mixed_domain_domain_tools())
        for entry in transcript:
            if entry.get("kind") != "tool_result":
                continue
            tool_name = str(entry.get("tool") or "").strip()
            if tool_name not in domain_tools:
                continue
            payload = entry.get("payload") if isinstance(entry.get("payload"), dict) else {}
            if payload.get("error") or payload.get("ok") is False:
                continue
            count += 1
        return count

    def _maybe_handle_mixed_domain_exploration(
        self,
        *,
        transcript: list[dict[str, Any]],
        tool_name: str,
        arguments: dict[str, Any],
        result_payload: dict[str, Any],
        plan_id: str,
        slice_id: str,
        started: float,
        tool_call_count: int,
        expensive_tool_call_count: int,
    ) -> LmStudioToolLoopResult | None:
        if not (
            self._is_mixed_domain_contract_slice()
            or self._is_backtests_analysis_context_slice()
            or self._is_backtests_slice()
        ):
            return None
        if result_payload.get("error") or result_payload.get("ok") is False:
            return None
        if tool_name in self._mixed_domain_domain_tools():
            self._mixed_domain_read_streak = 0
            self._mixed_domain_nudge_sent = False
            self._mixed_domain_finalize_nudge_sent = False
            return None
        if tool_name != "research_memory":
            return None
        action = str(arguments.get("action") or "").strip().lower()
        if action in {"create", "update", "complete_work_item", "complete_work_items"}:
            self._mixed_domain_read_streak = 0
            return None
        self._mixed_domain_read_streak += 1
        domain_success_count = self._successful_mixed_domain_domain_tool_count(transcript)
        if domain_success_count >= 2:
            if self._mixed_domain_read_streak >= 4:
                salvage_raw = self._build_budget_salvage_report(transcript, salvage_reason="late_context_loop")
                if salvage_raw is not None:
                    self.incident_store.record(
                        summary="Direct mixed-domain slice auto-finalized after late research-only loop",
                        metadata={
                            "plan_id": plan_id,
                            "slice_id": slice_id,
                            "slice_title": self.slice_title,
                            "tool_name": tool_name,
                            "read_streak": self._mixed_domain_read_streak,
                            "domain_success_count": domain_success_count,
                            "allowed_tools": sorted(self.allowed_tools),
                        },
                        source="direct_runtime",
                        severity="low",
                    )
                    # Late-context-loop salvage is telemetry only; zero the count so
                    # the quality gate rejects the synthesized final_report and the
                    # orchestrator runs the fallback chain.
                    return LmStudioToolLoopResult(
                        response=AdapterResponse(
                            success=True,
                            raw_output=salvage_raw,
                            duration_seconds=time.monotonic() - started,
                        ),
                        transcript=transcript,
                        tool_call_count=0,
                        expensive_tool_call_count=expensive_tool_call_count,
                    )
            if self._mixed_domain_read_streak >= 2 and not self._mixed_domain_finalize_nudge_sent:
                self._mixed_domain_finalize_nudge_sent = True
                if self._is_backtests_analysis_context_slice() or self._is_backtests_slice():
                    self._pending_system_nudge = (
                        "Backtests protocol reminder: you already have live non-research backtest evidence. "
                        "Do not keep searching research_memory for wording. Return final_report now unless one last "
                        "targeted non-research probe is strictly necessary."
                    )
                else:
                    self._pending_system_nudge = (
                        "Mixed-domain protocol reminder: you already have live non-research evidence. "
                        "Do not keep searching research_memory for wording. Return final_report now unless one last "
                        "targeted non-research probe is strictly necessary."
                    )
            return None
        next_call = (
            self._backtests_analysis_next_call()
            if self._is_backtests_analysis_context_slice() or self._is_backtests_slice()
            else self._feature_contract_next_call()
        )
        if self._mixed_domain_read_streak >= 5:
            self.incident_store.record(
                summary="Direct mixed-domain slice stayed in research-only exploration",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "slice_title": self.slice_title,
                    "tool_name": tool_name,
                    "read_streak": self._mixed_domain_read_streak,
                    "next_call": next_call,
                    "allowed_tools": sorted(self.allowed_tools),
                },
                source="direct_runtime",
                severity="medium",
            )
            return LmStudioToolLoopResult(
                response=AdapterResponse(
                    success=True,
                    raw_output=build_watchdog_checkpoint(
                        summary=(
                            "Slice stayed in research-only exploration without non-research evidence. "
                            f"Next step should use {next_call['tool']} instead of another research_memory search."
                        ),
                        reason_code="direct_mixed_domain_exploration_loop",
                        facts={
                            "research_only_read_streak": self._mixed_domain_read_streak,
                            "next_call_example": next_call,
                        },
                    ),
                    duration_seconds=time.monotonic() - started,
                ),
                transcript=transcript,
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_tool_call_count,
            )
        if self._mixed_domain_read_streak >= 1 and not self._mixed_domain_nudge_sent:
            self._mixed_domain_nudge_sent = True
            if self._is_backtests_analysis_context_slice() or self._is_backtests_slice():
                self._pending_system_nudge = (
                    "Backtests protocol reminder: research_memory search is context recovery only. "
                    f"Stop repeating it and call {next_call['tool']} next to gather live backtest evidence."
                )
            else:
                self._pending_system_nudge = (
                    "Feature contract protocol reminder: research_memory search is context recovery only. "
                    f"Stop repeating it and call {next_call['tool']} next to gather non-research contract evidence."
                )
        return None

    def _openai_tool_schemas(self) -> list[dict[str, Any]]:
        if self.catalog_snapshot is None:
            return []
        result: list[dict[str, Any]] = []
        for tool in self.catalog_snapshot.tools:
            name = str(tool.name or "").strip()
            if name not in self.allowed_tools or name in self.safe_exclude_tools:
                continue
            result.append(tool.to_openai_tool_schema())
        return result

    async def _call_tool(self, *, tool_name: str, arguments: dict[str, Any], plan_id: str, slice_id: str) -> dict[str, Any]:
        try:
            payload = _to_jsonable(await self.mcp_client.call_tool(tool_name, arguments))
        except Exception as exc:
            error_text = str(exc)
            self.incident_store.record(
                summary=f"Direct dev_space1 tool call failed: {tool_name}",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "error": error_text,
                    "runtime": "direct_lmstudio_mcp",
                },
                source="direct_runtime",
            )
            return {"error": error_text, "tool_name": tool_name}
        text = json.dumps(payload, ensure_ascii=False).lower()
        if any(token in text for token in ("resource_not_found", "schema_validation_failed", "agent_contract_misuse", "not valid under any of the given schemas")):
            issue_class = classify_issue_payload(payload)
            self.incident_store.record(
                summary=f"Direct dev_space1 {'contract' if issue_class == 'contract_misuse' else 'infra'} issue: {tool_name}",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "issue_class": issue_class,
                    "raw_payload_excerpt": json.dumps(payload, ensure_ascii=False)[:4000],
                    "runtime": "direct_lmstudio_mcp",
                },
                source="direct_runtime",
            )
        return {"ok": True, "payload": payload}

    async def _chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: int, tool_choice: str = "auto") -> dict[str, Any]:
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(self._chat_sync, messages=messages, tools=tools, timeout=timeout, tool_choice=tool_choice),
                    timeout=max(1, timeout + 2),
                )
                if result.get("error") and attempt < max_retries:
                    err = str(result.get("error", ""))
                    # Model crash is not retryable — fail immediately
                    if "lmstudio_model_crash" in err:
                        logger.error("LM Studio model crash — no retry: %s", err[:300])
                        return result
                    retryable = any(tok in err.lower() for tok in ("no host specified", "connection", "timeout", "refused", "reset"))
                    if retryable:
                        logger.warning(f"LM Studio _chat retryable error (attempt {attempt + 1}/{max_retries}): {err}")
                        await asyncio.sleep(2.0)
                        continue
                return result
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"LM Studio _chat timeout (attempt {attempt + 1}/{max_retries}), retrying")
                    await asyncio.sleep(2.0)
                    continue
                return {"error": f"lmstudio_chat_timeout_after_{timeout}s"}
        return {"error": f"lmstudio_chat_timeout_after_{timeout}s"}

    def _chat_sync(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: int, tool_choice: str = "auto") -> dict[str, Any]:
        return self._connection_pool.chat_completion(
            messages=messages,
            tools=tools,
            temperature=self._get_adaptive_temp(self.adapter.temperature),
            max_tokens=self.adapter.max_tokens,
            model=self.adapter.model or "",
            reasoning_effort=self.adapter.reasoning_effort or "",
            extra_body=self.adapter.extra_body,
            tool_choice=tool_choice,
        )


def _is_contract_issue_payload(result_payload: dict[str, Any]) -> bool:
    text = json.dumps(result_payload, ensure_ascii=False).lower()
    return any(
        token in text
        for token in (
            "resource_not_found",
            "schema_validation_failed",
            "agent_contract_misuse",
            "not valid under any of the given schemas",
            "unknown research project",
        )
    )


def _looks_like_transient_project_id(value: str) -> bool:
    text = str(value or "").strip().lower()
    if len(text) < 16:
        return False
    return all(ch in "0123456789abcdef" for ch in text)


def _is_missing_fact(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not bool(value)
    return False


__all__ = ["LmStudioToolLoop", "LmStudioToolLoopResult"]
