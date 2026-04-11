"""
LM Studio direct OpenAI-tool loop backed by dev_space1 MCP calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import Any, Callable
from urllib.parse import urlparse

from app.adapters.base import AdapterResponse
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.lmstudio_finalization import (
    build_backtests_budget_salvage_report,
    build_catalog_only_final_report,
    build_generic_transcript_salvage_report,
    build_research_budget_salvage_report,
)
from app.services.direct_execution.mcp_client import DirectMcpClient, _to_jsonable
from app.services.direct_execution.research_handles import (
    ResearchHandleState,
    repair_atlas_coordinates,
    repair_experiment_handle,
    repair_project_handle,
    update_handle_state,
)
from app.services.direct_execution.semantic_progress import (
    build_auto_final_report,
    build_semantic_loop_abort,
    build_watchdog_checkpoint,
    compact_tool_result_message,
    should_auto_finalize_research_slice,
    tool_call_signature,
)
from app.services.direct_execution.tool_preflight import preflight_direct_tool_call
from app.services.direct_execution.tool_catalog import EXPENSIVE_DIRECT_TOOLS

logger = logging.getLogger("orchestrator.direct.lmstudio")


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
        allowed_tools: set[str],
        slice_title: str,
        success_criteria: list[str],
        required_output_facts: list[str],
        max_tool_calls: int,
        max_expensive_tool_calls: int,
        safe_exclude_tools: set[str],
        first_action_timeout_seconds: int,
        stalled_action_timeout_seconds: int,
    ) -> None:
        self.adapter = adapter
        self.mcp_client = mcp_client
        self.incident_store = incident_store
        self.allowed_tools = allowed_tools
        self.slice_title = slice_title
        self.success_criteria = success_criteria
        self.required_output_facts = list(required_output_facts)
        self.max_tool_calls = max(1, int(max_tool_calls or 1))
        self.max_expensive_tool_calls = max(0, int(max_expensive_tool_calls or 0))
        self.safe_exclude_tools = safe_exclude_tools
        self.first_action_timeout_seconds = max(5, int(first_action_timeout_seconds or 5))
        self.stalled_action_timeout_seconds = max(5, int(stalled_action_timeout_seconds or 5))

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
            try:
                tools = await asyncio.wait_for(self._openai_tool_schemas(), timeout=self.first_action_timeout_seconds)
                if on_progress is not None:
                    on_progress("tool_schema_ready", {"tool_count": len(tools)})
                    on_progress(
                        "schema_fetch",
                        {"elapsed_ms": int((time.monotonic() - started) * 1000), "tool_count": len(tools), "fallback": False},
                    )
            except Exception as exc:
                self.incident_store.record(
                    summary="Direct schema fetch failed; using fallback tool schemas",
                    metadata={
                        "plan_id": plan_id,
                        "slice_id": slice_id,
                        "slice_title": self.slice_title,
                        "error": str(exc),
                    },
                    source="direct_runtime",
                    severity="medium",
                )
                tools = self._fallback_tool_schemas()
                if on_progress is not None:
                    on_progress("tool_schema_fallback", {"tool_count": len(tools)})
                    on_progress(
                        "schema_fetch",
                        {"elapsed_ms": int((time.monotonic() - started) * 1000), "tool_count": len(tools), "fallback": True},
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
            expensive_count = 0
            last_signature = ""
            repeat_count = 0
            had_contract_issue = False
            contract_error_streak = 0
            handle_state = ResearchHandleState()
            deadline = time.monotonic() + max(1, int(timeout_seconds or 1))
            timing_marks: dict[str, int] = {"schema_fetch": int((time.monotonic() - started) * 1000)}
            while time.monotonic() < deadline:
                idle_budget = self.first_action_timeout_seconds if tool_call_count == 0 else self.stalled_action_timeout_seconds
                response = await self._chat(messages=messages, tools=tools, timeout=max(1, min(int(deadline - time.monotonic()), idle_budget)))
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
                        return LmStudioToolLoopResult(
                            response=AdapterResponse(
                                success=True,
                                raw_output=salvage_raw,
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
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
                    repaired_arguments, handle_notes = repair_project_handle(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        state=handle_state,
                    )
                    if handle_notes:
                        effective_arguments = repaired_arguments
                        repair_notes.extend(handle_notes)
                    experiment_arguments, experiment_notes = repair_experiment_handle(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        state=handle_state,
                    )
                    if experiment_notes:
                        effective_arguments = experiment_arguments
                        repair_notes.extend(experiment_notes)
                    atlas_arguments, atlas_notes = repair_atlas_coordinates(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        state=handle_state,
                    )
                    if atlas_notes:
                        effective_arguments = atlas_arguments
                        repair_notes.extend(atlas_notes)
                    preflight = preflight_direct_tool_call(tool_name, effective_arguments)
                    effective_arguments = preflight.arguments
                    if preflight.repair_notes:
                        repair_notes.extend(preflight.repair_notes)
                    # Preflight may hoist atlas from record -> top-level atlas; run atlas repair
                    # again so coordinate coercion can use the latest normalized shape.
                    atlas_arguments_post, atlas_notes_post = repair_atlas_coordinates(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        state=handle_state,
                    )
                    if atlas_notes_post:
                        effective_arguments = atlas_arguments_post
                        repair_notes.extend(atlas_notes_post)
                    if tool_name not in self.allowed_tools or tool_name in self.safe_exclude_tools:
                        result_payload = {"error": f"direct_tool_not_allowed:{tool_name}", "allowed_tools": sorted(self.allowed_tools)}
                    elif preflight.local_payload is not None:
                        result_payload = preflight.local_payload
                    elif tool_call_count >= self.max_tool_calls:
                        salvage_raw = self._build_research_budget_salvage_report(transcript)
                        if salvage_raw is None:
                            salvage_raw = self._build_backtests_budget_salvage_report(transcript)
                        if salvage_raw is None:
                            salvage_raw = build_generic_transcript_salvage_report(
                                transcript=transcript,
                                allowed_tools=self.allowed_tools,
                                success_criteria=self.success_criteria,
                                required_output_facts=self.required_output_facts,
                                slice_title=self.slice_title,
                            )
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
                            return LmStudioToolLoopResult(
                                response=AdapterResponse(
                                    success=True,
                                    raw_output=salvage_raw,
                                    duration_seconds=time.monotonic() - started,
                                ),
                                transcript=transcript,
                                tool_call_count=tool_call_count,
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
                    elif tool_name in EXPENSIVE_DIRECT_TOOLS and expensive_count >= self.max_expensive_tool_calls:
                        expensive_salvage_raw = self._build_backtests_budget_salvage_report(transcript)
                        if expensive_salvage_raw is not None:
                            self.incident_store.record(
                                summary="Direct slice auto-finalized from backtests transcript after expensive budget exhaustion",
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
                            return LmStudioToolLoopResult(
                                response=AdapterResponse(
                                    success=True,
                                    raw_output=expensive_salvage_raw,
                                    duration_seconds=time.monotonic() - started,
                                ),
                                transcript=transcript,
                                tool_call_count=tool_call_count,
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
                            if tool_name in EXPENSIVE_DIRECT_TOOLS:
                                expensive_count += 1
                        result_payload = await self._call_tool(tool_name=tool_name, arguments=effective_arguments, plan_id=plan_id, slice_id=slice_id)
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
                        on_progress("tool_result", {"transcript_len": len(transcript), "tool_name": tool_name, "tool_call_count": tool_call_count, "expensive_tool_call_count": expensive_count})
                    if not result_payload.get("error"):
                        update_handle_state(
                            tool_name=tool_name,
                            arguments=effective_arguments,
                            result_payload=result_payload,
                            state=handle_state,
                        )
                    if should_auto_finalize_research_slice(
                        tool_name=tool_name,
                        arguments=effective_arguments,
                        result_payload=result_payload,
                        success_criteria=self.success_criteria,
                        allowed_tools=self.allowed_tools,
                        required_output_facts=self.required_output_facts,
                        prior_contract_issue=had_contract_issue,
                    ):
                        self.incident_store.record(
                            summary="Direct slice auto-finalized after successful research_record",
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
                                    ),
                                    duration_seconds=time.monotonic() - started,
                                    metadata={"timings_ms": timing_marks},
                                ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    if self._should_auto_finalize_catalog_only(
                        tool_name=tool_name,
                        result_payload=result_payload,
                        tool_call_count=tool_call_count,
                    ):
                        self.incident_store.record(
                            summary="Direct slice auto-finalized after repeated features_catalog reads",
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
                                raw_output=build_catalog_only_final_report(
                                    transcript=transcript,
                                    success_criteria=self.success_criteria,
                                    required_output_facts=self.required_output_facts,
                                ) or build_watchdog_checkpoint(
                                    summary="Strict evidence gate rejected catalog-only auto-finalization.",
                                    reason_code="direct_contract_blocker",
                                ),
                                duration_seconds=time.monotonic() - started,
                                metadata={"timings_ms": timing_marks},
                            ),
                            transcript=transcript,
                            tool_call_count=tool_call_count,
                            expensive_tool_call_count=expensive_count,
                        )
                    if repeat_count >= 3 and signature:
                        summary = (
                            f"Direct model entered a semantic loop repeating the same research_record write for "
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
                            )[:4000],
                        }
                    )
                    nudge = self._maybe_build_nudge_message(tool_call_count)
                    if nudge:
                        messages.append({"role": "system", "content": nudge})
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

    def _should_auto_finalize_catalog_only(
        self,
        *,
        tool_name: str,
        result_payload: dict[str, Any],
        tool_call_count: int,
    ) -> bool:
        if self.allowed_tools != {"features_catalog"}:
            return False
        if str(tool_name or "").strip() != "features_catalog":
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
        guide = f"First action guide: use the minimum useful step from allowed tools [{tools}]."
        if first_criterion:
            guide += f" Prioritize evidence for: {first_criterion}."
        guide += f"Call at least one tool from {tools} to gather evidence."
        guide += "After calling tools, return exactly one terminal JSON final_report."
        return guide

    def _maybe_auto_finalize_after_stall(self, transcript: list[dict[str, Any]]) -> str | None:
        if self.allowed_tools == {"features_catalog"}:
            return build_catalog_only_final_report(
                transcript=transcript,
                success_criteria=self.success_criteria,
                required_output_facts=self.required_output_facts,
            )
        research = self._build_research_budget_salvage_report(transcript)
        if research is not None:
            return research
        backtests = self._build_backtests_budget_salvage_report(transcript)
        if backtests is not None:
            return backtests
        return build_generic_transcript_salvage_report(
            transcript=transcript,
            allowed_tools=self.allowed_tools,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
            slice_title=self.slice_title,
        )

    def _build_backtests_budget_salvage_report(self, transcript: list[dict[str, Any]]) -> str | None:
        return build_backtests_budget_salvage_report(
            transcript=transcript,
            allowed_tools=self.allowed_tools,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
        )

    def _build_research_budget_salvage_report(self, transcript: list[dict[str, Any]]) -> str | None:
        return build_research_budget_salvage_report(
            transcript=transcript,
            allowed_tools=self.allowed_tools,
            success_criteria=self.success_criteria,
            required_output_facts=self.required_output_facts,
        )

    async def _openai_tool_schemas(self) -> list[dict[str, Any]]:
        mcp_tools = await self.mcp_client.list_tools()
        result: list[dict[str, Any]] = []
        for tool in mcp_tools:
            name = str(tool.get("name", "") or "").strip()
            if name not in self.allowed_tools or name in self.safe_exclude_tools:
                continue
            schema = tool.get("inputSchema") or tool.get("input_schema") or {"type": "object", "properties": {}}
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(tool.get("description", "") or tool.get("title", "") or name)[:900],
                        "parameters": schema,
                    },
                }
            )
        return result

    def _fallback_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Fallback schema for {name}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in sorted(self.allowed_tools)
            if name not in self.safe_exclude_tools
        ]

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
            self.incident_store.record(
                summary=f"Direct dev_space1 contract/infra issue: {tool_name}",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "raw_payload_excerpt": json.dumps(payload, ensure_ascii=False)[:4000],
                    "runtime": "direct_lmstudio_mcp",
                },
                source="direct_runtime",
            )
        return {"ok": True, "payload": payload}

    async def _chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._chat_sync, messages=messages, tools=tools, timeout=timeout),
                timeout=max(1, timeout + 2),
            )
        except asyncio.TimeoutError:
            return {"error": f"lmstudio_chat_timeout_after_{timeout}s"}

    def _chat_sync(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], timeout: int) -> dict[str, Any]:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": self.adapter.temperature,
            "max_tokens": self.adapter.max_tokens,
        }
        if self.adapter.model:
            body["model"] = self.adapter.model
        if self.adapter.reasoning_effort and self.adapter.reasoning_effort not in ("none", "off"):
            body["reasoning_effort"] = self.adapter.reasoning_effort
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self.adapter.extra_body)
        try:
            parsed = urlparse(self.adapter.base_url)
            conn = HTTPConnection(parsed.hostname, parsed.port or 1234, timeout=timeout)
            headers = {"Content-Type": "application/json"}
            if self.adapter.api_key:
                headers["Authorization"] = f"Bearer {self.adapter.api_key}"
            conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8")
            conn.close()
            if resp.status != 200:
                return {"error": f"HTTP {resp.status}: {raw[:800]}"}
            return json.loads(raw)
        except Exception as exc:
            return {"error": str(exc)}


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


__all__ = ["LmStudioToolLoop", "LmStudioToolLoopResult"]
