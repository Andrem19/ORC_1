"""
Provider-neutral direct slice executor.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.adapters.base import BaseAdapter
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.adapters.qwen_worker_cli import QwenWorkerCli
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice, WorkerAction
from app.execution_parsing import StructuredOutputError, parse_worker_action_output
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.backtests_protocol import augment_allowed_tools_for_backtests
from app.services.direct_execution.invocation import invoke_adapter_with_retries
from app.services.direct_execution.guardrails import synthesize_transcript_evidence_refs
from app.services.direct_execution.lmstudio_finalization import build_generic_transcript_salvage_report
from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from app.services.direct_execution.handle_hygiene import DURABLE_HANDLE_FIELDS
from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig
from app.services.direct_execution.prompt import build_direct_slice_prompt
from app.services.direct_execution.runtime_profiles import resolve_runtime_slice_metadata_with_prerequisites
from app.services.direct_execution.semantic_progress import build_watchdog_checkpoint
from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.direct_execution.transcript_facts import derive_facts_from_transcript


@dataclass(frozen=True)
class DirectExecutionResult:
    action: WorkerAction | None
    artifact_path: str
    raw_output: str
    error: str = ""
    provider: str = ""
    duration_ms: int = 0
    tool_call_count: int = 0
    expensive_tool_call_count: int = 0
    parse_retry_count: int = 0
    fallback_provider_index: int = 0  # 0=primary, 1=fallback_1, 2=fallback_2
    transcript: list[dict[str, Any]] = field(default_factory=list)
    acceptance_result: dict[str, Any] = field(default_factory=dict)


class DirectSliceExecutor:
    def __init__(
        self,
        *,
        adapter: BaseAdapter,
        artifact_store: ExecutionArtifactStore,
        incident_store: LocalIncidentStore,
        direct_config: Any,
        worker_system_prompt: str = "",
        invoker: Any = invoke_adapter_with_retries,
        provider_name: str = "",
        catalog_snapshot: McpCatalogSnapshot | None = None,
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.incident_store = incident_store
        self.direct_config = direct_config
        self.worker_system_prompt = worker_system_prompt
        self.invoker = invoker
        self.provider_name = str(provider_name or "").strip()
        self.catalog_snapshot = catalog_snapshot

    async def _run_qwen_primary_preflight(self, *, required_tools: list[str]) -> dict[str, Any]:
        if not isinstance(self.adapter, QwenWorkerCli):
            return {}
        max_attempts = max(1, int(getattr(self.direct_config, "qwen_primary_preflight_max_attempts", 3) or 1))
        retry_delay = max(0.0, float(getattr(self.direct_config, "qwen_primary_preflight_retry_delay_seconds", 2.0) or 0.0))
        probe_timeout = int(getattr(self.direct_config, "qwen_preflight_timeout_seconds", 60) or 60)
        last_registry: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            try:
                registry = self.adapter.preflight_tool_registry(
                    required_tools=required_tools,
                    timeout=probe_timeout,
                )
            except TypeError:
                registry = self.adapter.preflight_tool_registry(required_tools=required_tools)
            last_registry = dict(registry or {})
            last_registry["attempts"] = attempt
            if not last_registry.get("missing_required_tools"):
                last_registry["primary_preflight_unavailable"] = False
                return last_registry
            if attempt < max_attempts and retry_delay > 0:
                await asyncio.sleep(retry_delay)
        last_registry["primary_preflight_unavailable"] = bool(last_registry.get("missing_required_tools"))
        return last_registry

    async def execute(
        self,
        *,
        plan_id: str,
        slice_obj: PlanSlice,
        baseline_bootstrap: dict[str, Any],
        known_facts: dict[str, Any],
        required_output_facts: list[str],
        recent_turn_summaries: list[str],
        checkpoint_summary: str,
        extra_prompt_section: str = "",
        on_tool_progress: Any = None,
        attempt_label: str = "direct",
        adapter_invoke_kwargs: dict[str, Any] | None = None,
    ) -> DirectExecutionResult:
        attempt_id = f"{attempt_label}_{slice_obj.slice_id}_{slice_obj.turn_count + 1}"
        provider = _resolve_provider_name(
            adapter=self.adapter,
            explicit_provider=self.provider_name,
            configured_provider=str(getattr(self.direct_config, "provider", "") or ""),
        )
        allowed_tools = augment_allowed_tools_for_backtests(
            allowed_tools=_allowed_tools(slice_obj.allowed_tools),
            catalog_snapshot=self.catalog_snapshot,
            runtime_profile=str(slice_obj.runtime_profile or ""),
            title=str(slice_obj.title or ""),
            objective=str(slice_obj.objective or ""),
            success_criteria=list(slice_obj.success_criteria or []),
            policy_tags=list(slice_obj.policy_tags or []),
        )
        qwen_primary_registry: dict[str, Any] = {}
        extra_prompt_sections: list[str] = []
        if (
            isinstance(self.adapter, QwenWorkerCli)
            and bool(getattr(self.direct_config, "qwen_primary_preflight_enabled", True))
            and bool(getattr(self.direct_config, "qwen_tool_registry_preflight", True))
            and allowed_tools
        ):
            qwen_primary_registry = await self._run_qwen_primary_preflight(
                required_tools=sorted(allowed_tools),
            )
            if qwen_primary_registry.get("primary_preflight_unavailable"):
                artifact_path = self.artifact_store.save_direct_attempt(
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    payload={
                        "attempt_id": attempt_id,
                        "provider": provider,
                        "status": "failed",
                        "error": "qwen_mcp_tools_unavailable",
                        "allowed_tools": sorted(allowed_tools),
                        "mcp_catalog_hash": getattr(self.catalog_snapshot, "schema_hash", ""),
                        "qwen_preflight": qwen_primary_registry,
                    },
                )
                self.incident_store.record(
                    summary="Qwen primary preflight failed to expose required MCP tools",
                    metadata={
                        "plan_id": plan_id,
                        "slice_id": slice_obj.slice_id,
                        "provider": provider,
                        "missing_required_tools": list(qwen_primary_registry.get("missing_required_tools") or []),
                        "attempts": int(qwen_primary_registry.get("attempts") or 0),
                        "runtime": "direct_execution",
                    },
                    source="direct_runtime",
                    severity="medium",
                )
                return DirectExecutionResult(
                    action=None,
                    artifact_path=str(artifact_path),
                    raw_output="",
                    error="qwen_mcp_tools_unavailable",
                    provider=provider,
                    duration_ms=0,
                    tool_call_count=0,
                    expensive_tool_call_count=0,
                    transcript=[],
                )
            registry_section = _build_qwen_registry_prompt_section(registry=qwen_primary_registry)
            if registry_section:
                extra_prompt_sections.append(registry_section)
        if not allowed_tools:
            artifact_path = self.artifact_store.save_direct_attempt(
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                payload={
                    "attempt_id": attempt_id,
                    "provider": provider,
                    "status": "failed",
                    "error": "direct_empty_allowed_tools",
                    "allowed_tools": [],
                    "mcp_catalog_hash": getattr(self.catalog_snapshot, "schema_hash", ""),
                },
            )
            self.incident_store.record(
                summary="Direct slice has empty allowed_tools and cannot execute",
                metadata={
                    "plan_id": plan_id,
                    "slice_id": slice_obj.slice_id,
                    "provider": provider,
                    "runtime": "direct_execution",
                },
                source="direct_runtime",
                severity="high",
            )
            return DirectExecutionResult(
                action=None,
                artifact_path=str(artifact_path),
                raw_output="",
                error="direct_empty_allowed_tools",
                provider=provider,
                duration_ms=0,
                tool_call_count=0,
                expensive_tool_call_count=0,
                transcript=[],
            )
        (
            effective_runtime_profile,
            effective_required_output_facts,
            effective_required_prerequisite_facts,
            effective_finalization_mode,
        ) = resolve_runtime_slice_metadata_with_prerequisites(
            runtime_profile=str(slice_obj.runtime_profile or ""),
            required_output_facts=list(required_output_facts),
            required_prerequisite_facts=list(slice_obj.required_prerequisite_facts or []),
            finalization_mode=str(slice_obj.finalization_mode or ""),
            allowed_tools=sorted(allowed_tools),
            catalog_snapshot=self.catalog_snapshot,
            title=str(slice_obj.title or ""),
            objective=str(slice_obj.objective or ""),
            success_criteria=list(slice_obj.success_criteria or []),
            policy_tags=list(slice_obj.policy_tags or []),
        )
        slice_obj.allowed_tools = sorted(allowed_tools)
        slice_obj.runtime_profile = effective_runtime_profile
        slice_obj.required_prerequisite_facts = list(effective_required_prerequisite_facts)
        slice_obj.required_output_facts = list(effective_required_output_facts)
        slice_obj.finalization_mode = effective_finalization_mode
        prompt = build_direct_slice_prompt(
            plan_id=plan_id,
            slice_payload=asdict(slice_obj),
            baseline_bootstrap=baseline_bootstrap,
            known_facts=known_facts,
            recent_turn_summaries=recent_turn_summaries,
            checkpoint_summary=checkpoint_summary,
            allowed_tools=sorted(allowed_tools),
            max_tool_calls=min(int(slice_obj.max_tool_calls or 1), int(self.direct_config.max_tool_calls_per_slice or 1)),
            max_expensive_tool_calls=min(int(slice_obj.max_expensive_calls or 0), int(self.direct_config.max_expensive_tool_calls_per_slice or 0)),
            worker_system_prompt=self.worker_system_prompt,
            required_prerequisite_facts=effective_required_prerequisite_facts,
            required_output_facts=effective_required_output_facts,
            provider=provider,
            catalog_snapshot=self.catalog_snapshot,
        )
        if extra_prompt_section:
            extra_prompt_sections.append(extra_prompt_section)
        if extra_prompt_sections:
            prompt = prompt + "\n\n" + "\n\n".join(section for section in extra_prompt_sections if str(section).strip()) + "\n"
        transcript: list[dict[str, Any]] = []
        tool_call_count = 0
        expensive_count = 0
        invoke_kwargs = dict(adapter_invoke_kwargs or {})
        in_flight_path = self.artifact_store.save_direct_attempt(
            plan_id=plan_id,
            slice_id=slice_obj.slice_id,
            payload={
                "attempt_id": attempt_id,
                "provider": provider,
                "status": "in_progress",
                "prompt": prompt,
                "allowed_tools": sorted(allowed_tools),
                "required_output_facts": list(effective_required_output_facts),
                "tool_call_count": 0,
                "expensive_tool_call_count": 0,
                "mcp_catalog_hash": getattr(self.catalog_snapshot, "schema_hash", ""),
                "response": {},
                "transcript": [],
            },
        )
        if provider in ("lmstudio", "minimax") and isinstance(self.adapter, LmStudioWorkerApi):
            loop = LmStudioToolLoop(
                adapter=self.adapter,
                mcp_client=DirectMcpClient(
                    DirectMcpConfig(
                        endpoint_url=str(self.direct_config.mcp_endpoint_url),
                        auth_mode=str(self.direct_config.mcp_auth_mode),
                        token_env_var=str(self.direct_config.mcp_token_env_var),
                        connect_timeout_seconds=float(self.direct_config.connect_timeout_seconds),
                        read_timeout_seconds=float(self.direct_config.read_timeout_seconds),
                        retry_budget=int(self.direct_config.retry_budget),
                    )
                ),
                incident_store=self.incident_store,
                catalog_snapshot=self.catalog_snapshot,
                allowed_tools=allowed_tools,
                slice_title=slice_obj.title,
                slice_objective=slice_obj.objective,
                success_criteria=list(slice_obj.success_criteria or []),
                policy_tags=list(slice_obj.policy_tags or []),
                required_prerequisite_facts=list(effective_required_prerequisite_facts),
                required_output_facts=list(effective_required_output_facts),
                runtime_profile=effective_runtime_profile,
                finalization_mode=effective_finalization_mode,
                max_tool_calls=min(int(slice_obj.max_tool_calls or 1), int(self.direct_config.max_tool_calls_per_slice or 1)),
                max_expensive_tool_calls=min(int(slice_obj.max_expensive_calls or 0), int(self.direct_config.max_expensive_tool_calls_per_slice or 0)),
                safe_exclude_tools=set(self.direct_config.safe_exclude_tools or []),
                first_action_timeout_seconds=int(getattr(self.direct_config, "first_action_timeout_seconds", 75) or 75),
                stalled_action_timeout_seconds=int(getattr(self.direct_config, "stalled_action_timeout_seconds", 60) or 60),
                zero_tool_retries=int(getattr(self.direct_config, "lmstudio_zero_tool_retries", 2) or 0),
                first_turn_tool_choice=str(getattr(self.direct_config, "lmstudio_first_turn_tool_choice", "required") or "auto"),
                baseline_bootstrap=baseline_bootstrap,
                known_facts=known_facts,
            )
            progress = {"ts": time.monotonic(), "count": 0, "last_kind": "started"}
            progress_metrics = {
                "tool_call_count": 0,
                "expensive_tool_call_count": 0,
                "transcript_len": 0,
                "last_tool_name": "",
            }
            current_transcript: list[dict[str, Any]] = []

            def _on_progress(kind: str, payload: dict[str, Any]) -> None:
                progress["ts"] = time.monotonic()
                progress["count"] = int(progress["count"]) + 1
                progress["last_kind"] = kind
                for field_name in ("tool_call_count", "expensive_tool_call_count", "transcript_len"):
                    if payload.get(field_name) is not None:
                        progress_metrics[field_name] = int(payload.get(field_name) or 0)
                if payload.get("tool_name") is not None:
                    progress_metrics["last_tool_name"] = str(payload.get("tool_name") or "")
                payload_transcript = payload.get("transcript")
                if isinstance(payload_transcript, list):
                    current_transcript.clear()
                    current_transcript.extend(item for item in payload_transcript if isinstance(item, dict))
                if kind == "tool_result" and on_tool_progress is not None:
                    try:
                        on_tool_progress(
                            tool_call_count=progress_metrics["tool_call_count"],
                            expensive_call_count=progress_metrics["expensive_tool_call_count"],
                        )
                    except Exception:
                        pass
                self.artifact_store.save_direct_attempt(
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    payload={
                        "attempt_id": attempt_id,
                        "provider": provider,
                        "status": "in_progress",
                        "prompt": prompt,
                        "allowed_tools": sorted(allowed_tools),
                        "required_output_facts": list(effective_required_output_facts),
                        "tool_call_count": progress_metrics["tool_call_count"],
                        "expensive_tool_call_count": progress_metrics["expensive_tool_call_count"],
                        "mcp_catalog_hash": getattr(self.catalog_snapshot, "schema_hash", ""),
                        "response": {"last_progress_kind": kind},
                        "transcript": [{"kind": "heartbeat", "transcript_len": progress_metrics["transcript_len"]}],
                        "transcript_len": progress_metrics["transcript_len"],
                        "last_tool_name": progress_metrics["last_tool_name"],
                        "last_progress_kind": kind,
                        "heartbeat": {"kind": kind, "payload": payload},
                    },
                )

            loop_task = asyncio.create_task(
                loop.invoke(
                    prompt=prompt,
                    timeout_seconds=int(self.direct_config.timeout_seconds or 600),
                    plan_id=plan_id,
                    slice_id=slice_obj.slice_id,
                    on_progress=_on_progress,
                )
            )
            first_window = int(getattr(self.direct_config, "first_action_timeout_seconds", 75) or 75)
            stalled_window = int(getattr(self.direct_config, "stalled_action_timeout_seconds", 60) or 60)
            while True:
                window = first_window if int(progress["count"]) == 0 else stalled_window
                elapsed_since_progress = time.monotonic() - float(progress["ts"])
                remaining = window - elapsed_since_progress
                if remaining <= 0:
                    loop_task.cancel()
                    response = type("Resp", (), {})()
                    response.success = True
                    response.raw_output = build_watchdog_checkpoint(
                        summary=(
                            f"Direct model stalled before producing the first action for slice '{slice_obj.title or slice_obj.slice_id}'."
                            if int(progress["count"]) == 0
                            else f"Direct model stalled without new progress for slice '{slice_obj.title or slice_obj.slice_id}'."
                        ),
                        reason_code="direct_model_stalled_before_first_action" if int(progress["count"]) == 0 else "direct_model_stalled_between_actions",
                    )
                    response.error = ""
                    response.timed_out = True
                    response.finish_reason = ""
                    response.metadata = {"stalled_after_progress_kind": progress["last_kind"]}
                    response.duration_seconds = float(window)
                    transcript = list(current_transcript)
                    tool_call_count = int(progress_metrics["tool_call_count"] or 0)
                    expensive_count = int(progress_metrics["expensive_tool_call_count"] or 0)
                    loop_result = None
                    break
                try:
                    loop_result = await asyncio.wait_for(asyncio.shield(loop_task), timeout=max(0.05, remaining))
                    break
                except asyncio.TimeoutError:
                    continue
            if loop_result is not None:
                response = loop_result.response
                transcript = loop_result.transcript
                tool_call_count = loop_result.tool_call_count
                expensive_count = loop_result.expensive_tool_call_count
        else:
            response = await self.invoker(
                adapter=self.adapter,
                prompt=prompt,
                timeout_seconds=int(self.direct_config.timeout_seconds or 600),
                first_action_timeout_seconds=getattr(self.direct_config, "first_action_timeout_seconds", None),
                stalled_action_timeout_seconds=getattr(self.direct_config, "stalled_action_timeout_seconds", None),
                max_attempts=max(1, int(self.direct_config.max_attempts_per_slice or 1)),
                base_backoff_seconds=0.25,
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                exclude_tools=list(getattr(self.direct_config, "safe_exclude_tools", []) or []),
                **invoke_kwargs,
            )
            # Extract tool call count from CLI adapter metadata (set by
            # claude_worker_cli / qwen_worker_cli stream-json parsing).
            meta = response.metadata or {}
            tool_call_count = int(meta.get("tool_call_count", 0) or 0)
            if tool_call_count == 0 and meta.get("raw_stdout"):
                from app.services.direct_execution.stream_tool_counter import count_tool_calls_from_stream_json

                counted = count_tool_calls_from_stream_json(
                    str(meta["raw_stdout"]),
                    provider,
                    allowed_tool_names=self.catalog_snapshot.tool_name_set() if self.catalog_snapshot is not None else None,
                )
                tool_call_count = counted.tool_call_count
        artifact_path = self.artifact_store.save_direct_attempt(
            plan_id=plan_id,
            slice_id=slice_obj.slice_id,
            payload={
                "attempt_id": attempt_id,
                "provider": provider,
                "status": "completed",
                "prompt": prompt,
                "response": {
                    "success": response.success,
                    "raw_output": response.raw_output,
                    "error": response.error,
                    "timed_out": response.timed_out,
                    "finish_reason": response.finish_reason,
                    "metadata": dict(response.metadata or {}),
                    "duration_seconds": response.duration_seconds,
                },
                "transcript": transcript,
                "allowed_tools": sorted(allowed_tools),
                "tool_call_count": tool_call_count,
                "expensive_tool_call_count": expensive_count,
                "mcp_catalog_hash": getattr(self.catalog_snapshot, "schema_hash", ""),
                "in_flight_artifact_path": str(in_flight_path),
            },
        )
        if not response.success:
            self._record_direct_incident(plan_id=plan_id, slice_obj=slice_obj, error=response.error or response.finish_reason)
            return DirectExecutionResult(
                action=None,
                artifact_path=str(artifact_path),
                raw_output=response.raw_output,
                error=response.error or response.finish_reason or "direct_adapter_failed",
                provider=provider,
                duration_ms=int((response.duration_seconds or 0.0) * 1000),
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_count,
                transcript=list(transcript),
            )
        try:
            action = parse_worker_action_output(response.raw_output, allowlist=allowed_tools, provider=provider)
        except StructuredOutputError as exc:
            self._record_direct_incident(plan_id=plan_id, slice_obj=slice_obj, error=f"direct_output_parse_failed:{exc}")
            return DirectExecutionResult(
                action=None,
                artifact_path=str(artifact_path),
                raw_output=response.raw_output,
                error=f"direct_output_parse_failed:{exc}",
                provider=provider,
                duration_ms=int((response.duration_seconds or 0.0) * 1000),
                tool_call_count=tool_call_count,
                expensive_tool_call_count=expensive_count,
                parse_retry_count=1,
                transcript=list(transcript),
            )
        if action.action_type == "tool_call":
            action = WorkerAction(
                action_id=action.action_id,
                action_type="checkpoint",
                status="blocked",
                summary=f"Direct model returned forbidden terminal tool_call for {action.tool}.",
                facts={"direct.invalid_terminal_tool_call": action.tool, "direct.invalid_arguments": str(action.arguments or {})},
                reason="Direct runtime accepts only terminal final_report/checkpoint/abort results.",
            )
        result_error = ""
        if (
            provider == "qwen_cli"
            and action.action_type == "checkpoint"
            and _detect_qwen_mcp_tools_unavailable(
                raw_output=response.raw_output,
                tool_call_count=tool_call_count,
            )
        ):
            result_error = "qwen_mcp_tools_unavailable"
            action.reason_code = action.reason_code or "qwen_mcp_tools_unavailable"
            action.facts.setdefault("direct.qwen_mcp_tools_unavailable", True)
        transcript_facts = derive_facts_from_transcript(
            transcript,
            runtime_profile=str(slice_obj.runtime_profile or ""),
        )
        for key, value in transcript_facts.items():
            action.facts.setdefault(key, value)
        # Transcript-derived handle facts take precedence when the model
        # emitted a transcript reference (e.g. "transcript:2:research_project")
        # instead of a real project_id / job_id / etc.
        _prefer_transcript_handle_facts(action.facts, transcript_facts)
        metadata_facts = _derive_facts_from_adapter_metadata(
            metadata=response.metadata or {},
            allowed_tools=allowed_tools,
        )
        for key, value in metadata_facts.items():
            action.facts.setdefault(key, value)
        # Auto-synthesize evidence_refs when the model returned none but the
        # transcript contains successful tool results.  Models like MiniMax
        # sometimes make 10+ tool calls yet report evidence_refs=[].
        # The synthesized refs are transcript:N:tool_name format — real,
        # verifiable references into the transcript, not fabricated ids.
        if (
            action.action_type == "final_report"
            and not action.evidence_refs
            and transcript
            and int(action.facts.get("direct.successful_tool_count") or 0) >= 1
        ):
            synthesized = synthesize_transcript_evidence_refs(transcript)
            if synthesized:
                action.evidence_refs = synthesized
        # Transcript synthesis: if LMStudio produced a non-terminal result
        # (checkpoint/abort) but the transcript has enough evidence, attempt
        # to synthesize a valid final_report.
        # IMPORTANT: Skip synthesis when the model explicitly returned
        # status="blocked". A blocked checkpoint means the model recognised a
        # genuine obstacle (missing prerequisite data, no runs to analyse, etc.)
        # and is honestly reporting that it cannot proceed.  Synthesising a
        # salvage final_report in that case stamps auto-salvage facts which the
        # quality gate rejects unconditionally, creating a guaranteed failure
        # loop: checkpoint → salvage → gate reject → fallback (same obstacle).
        # Respecting the blocked status lets the orchestrator route the slice
        # through its normal prerequisite-block detection instead.
        is_explicit_block = (
            action.action_type == "checkpoint"
            and str(action.status or "").strip() == "blocked"
        )
        if (
            action.action_type in ("checkpoint", "abort")
            and not is_explicit_block
            and transcript
            and provider in ("lmstudio", "minimax")
        ):
            synthesis = build_generic_transcript_salvage_report(
                transcript=transcript,
                success_criteria=list(slice_obj.success_criteria or []),
                required_output_facts=list(effective_required_output_facts),
                runtime_profile=str(slice_obj.runtime_profile or ""),
                slice_title=slice_obj.title,
            )
            if synthesis is not None:
                try:
                    action = parse_worker_action_output(synthesis, allowlist=allowed_tools, provider=provider)
                    for key, value in transcript_facts.items():
                        action.facts.setdefault(key, value)
                    _prefer_transcript_handle_facts(action.facts, transcript_facts)
                except StructuredOutputError:
                    pass
        return DirectExecutionResult(
            action=action,
            artifact_path=str(artifact_path),
            raw_output=response.raw_output,
            error=result_error,
            provider=provider,
            duration_ms=int((response.duration_seconds or 0.0) * 1000),
            tool_call_count=tool_call_count,
            expensive_tool_call_count=expensive_count,
            transcript=list(transcript),
        )

    def _record_direct_incident(self, *, plan_id: str, slice_obj: PlanSlice, error: str) -> None:
        self.incident_store.record(
            summary="Direct slice execution failed",
            metadata={
                "plan_id": plan_id,
                "slice_id": slice_obj.slice_id,
                "error": error,
                "provider": _resolve_provider_name(
                    adapter=self.adapter,
                    explicit_provider=self.provider_name,
                    configured_provider=str(getattr(self.direct_config, "provider", "") or ""),
                ),
                "runtime": "direct_execution",
            },
            source="direct_runtime",
        )


def _allowed_tools(slice_tools: list[str]) -> set[str]:
    return {str(item).strip() for item in slice_tools if str(item).strip()}


def _prefer_transcript_handle_facts(
    action_facts: dict[str, Any],
    transcript_facts: dict[str, Any],
) -> None:
    """Override model-emitted handle values with transcript-derived ones when
    the model emitted a transcript reference (e.g. ``transcript:2:tool_name``)
    instead of a real handle value.
    """
    for field in DURABLE_HANDLE_FIELDS:
        transcript_val = transcript_facts.get(field)
        model_val = action_facts.get(field)
        if (
            transcript_val
            and model_val
            and str(model_val).startswith("transcript:")
            and not str(transcript_val).startswith("transcript:")
        ):
            action_facts[field] = transcript_val
            if field == "project_id":
                action_facts["research.project_id"] = transcript_val


def _derive_facts_from_adapter_metadata(*, metadata: dict[str, Any], allowed_tools: set[str]) -> dict[str, Any]:
    if not isinstance(metadata, dict) or not allowed_tools:
        return {}
    raw_tool_names = metadata.get("tool_names")
    if not isinstance(raw_tool_names, list):
        return {}
    successful_tool_names: list[str] = []
    for item in raw_tool_names:
        tool_name = str(item or "").strip()
        if not tool_name or tool_name not in allowed_tools:
            continue
        if tool_name not in successful_tool_names:
            successful_tool_names.append(tool_name)
    if not successful_tool_names:
        return {}
    facts: dict[str, Any] = {
        "direct.successful_tool_names": successful_tool_names[:20],
        "direct.successful_tool_count": int(metadata.get("tool_call_count") or len(successful_tool_names) or 0),
    }
    return facts


def _resolve_provider_name(*, adapter: BaseAdapter, explicit_provider: str, configured_provider: str) -> str:
    if explicit_provider:
        return _maybe_relabel_glm_provider(explicit_provider)
    adapter_name = str(adapter.name() or "").strip().lower()
    if adapter_name == "qwen_worker_cli":
        return "qwen_cli"
    if adapter_name == "claude_worker_cli":
        return _maybe_relabel_glm_provider("claude_cli")
    if adapter_name == "lmstudio_worker_api":
        return "lmstudio"
    return _maybe_relabel_glm_provider(str(configured_provider or adapter.name()).strip())


def _maybe_relabel_glm_provider(name: str) -> str:
    """Label the z.ai/glm backend honestly instead of claiming to be `claude_cli`."""
    normalized = str(name or "").strip()
    if normalized != "claude_cli":
        return normalized
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "").strip().lower()
    if "z.ai" in base_url:
        return "glm_cli"
    return normalized


_QWEN_MISSING_TOOL_PATTERN = re.compile(
    r"Tool\s+['\"`]?([A-Za-z0-9_\-]+)['\"`]?\s+not\s+found\s+in\s+registry",
    re.IGNORECASE,
)


def _detect_qwen_mcp_tools_unavailable(*, raw_output: str, tool_call_count: int) -> bool:
    """Detect the qwen-specific signal that MCP tools were not exposed at runtime."""
    text = str(raw_output or "")
    if not text:
        return False
    if tool_call_count > 2:
        return False
    return bool(_QWEN_MISSING_TOOL_PATTERN.search(text))


def _build_qwen_registry_prompt_section(*, registry: dict[str, Any]) -> str:
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


__all__ = ["DirectExecutionResult", "DirectSliceExecutor"]
