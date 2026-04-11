"""
Provider-neutral direct slice executor.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any

from app.adapters.base import BaseAdapter
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice, WorkerAction
from app.execution_parsing import StructuredOutputError, parse_worker_action_output
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.invocation import invoke_adapter_with_retries
from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop
from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig
from app.services.direct_execution.prompt import build_direct_slice_prompt
from app.services.direct_execution.semantic_progress import build_watchdog_checkpoint
from app.services.direct_execution.tool_catalog import DEFAULT_DEV_SPACE1_TOOLS
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
    ) -> None:
        self.adapter = adapter
        self.artifact_store = artifact_store
        self.incident_store = incident_store
        self.direct_config = direct_config
        self.worker_system_prompt = worker_system_prompt
        self.invoker = invoker
        self.provider_name = str(provider_name or "").strip()

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
        allowed_tools = _allowed_tools(slice_obj.allowed_tools)
        attempt_id = f"{attempt_label}_{slice_obj.slice_id}_{slice_obj.turn_count + 1}"
        provider = _resolve_provider_name(
            adapter=self.adapter,
            explicit_provider=self.provider_name,
            configured_provider=str(getattr(self.direct_config, "provider", "") or ""),
        )
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
            required_output_facts=required_output_facts,
            provider=provider,
        )
        if extra_prompt_section:
            prompt = prompt + "\n\n" + extra_prompt_section + "\n"
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
                "required_output_facts": list(required_output_facts),
                "tool_call_count": 0,
                "expensive_tool_call_count": 0,
                "response": {},
                "transcript": [],
            },
        )
        if provider == "lmstudio" and isinstance(self.adapter, LmStudioWorkerApi):
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
                allowed_tools=allowed_tools,
                slice_title=slice_obj.title,
                success_criteria=list(slice_obj.success_criteria or []),
                required_output_facts=list(required_output_facts),
                max_tool_calls=min(int(slice_obj.max_tool_calls or 1), int(self.direct_config.max_tool_calls_per_slice or 1)),
                max_expensive_tool_calls=min(int(slice_obj.max_expensive_calls or 0), int(self.direct_config.max_expensive_tool_calls_per_slice or 0)),
                safe_exclude_tools=set(self.direct_config.safe_exclude_tools or []),
                first_action_timeout_seconds=int(getattr(self.direct_config, "first_action_timeout_seconds", 45) or 45),
                stalled_action_timeout_seconds=int(getattr(self.direct_config, "stalled_action_timeout_seconds", 60) or 60),
            )
            progress = {"ts": time.monotonic(), "count": 0, "last_kind": "started"}

            def _on_progress(kind: str, payload: dict[str, Any]) -> None:
                progress["ts"] = time.monotonic()
                progress["count"] = int(progress["count"]) + 1
                progress["last_kind"] = kind
                if kind == "tool_result" and on_tool_progress is not None:
                    try:
                        on_tool_progress(
                            tool_call_count=payload.get("tool_call_count", 0),
                            expensive_call_count=payload.get("expensive_tool_call_count", 0),
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
                        "required_output_facts": list(required_output_facts),
                        "tool_call_count": 0,
                        "expensive_tool_call_count": 0,
                        "response": {},
                        "transcript": [],
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
            first_window = int(getattr(self.direct_config, "first_action_timeout_seconds", 45) or 45)
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
                    transcript = []
                    tool_call_count = 0
                    expensive_count = 0
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
                max_attempts=max(1, int(self.direct_config.max_attempts_per_slice or 1)),
                base_backoff_seconds=0.25,
                plan_id=plan_id,
                slice_id=slice_obj.slice_id,
                exclude_tools=list(getattr(self.direct_config, "safe_exclude_tools", []) or []),
                **invoke_kwargs,
            )
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
            )
        try:
            action = parse_worker_action_output(response.raw_output, allowlist=allowed_tools)
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
        transcript_facts = derive_facts_from_transcript(transcript)
        for key, value in transcript_facts.items():
            action.facts.setdefault(key, value)
        return DirectExecutionResult(
            action=action,
            artifact_path=str(artifact_path),
            raw_output=response.raw_output,
            provider=provider,
            duration_ms=int((response.duration_seconds or 0.0) * 1000),
            tool_call_count=tool_call_count,
            expensive_tool_call_count=expensive_count,
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
    allowed = {str(item).strip() for item in slice_tools if str(item).strip()}
    return allowed or set(DEFAULT_DEV_SPACE1_TOOLS)


def _resolve_provider_name(*, adapter: BaseAdapter, explicit_provider: str, configured_provider: str) -> str:
    if explicit_provider:
        return explicit_provider
    adapter_name = str(adapter.name() or "").strip().lower()
    if adapter_name == "qwen_worker_cli":
        return "qwen_cli"
    if adapter_name == "claude_worker_cli":
        return "claude_cli"
    if adapter_name == "lmstudio_worker_api":
        return "lmstudio"
    return str(configured_provider or adapter.name()).strip()


__all__ = ["DirectExecutionResult", "DirectSliceExecutor"]
