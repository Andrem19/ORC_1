"""
Planner service — wraps the planner adapter with prompt building and output parsing.

Supports both synchronous consult() and async start_consultation()/check_consultation().
If the adapter doesn't support async mode (start/check), falls back to synchronous.
"""

from __future__ import annotations

import json
import logging
import time as _time
from typing import TYPE_CHECKING, Any

from app.adapters.base import BaseAdapter, ProcessHandle
from app.models import (
    OrchestratorState,
    PlannerDecision,
    PlannerOutput,
    TaskResult,
)
from app.planner_json_schema import build_plan_json_schema
from app.planner_runtime import PlannerRunSnapshot
from app.planner_structured_output import extract_planner_structured_output
from app.prompts import build_planner_prompt
from app.result_parser import parse_plan_output, parse_planner_output

logger = logging.getLogger("orchestrator.planner_service")

if TYPE_CHECKING:
    from app.plan_models import ResearchPlan, TaskReport
    from app.plan_validation import PlanRepairRequest


class PlannerService:
    """High-level service for interacting with the planner model."""

    def __init__(self, adapter: BaseAdapter, timeout: int = 180) -> None:
        self.adapter = adapter
        self.timeout = timeout
        self._active_handle: ProcessHandle | None = None
        self._supports_async: bool | None = None
        self._sync_result: PlannerOutput | None = None
        self._last_plan_raw_output: str = ""
        self._plan_request_type: str = "create"
        self._plan_request_version: int = 0
        self._plan_request_attempt: int = 0
        self._plan_request_payload: dict[str, Any] | None = None
        self._plan_runtime: PlannerRunSnapshot | None = None
        self._last_completed_plan_runtime: PlannerRunSnapshot | None = None
        self._plan_timeout_retry_count: int = 0
        self._plan_transport_retry_count: int = 0

    def _check_async_support(self) -> bool:
        """Check if the adapter supports async start/check."""
        if self._supports_async is not None:
            return self._supports_async
        try:
            import inspect
            base_start = BaseAdapter.start
            adapter_start = self.adapter.__class__.start
            self._supports_async = adapter_start is not base_start
        except Exception:
            self._supports_async = False
        return self._supports_async

    # ---------------------------------------------------------------
    # Synchronous path (blocking)
    # ---------------------------------------------------------------

    def consult(
        self,
        state: OrchestratorState,
        new_results: list[TaskResult] | None = None,
        worker_ids: list[str] | None = None,
        research_context: str | None = None,
        mcp_problem_summary: str | None = None,
    ) -> PlannerOutput:
        """Call the planner with current state and get a decision (blocking)."""
        prompt = build_planner_prompt(
            state, new_results, worker_ids,
            research_context=research_context,
            mcp_problem_summary=mcp_problem_summary,
        )
        logger.info("Calling planner (cycle %d)", state.current_cycle)
        logger.debug("Planner prompt (%d chars):\n%s", len(prompt), prompt)

        response = self.adapter.invoke(prompt, timeout=self.timeout)

        if not response.success:
            logger.error("Planner call failed: %s", response.error[:200])
            return PlannerOutput(
                decision=PlannerDecision.WAIT,
                reason=f"Planner adapter error: {response.error[:100]}",
                check_after_seconds=60,
            )

        output = parse_planner_output(response.raw_output)
        logger.debug(
            "Planner raw output (%d chars):\n%s",
            len(response.raw_output), response.raw_output[:2000],
        )
        logger.info(
            "Planner decided: %s (reason: %s)",
            output.decision.value,
            output.reason[:100],
        )
        return output

    # ---------------------------------------------------------------
    # Async path (non-blocking)
    # ---------------------------------------------------------------

    def start_consultation(
        self,
        state: OrchestratorState,
        new_results: list[TaskResult] | None = None,
        worker_ids: list[str] | None = None,
        research_context: str | None = None,
        mcp_problem_summary: str | None = None,
    ) -> None:
        """Launch planner as a background process."""
        if not self._check_async_support():
            logger.info("Adapter doesn't support async mode, falling back to synchronous consult")
            self._sync_result = self.consult(
                state, new_results, worker_ids,
                research_context=research_context,
                mcp_problem_summary=mcp_problem_summary,
            )
            return

        prompt = build_planner_prompt(
            state, new_results, worker_ids,
            research_context=research_context,
            mcp_problem_summary=mcp_problem_summary,
        )
        logger.info("Starting planner consultation (cycle %d)", state.current_cycle)
        logger.debug("Planner prompt (%d chars):\n%s", len(prompt), prompt)

        self._active_handle = self.adapter.start(
            prompt,
            task_id=f"planner-cycle-{state.current_cycle}",
            worker_id="planner",
        )

    def check_consultation(self) -> tuple[PlannerOutput | None, bool]:
        """Non-blocking check on the running planner."""
        if self._sync_result is not None:
            result = self._sync_result
            self._sync_result = None
            return result, True

        if self._active_handle is None:
            logger.error("check_consultation called but no active planner handle")
            return PlannerOutput(
                decision=PlannerDecision.WAIT,
                reason="No active planner consultation",
                check_after_seconds=60,
            ), True

        handle = self._active_handle
        new_output, is_finished = self.adapter.check(handle)

        if new_output:
            logger.debug(
                "Planner: received %d chars (total: %d)",
                len(new_output), len(handle.partial_output),
            )

        if not is_finished:
            elapsed = _time.monotonic() - handle.started_at
            logger.info(
                "Planner still running (%.0fs elapsed, output_so_far=%d chars)",
                elapsed, len(handle.partial_output),
            )
            return None, False

        full_output = handle.partial_output
        self._active_handle = None

        proc = handle.process
        if proc is not None and proc.returncode is not None and proc.returncode != 0:
            logger.error("Planner process exited with code %d", proc.returncode)
            return PlannerOutput(
                decision=PlannerDecision.WAIT,
                reason=f"Planner process exited with code {proc.returncode}",
                check_after_seconds=60,
            ), True

        output = parse_planner_output(full_output)
        logger.debug(
            "Planner raw output (%d chars):\n%s",
            len(full_output), full_output[:2000],
        )
        logger.info(
            "Planner decided: %s (reason: %s)",
            output.decision.value,
            output.reason[:100],
        )
        return output, True

    @property
    def is_running(self) -> bool:
        return self._active_handle is not None or self._sync_result is not None

    # ---------------------------------------------------------------
    # Plan-mode methods
    # ---------------------------------------------------------------

    def start_plan_creation(
        self,
        goal: str,
        research_context: str | None = None,
        anti_patterns: list[dict] | None = None,
        cumulative_summary: str = "",
        worker_ids: list[str] | None = None,
        mcp_problem_summary: str | None = None,
        previous_plan_markdown: str | None = None,
        plan_version: int = 1,
        attempt_number: int = 1,
    ) -> None:
        """Launch planner to CREATE a new research plan."""
        from app.plan_prompts import build_plan_creation_prompt

        self._plan_timeout_retry_count = 0
        self._plan_transport_retry_count = 0
        self._plan_request_payload = {
            "request_type": "create",
            "goal": goal,
            "research_context": research_context,
            "anti_patterns": anti_patterns,
            "cumulative_summary": cumulative_summary,
            "worker_ids": worker_ids,
            "mcp_problem_summary": mcp_problem_summary,
            "previous_plan_markdown": previous_plan_markdown,
            "plan_version": plan_version,
            "attempt_number": attempt_number,
        }
        prompt = build_plan_creation_prompt(
            goal=goal,
            research_context=research_context,
            anti_patterns=anti_patterns,
            cumulative_summary=cumulative_summary,
            worker_ids=worker_ids,
            mcp_problem_summary=mcp_problem_summary,
            previous_plan_markdown=previous_plan_markdown,
        )
        logger.info(
            "Starting plan creation via planner (version=%d attempt=%d, %d chars)",
            plan_version, attempt_number, len(prompt),
        )
        self._launch_plan_request(
            prompt=prompt,
            request_type="create",
            request_version=plan_version,
            attempt_number=attempt_number,
            max_tasks=5,
            timeout_retry_count=0,
            transport_retry_count=0,
            task_id=f"plan-create-v{plan_version}-a{attempt_number}",
        )

    def start_plan_revision(
        self,
        goal: str,
        current_plan: "ResearchPlan",
        reports: list["TaskReport"],
        research_context: str | None = None,
        anti_patterns: list[dict] | None = None,
        worker_ids: list[str] | None = None,
        mcp_problem_summary: str | None = None,
    ) -> None:
        """Launch planner to REVISE the current plan based on worker reports."""
        from app.plan_prompts import build_plan_revision_prompt

        self._plan_timeout_retry_count = 0
        self._plan_transport_retry_count = 0
        self._plan_request_payload = {
            "request_type": "revision",
            "goal": goal,
            "current_plan": current_plan,
            "reports": reports,
            "research_context": research_context,
            "anti_patterns": anti_patterns,
            "worker_ids": worker_ids,
            "mcp_problem_summary": mcp_problem_summary,
        }
        prompt = build_plan_revision_prompt(
            goal=goal,
            current_plan=current_plan,
            reports=reports,
            research_context=research_context,
            anti_patterns=anti_patterns,
            worker_ids=worker_ids,
            mcp_problem_summary=mcp_problem_summary,
        )
        next_version = current_plan.version + 1
        logger.info(
            "Starting plan revision v%d → v%d (%d reports, %d chars)",
            current_plan.version, next_version, len(reports), len(prompt),
        )
        self._launch_plan_request(
            prompt=prompt,
            request_type="revision",
            request_version=next_version,
            attempt_number=1,
            max_tasks=None,
            timeout_retry_count=0,
            transport_retry_count=0,
            task_id=f"plan-revise-v{current_plan.version}",
        )

    def start_plan_repair(
        self,
        repair_request: "PlanRepairRequest",
        research_context: str | None = None,
        worker_ids: list[str] | None = None,
        mcp_problem_summary: str | None = None,
    ) -> None:
        """Launch planner to REPAIR an invalid create-plan attempt."""
        from app.plan_prompts import build_plan_repair_prompt

        self._plan_timeout_retry_count = 0
        self._plan_transport_retry_count = 0
        self._plan_request_payload = {
            "request_type": "repair",
            "repair_request": repair_request,
            "research_context": research_context,
            "worker_ids": worker_ids,
            "mcp_problem_summary": mcp_problem_summary,
        }
        prompt = build_plan_repair_prompt(
            repair_request=repair_request,
            research_context=research_context,
            worker_ids=worker_ids,
            mcp_problem_summary=mcp_problem_summary,
        )
        logger.info(
            "Starting plan repair via planner (version=%d attempt=%d, %d chars)",
            repair_request.plan_version,
            repair_request.attempt_number,
            len(prompt),
        )
        max_tasks = len(repair_request.invalid_plan_data.get("tasks", [])) or None
        self._launch_plan_request(
            prompt=prompt,
            request_type="repair",
            request_version=repair_request.plan_version,
            attempt_number=repair_request.attempt_number,
            max_tasks=max_tasks,
            timeout_retry_count=0,
            transport_retry_count=0,
            task_id=f"plan-repair-v{repair_request.plan_version}-a{repair_request.attempt_number}",
        )

    def restart_plan_request(self, *, reason: str = "timeout") -> bool:
        """Restart the current create/repair/revision request."""
        if not self._plan_request_payload:
            return False

        request = dict(self._plan_request_payload)
        request_type = request.pop("request_type", "")
        if reason == "transport_error":
            self._plan_transport_retry_count += 1
        else:
            self._plan_timeout_retry_count += 1

        if request_type == "create":
            from app.plan_prompts import build_plan_creation_prompt
            prompt = build_plan_creation_prompt(
                goal=request["goal"],
                research_context=request.get("research_context"),
                anti_patterns=request.get("anti_patterns"),
                cumulative_summary=request.get("cumulative_summary", ""),
                worker_ids=request.get("worker_ids"),
                mcp_problem_summary=request.get("mcp_problem_summary"),
                previous_plan_markdown=request.get("previous_plan_markdown"),
            )
            self._launch_plan_request(
                prompt=prompt,
                request_type="create",
                request_version=request["plan_version"],
                attempt_number=request["attempt_number"],
                max_tasks=5,
                timeout_retry_count=self._plan_timeout_retry_count,
                transport_retry_count=self._plan_transport_retry_count,
                task_id=f"plan-create-v{request['plan_version']}-a{request['attempt_number']}",
            )
            return True

        if request_type == "repair":
            from app.plan_prompts import build_plan_repair_prompt
            repair_request = request["repair_request"]
            prompt = build_plan_repair_prompt(
                repair_request=repair_request,
                research_context=request.get("research_context"),
                worker_ids=request.get("worker_ids"),
                mcp_problem_summary=request.get("mcp_problem_summary"),
            )
            max_tasks = len(repair_request.invalid_plan_data.get("tasks", [])) or None
            self._launch_plan_request(
                prompt=prompt,
                request_type="repair",
                request_version=repair_request.plan_version,
                attempt_number=repair_request.attempt_number,
                max_tasks=max_tasks,
                timeout_retry_count=self._plan_timeout_retry_count,
                transport_retry_count=self._plan_transport_retry_count,
                task_id=f"plan-repair-v{repair_request.plan_version}-a{repair_request.attempt_number}",
            )
            return True

        if request_type == "revision":
            from app.plan_prompts import build_plan_revision_prompt
            current_plan = request["current_plan"]
            prompt = build_plan_revision_prompt(
                goal=request["goal"],
                current_plan=current_plan,
                reports=request["reports"],
                research_context=request.get("research_context"),
                anti_patterns=request.get("anti_patterns"),
                worker_ids=request.get("worker_ids"),
                mcp_problem_summary=request.get("mcp_problem_summary"),
            )
            self._launch_plan_request(
                prompt=prompt,
                request_type="revision",
                request_version=current_plan.version + 1,
                attempt_number=1,
                max_tasks=None,
                timeout_retry_count=self._plan_timeout_retry_count,
                transport_retry_count=self._plan_transport_retry_count,
                task_id=f"plan-revise-v{current_plan.version}",
            )
            return True

        return False

    def check_plan_output(self) -> tuple[dict | None, bool]:
        """Non-blocking check on the running planner (plan-mode)."""
        if self._sync_result is not None:
            result = self._sync_result
            self._sync_result = None
            return None, True

        if self._active_handle is None:
            logger.error("check_plan_output called but no active planner handle")
            return {
                "plan_action": "continue",
                "reason": "No active planner consultation",
            }, True

        handle = self._active_handle
        new_output, is_finished = self.adapter.check(handle)
        self._refresh_plan_runtime_from_handle(handle)

        snapshot = self._plan_runtime
        if new_output:
            logger.debug(
                "Planner (plan): received %d chars (total: %d)",
                len(new_output),
                len(handle.partial_output),
            )

        if not is_finished:
            if snapshot is not None:
                stream_state = self._classify_plan_stream_state(snapshot)
                logger.info(
                    "Planner (plan) still running: type=%s version=%d attempt=%d elapsed=%.0fs state=%s text=%d chars stderr=%d chars events=%d structured=%dB delta=%dB timeout_retries=%d transport_retries=%d",
                    snapshot.request_type,
                    snapshot.request_version,
                    snapshot.attempt_number,
                    snapshot.elapsed_seconds,
                    stream_state,
                    len(handle.partial_output),
                    len(handle.partial_error_output),
                    snapshot.stream_event_count,
                    snapshot.structured_payload_bytes,
                    snapshot.structured_delta_bytes,
                    snapshot.timeout_retry_count,
                    getattr(snapshot, "transport_retry_count", 0),
                )
            return None, False

        extraction = extract_planner_structured_output(
            str(handle.metadata.get("raw_stdout", "")),
            rendered_text=handle.partial_output,
        )
        full_output = handle.partial_output
        self._active_handle = None

        proc = handle.process
        exit_code = proc.returncode if proc is not None else None
        if snapshot is not None:
            snapshot.finish(
                exit_code=exit_code,
                rendered_output=full_output,
                termination_reason="completed",
            )
            snapshot.structured_payload = extraction.structured_payload
            snapshot.structured_payload_source = extraction.structured_payload_source
            snapshot.structured_payload_bytes = extraction.structured_payload_bytes
            snapshot.structured_delta_bytes = extraction.structured_delta_bytes
            snapshot.transport_errors = list(extraction.transport_errors)
            self._last_completed_plan_runtime = snapshot

        if exit_code is not None and exit_code != 0:
            logger.error("Planner process exited with code %d", exit_code)
            return {
                "plan_action": "continue",
                "reason": f"Planner process exited with code {exit_code}",
            }, True

        canonical_output = full_output
        failure_class = "invalid_content"
        parse_status = "parsed_from_rendered_text"
        if extraction.structured_payload is not None:
            canonical_output = json.dumps(extraction.structured_payload, ensure_ascii=False)
            parsed = parse_plan_output(canonical_output)
            if parsed.get("_parse_failed"):
                failure_class = "invalid_content"
                parse_status = "structured_payload_invalid_content"
            else:
                failure_class = "none"
                parse_status = f"parsed_from_{extraction.structured_payload_source}"
        else:
            parsed = parse_plan_output(full_output)
            if parsed.get("_parse_failed"):
                if extraction.saw_structured_output_activity:
                    failure_class = "transport_error"
                    parse_status = "structured_output_transport_error"
                else:
                    failure_class = "parse_error"
                    parse_status = "rendered_text_parse_error"
            else:
                failure_class = "none"
                parse_status = "parsed_from_rendered_text"

        self._last_plan_raw_output = canonical_output
        parsed["_failure_class"] = failure_class
        parsed["_request_type"] = self._plan_request_type
        parsed["_request_version"] = self._plan_request_version
        parsed["_attempt_number"] = self._plan_request_attempt
        parsed["_transport_errors"] = list(extraction.transport_errors)
        parsed["_structured_payload_source"] = extraction.structured_payload_source
        parsed["_structured_payload"] = extraction.structured_payload
        parsed["_parse_status"] = parse_status

        if snapshot is not None:
            snapshot.parse_status = parse_status

        logger.info(
            "Planner (plan): type=%s attempt=%d action=%s version=%d tasks=%d failure_class=%s source=%s",
            self._plan_request_type,
            self._plan_request_attempt,
            parsed.get("plan_action"),
            parsed.get("_request_version", parsed.get("plan_version", 0)),
            len(parsed.get("tasks", [])),
            failure_class,
            extraction.structured_payload_source,
        )
        return parsed, True

    def terminate_plan_run(self, reason: str) -> PlannerRunSnapshot | None:
        """Terminate the active plan-mode subprocess and return final telemetry."""
        handle = self._active_handle
        snapshot = self._plan_runtime
        if handle is None or snapshot is None:
            return snapshot

        self.adapter.terminate(handle)
        self.adapter.check(handle)
        self._refresh_plan_runtime_from_handle(handle)
        proc = handle.process
        exit_code = proc.returncode if proc is not None else None
        snapshot.finish(
            exit_code=exit_code,
            rendered_output=handle.partial_output,
            termination_reason=reason,
        )
        self._last_plan_raw_output = handle.partial_output
        self._last_completed_plan_runtime = snapshot
        self._active_handle = None
        self._plan_runtime = None
        return snapshot

    def plan_runtime_snapshot(self) -> PlannerRunSnapshot | None:
        return self._plan_runtime

    def consume_last_completed_plan_runtime(self) -> PlannerRunSnapshot | None:
        snapshot = self._last_completed_plan_runtime
        self._last_completed_plan_runtime = None
        return snapshot

    @property
    def last_plan_raw_output(self) -> str:
        return self._last_plan_raw_output

    def planner_runtime_summary(self) -> dict[str, Any]:
        runtime_summary = {}
        if hasattr(self.adapter, "runtime_summary"):
            try:
                runtime_summary = getattr(self.adapter, "runtime_summary")()
            except Exception:
                runtime_summary = {}
        return runtime_summary

    @property
    def plan_transport_retry_count(self) -> int:
        return self._plan_transport_retry_count

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _launch_plan_request(
        self,
        *,
        prompt: str,
        request_type: str,
        request_version: int,
        attempt_number: int,
        max_tasks: int | None,
        timeout_retry_count: int,
        transport_retry_count: int,
        task_id: str,
    ) -> None:
        json_schema = build_plan_json_schema(max_tasks=max_tasks)
        self._plan_request_type = request_type
        self._plan_request_version = request_version
        self._plan_request_attempt = attempt_number
        self._last_plan_raw_output = ""
        self._active_handle = self.adapter.start(
            prompt,
            task_id=task_id,
            worker_id="planner",
            json_schema=json_schema,
        )
        output_mode = str(self._active_handle.metadata.get("output_mode", "text"))
        self._plan_runtime = PlannerRunSnapshot(
            request_type=request_type,
            request_version=request_version,
            attempt_number=attempt_number,
            prompt_length=len(prompt),
            output_mode=output_mode,
            timeout_retry_count=timeout_retry_count,
            transport_retry_count=transport_retry_count,
        )

    def _refresh_plan_runtime_from_handle(self, handle: ProcessHandle) -> None:
        snapshot = self._plan_runtime
        if snapshot is None:
            return

        raw_stdout = str(handle.metadata.get("raw_stdout", ""))
        raw_stderr = str(handle.metadata.get("raw_stderr", ""))

        stdout_fragment = raw_stdout[len(snapshot.raw_stdout):]
        stderr_fragment = raw_stderr[len(snapshot.raw_stderr):]
        rendered_fragment = handle.partial_output[len(snapshot.rendered_output):]

        if stdout_fragment or stderr_fragment or rendered_fragment:
            snapshot.record_output(
                stdout_fragment=stdout_fragment,
                stderr_fragment=stderr_fragment,
                rendered_fragment=rendered_fragment,
            )

        snapshot.stream_event_count = int(handle.metadata.get("stream_event_count", 0))
        extraction = extract_planner_structured_output(
            raw_stdout,
            rendered_text=handle.partial_output,
        )
        snapshot.structured_payload = extraction.structured_payload
        snapshot.structured_payload_source = extraction.structured_payload_source
        snapshot.structured_payload_bytes = extraction.structured_payload_bytes
        snapshot.structured_delta_bytes = extraction.structured_delta_bytes
        snapshot.transport_errors = list(extraction.transport_errors)

    @staticmethod
    def _classify_plan_stream_state(snapshot: PlannerRunSnapshot) -> str:
        if snapshot.structured_payload_source in {"tool_use_input", "input_json_delta"} or snapshot.structured_delta_bytes > 0:
            return "structured_output_stream_active"
        if snapshot.output_bytes > 0:
            return "text_stream_active"
        if snapshot.stderr_bytes > 0:
            return "stderr_only"
        return "no_output"
