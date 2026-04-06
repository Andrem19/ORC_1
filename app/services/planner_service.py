"""
Planner service — wraps the planner adapter with prompt building and output parsing.

Supports both synchronous consult() and async start_consultation()/check_consultation().
If the adapter doesn't support async mode (start/check), falls back to synchronous.
"""

from __future__ import annotations

import logging
import time as _time
from typing import TYPE_CHECKING, Any

from app.adapters.base import BaseAdapter, ProcessHandle
from app.models import (
    MemoryEntry,
    OrchestratorState,
    PlannerDecision,
    PlannerOutput,
    TaskResult,
)
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
        self._supports_async: bool | None = None  # lazy detection
        self._last_plan_raw_output: str = ""
        self._plan_request_type: str = "create"
        self._plan_request_version: int = 0
        self._plan_request_attempt: int = 0

    def _check_async_support(self) -> bool:
        """Check if the adapter supports async start/check."""
        if self._supports_async is not None:
            return self._supports_async
        try:
            # Test by calling start — adapters that don't support it raise NotImplementedError
            # We detect this by checking if start() is overridden
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
        """Launch planner as a background process. Returns immediately.

        Falls back to synchronous consult() if the adapter doesn't support
        background execution (stores result for immediate pickup by check_consultation).
        """
        if not self._check_async_support():
            # Adapter doesn't support async — run synchronously and cache result
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
        """Non-blocking check on the running planner.

        Returns:
            (output, is_finished):
              - (None, False) — planner still working
              - (output, True) — planner finished, output parsed
        """
        # Handle synchronous fallback result
        if hasattr(self, "_sync_result") and self._sync_result is not None:
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

        # Planner finished — parse output
        full_output = handle.partial_output
        self._active_handle = None

        proc = handle.process
        if proc is not None and proc.returncode is not None and proc.returncode != 0:
            logger.error(
                "Planner process exited with code %d",
                proc.returncode,
            )
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
        """Whether a planner consultation is currently in progress."""
        return (
            self._active_handle is not None
            or (hasattr(self, "_sync_result") and self._sync_result is not None)
        )

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
        """Launch planner to CREATE a new research plan (plan_v1 or first plan)."""
        from app.plan_prompts import build_plan_creation_prompt

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
        self._plan_request_type = "create"
        self._plan_request_version = plan_version
        self._plan_request_attempt = attempt_number
        self._last_plan_raw_output = ""

        self._active_handle = self.adapter.start(
            prompt,
            task_id=f"plan-create-v{plan_version}-a{attempt_number}",
            worker_id="planner",
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

        prompt = build_plan_revision_prompt(
            goal=goal,
            current_plan=current_plan,
            reports=reports,
            research_context=research_context,
            anti_patterns=anti_patterns,
            worker_ids=worker_ids,
            mcp_problem_summary=mcp_problem_summary,
        )
        logger.info(
            "Starting plan revision v%d → v%d (%d reports, %d chars)",
            current_plan.version, current_plan.version + 1,
            len(reports), len(prompt),
        )
        self._plan_request_type = "revision"
        self._plan_request_version = current_plan.version + 1
        self._plan_request_attempt = 1
        self._last_plan_raw_output = ""

        self._active_handle = self.adapter.start(
            prompt,
            task_id=f"plan-revise-v{current_plan.version}",
            worker_id="planner",
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
        self._plan_request_type = "repair"
        self._plan_request_version = repair_request.plan_version
        self._plan_request_attempt = repair_request.attempt_number
        self._last_plan_raw_output = ""

        self._active_handle = self.adapter.start(
            prompt,
            task_id=f"plan-repair-v{repair_request.plan_version}-a{repair_request.attempt_number}",
            worker_id="planner",
        )

    def check_plan_output(self) -> tuple[dict | None, bool]:
        """Non-blocking check on the running planner (plan-mode).

        Returns:
            (parsed_dict, is_finished):
              - (None, False) — planner still working
              - (dict, True) — planner finished, output parsed
        """
        from app.result_parser import parse_plan_output

        # Handle synchronous fallback
        if hasattr(self, "_sync_result") and self._sync_result is not None:
            result = self._sync_result
            self._sync_result = None
            # _sync_result shouldn't be used in plan mode, but just in case
            return None, True

        if self._active_handle is None:
            logger.error("check_plan_output called but no active planner handle")
            return {
                "plan_action": "continue",
                "reason": "No active planner consultation",
            }, True

        handle = self._active_handle
        new_output, is_finished = self.adapter.check(handle)

        if new_output:
            logger.debug(
                "Planner (plan): received %d chars (total: %d)",
                len(new_output), len(handle.partial_output),
            )

        if not is_finished:
            elapsed = _time.monotonic() - handle.started_at
            logger.info(
                "Planner (plan) still running: type=%s version=%d attempt=%d elapsed=%.0fs output=%d chars",
                self._plan_request_type,
                self._plan_request_version,
                self._plan_request_attempt,
                elapsed,
                len(handle.partial_output),
            )
            return None, False

        full_output = handle.partial_output
        self._last_plan_raw_output = full_output
        self._active_handle = None

        proc = handle.process
        if proc is not None and proc.returncode is not None and proc.returncode != 0:
            logger.error("Planner process exited with code %d", proc.returncode)
            return {
                "plan_action": "continue",
                "reason": f"Planner process exited with code {proc.returncode}",
            }, True

        parsed = parse_plan_output(full_output)
        logger.info(
            "Planner (plan): type=%s attempt=%d action=%s version=%d tasks=%d",
            self._plan_request_type,
            self._plan_request_attempt,
            parsed.get("plan_action"),
            parsed.get("plan_version", 0),
            len(parsed.get("tasks", [])),
        )
        return parsed, True

    @property
    def last_plan_raw_output(self) -> str:
        return self._last_plan_raw_output
