"""
Worker service — manages worker lifecycle: launch, monitor, collect results.

Supports both synchronous execute_task() and asynchronous start_task/check_task.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import BaseAdapter, ProcessHandle
from app.models import (
    MemoryEntry,
    ProcessInfo,
    Task,
    TaskResult,
)
from app.plan_models import task_report_to_task_result
from app.prompts import build_worker_prompt
from app.research_context import MCP_WORKER_INSTRUCTIONS, is_mcp_task
from app.result_parser import parse_task_report, parse_worker_output

logger = logging.getLogger("orchestrator.worker_service")


class WorkerService:
    """High-level service for managing worker agents."""

    def __init__(
        self,
        adapter: BaseAdapter,
        timeout: int = 300,
    ) -> None:
        self.adapter = adapter
        self.timeout = timeout
        self._active_handles: dict[str, ProcessHandle] = {}

    # ---------------------------------------------------------------
    # Synchronous path (legacy, used by planner adapter and tests)
    # ---------------------------------------------------------------

    def execute_task(
        self,
        task: Task,
        memory_entries: list[MemoryEntry] | None = None,
    ) -> TaskResult:
        """Send a task to the worker adapter and return a parsed result (blocking)."""
        # Auto-detect MCP-related tasks and inject tool instructions
        is_mcp = is_mcp_task(task.description)
        mcp_instructions = MCP_WORKER_INSTRUCTIONS if is_mcp else None
        prompt = build_worker_prompt(task, memory_entries, mcp_instructions=mcp_instructions)
        logger.info("Executing task %s via worker adapter (mcp=%s)", task.task_id, is_mcp)
        logger.debug("Worker prompt for task %s (%d chars):\n%s", task.task_id, len(prompt), prompt)

        response = self.adapter.invoke(prompt, timeout=self.timeout)

        if not response.success:
            logger.warning("Worker call failed for task %s: %s", task.task_id, response.error[:200])
            return TaskResult(
                task_id=task.task_id,
                worker_id=task.assigned_worker_id or "unknown",
                status="error",
                error=response.error[:500] if response.error else "Worker call failed",
                raw_output=response.raw_output[:500],
            )

        logger.debug(
            "Worker raw output for task %s (%d chars):\n%s",
            task.task_id, len(response.raw_output), response.raw_output[:2000],
        )

        result = self._parse_task_output(
            task=task,
            raw_output=response.raw_output,
        )
        logger.info(
            "Task %s result: status=%s, confidence=%.2f",
            task.task_id,
            result.status,
            result.confidence,
        )
        return result

    # ---------------------------------------------------------------
    # Async path (non-blocking background execution)
    # ---------------------------------------------------------------

    def start_task(
        self,
        task: Task,
        memory_entries: list[MemoryEntry] | None = None,
    ) -> ProcessInfo:
        """Launch a worker as a background process. Returns immediately."""
        is_mcp = is_mcp_task(task.description)
        mcp_instructions = MCP_WORKER_INSTRUCTIONS if is_mcp else None
        prompt = build_worker_prompt(task, memory_entries, mcp_instructions=mcp_instructions)

        logger.info("Starting task %s as background process (mcp=%s)", task.task_id, is_mcp)
        logger.debug("Worker prompt for task %s (%d chars)", task.task_id, len(prompt))

        handle = self.adapter.start(
            prompt,
            task_id=task.task_id,
            worker_id=task.assigned_worker_id or "unknown",
        )

        process_info = ProcessInfo(
            task_id=task.task_id,
            worker_id=task.assigned_worker_id or "unknown",
            pid=handle.process.pid if handle.process else None,
            prompt=prompt,
        )

        self._active_handles[task.task_id] = handle
        logger.info("Task %s started (pid=%s)", task.task_id, process_info.pid)
        return process_info

    def check_task(
        self,
        task: Task,
        process_info: ProcessInfo,
    ) -> tuple[TaskResult | None, bool]:
        """Check on a running worker. Non-blocking.

        Returns:
            (result, is_finished):
              - (None, False)  — worker still running
              - (result, True) — worker finished, result parsed
        """
        handle = self._active_handles.get(task.task_id)
        if handle is None:
            logger.error("No handle for task %s — process lost", task.task_id)
            return TaskResult(
                task_id=task.task_id,
                worker_id=task.assigned_worker_id or "unknown",
                status="error",
                error="Process handle lost (orchestrator may have restarted)",
            ), True

        prev_stdout_len = len(process_info.partial_output or "")
        prev_stderr_len = len(process_info.partial_error_output or "")
        new_output, is_finished = self.adapter.check(handle)
        process_info.partial_output = handle.partial_output
        process_info.partial_error_output = handle.partial_error_output
        self._update_process_runtime(process_info, prev_stdout_len=prev_stdout_len, prev_stderr_len=prev_stderr_len)

        if new_output:
            logger.debug(
                "Task %s: received %d chars (total: %d)",
                task.task_id, len(new_output), len(handle.partial_output),
            )

        if not is_finished:
            return None, False

        # Process finished — clean up handle
        del self._active_handles[task.task_id]

        # Store return code
        if handle.process is not None:
            process_info.returncode = handle.process.returncode

        full_output = handle.partial_output

        # Non-zero exit code → error
        if process_info.returncode is not None and process_info.returncode != 0:
            logger.warning(
                "Task %s worker exited with code %d (stderr=%s)",
                task.task_id, process_info.returncode, process_info.partial_error_output[:200],
            )
            return TaskResult(
                task_id=task.task_id,
                worker_id=task.assigned_worker_id or "unknown",
                status="error",
                error=(
                    process_info.partial_error_output[:500]
                    or f"Worker exited with code {process_info.returncode}"
                ),
                raw_output=full_output,
            ), True

        # Parse successful output
        result = self._parse_task_output(task=task, raw_output=full_output)
        logger.info(
            "Task %s result: status=%s, confidence=%.2f",
            task.task_id, result.status, result.confidence,
        )
        return result, True

    def _update_process_runtime(
        self,
        process_info: ProcessInfo,
        *,
        prev_stdout_len: int,
        prev_stderr_len: int,
    ) -> None:
        stdout_now = process_info.partial_output or ""
        stderr_now = process_info.partial_error_output or ""
        process_info.stdout_bytes = len(stdout_now.encode("utf-8", errors="replace"))
        process_info.stderr_bytes = len(stderr_now.encode("utf-8", errors="replace"))

        if len(stdout_now) > prev_stdout_len or len(stderr_now) > prev_stderr_len:
            now_iso = datetime.now(timezone.utc).isoformat()
            if process_info.first_output_at is None:
                process_info.first_output_at = now_iso
            process_info.last_output_at = now_iso
            process_info.monitor_warning_sent = False
            process_info.monitor_state = ""

    def _parse_task_output(self, task: Task, raw_output: str) -> TaskResult:
        """Parse output using the task-specific schema."""
        worker_id = task.assigned_worker_id or "unknown"
        if task.metadata.get("plan_mode"):
            report = parse_task_report(
                raw_output,
                task_id=task.task_id,
                worker_id=worker_id,
                plan_version=task.metadata.get("plan_version", 0),
            )
            return task_report_to_task_result(report)

        return parse_worker_output(
            raw_output,
            task_id=task.task_id,
            worker_id=worker_id,
        )

    def terminate_task(self, task_id: str) -> None:
        """Terminate a running worker process."""
        handle = self._active_handles.pop(task_id, None)
        if handle is None:
            return
        self.adapter.terminate(handle)
        logger.info("Terminated worker for task %s", task_id)

    # ---------------------------------------------------------------
    # MCP health check
    # ---------------------------------------------------------------

    def check_mcp_health(
        self, cli_path: str, timeout: int = 60, retries: int = 3, delay: int = 10,
    ) -> bool:
        """Run a lightweight CLI probe to check if MCP tools are accessible.

        Launches a fresh CLI invocation (which establishes its own MCP
        connections) with a simple probe prompt.  Retries up to ``retries``
        times with ``delay`` seconds between attempts.  Returns True if any
        attempt succeeds.
        """
        import subprocess as sp
        import time

        probe_prompt = (
            "Call the MCP tool backtests_runs with action='catalog' and respond "
            "with just the word OK if it works, or ERROR if it fails. "
            "Output ONLY OK or ERROR — nothing else."
        )
        for attempt in range(1, retries + 1):
            try:
                result = sp.run(
                    [cli_path, "-p", probe_prompt],
                    capture_output=True, text=True, timeout=timeout,
                )
                output = result.stdout.strip().lower()
                if "ok" in output and "error" not in output:
                    if attempt > 1:
                        logger.info("MCP health probe OK on attempt %d", attempt)
                    return True
                logger.warning(
                    "MCP health probe returned non-OK (attempt %d/%d): "
                    "rc=%d stdout=%s stderr=%s",
                    attempt, retries,
                    result.returncode,
                    result.stdout[:200],
                    result.stderr[:200],
                )
            except sp.TimeoutExpired:
                logger.warning(
                    "MCP health probe timed out after %ds (attempt %d/%d)",
                    timeout, attempt, retries,
                )
            except FileNotFoundError:
                logger.warning("MCP health probe: CLI not found at '%s'", cli_path)
                return False  # no point retrying
            except Exception as exc:
                logger.warning(
                    "MCP health probe failed (attempt %d/%d): %s",
                    attempt, retries, exc,
                )

            if attempt < retries:
                logger.info("Retrying MCP health probe in %ds...", delay)
                time.sleep(delay)

        return False

    # ---------------------------------------------------------------
    # Plan-mode path (structured task with results table)
    # ---------------------------------------------------------------

    def start_plan_task(
        self,
        task: Task,
        plan_version: int = 0,
        stage_number: int = 0,
        stage_name: str = "",
        theory: str = "",
        agent_instructions: list[str] | None = None,
        steps: list[Any] | None = None,
        results_table_columns: list[str] | None = None,
        dependency_reports: list[Any] | None = None,
    ) -> ProcessInfo:
        """Launch a worker for a structured plan task (plan-mode)."""
        from app.plan_prompts import build_plan_task_prompt
        from app.research_context import MCP_WORKER_INSTRUCTIONS, is_mcp_task

        instructions = agent_instructions or []
        step_text = " ".join(str(getattr(step, "instruction", "")) for step in (steps or []))
        is_mcp = is_mcp_task(task.description) or is_mcp_task(" ".join(instructions)) or is_mcp_task(step_text)
        mcp_instructions = MCP_WORKER_INSTRUCTIONS if is_mcp else None

        prompt = build_plan_task_prompt(
            stage_number=stage_number,
            stage_name=stage_name,
            theory=theory,
            agent_instructions=instructions,
            steps=steps,
            results_table_columns=results_table_columns,
            plan_version=plan_version,
            mcp_instructions=mcp_instructions,
            dependency_reports=dependency_reports,
        )

        logger.info(
            "Starting plan task %s (stage %d: %s) as background process",
            task.task_id, stage_number, stage_name,
        )
        logger.debug("Plan task prompt for %s (%d chars)", task.task_id, len(prompt))

        handle = self.adapter.start(
            prompt,
            task_id=task.task_id,
            worker_id=task.assigned_worker_id or "unknown",
        )

        process_info = ProcessInfo(
            task_id=task.task_id,
            worker_id=task.assigned_worker_id or "unknown",
            pid=handle.process.pid if handle.process else None,
            prompt=prompt,
        )

        self._active_handles[task.task_id] = handle
        return process_info
