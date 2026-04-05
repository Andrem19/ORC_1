"""
Worker service — manages worker lifecycle: launch, monitor, collect results.
"""

from __future__ import annotations

import logging
from typing import Any

from app.adapters.base import BaseAdapter
from app.models import (
    MemoryEntry,
    OrchestratorState,
    Task,
    TaskResult,
)
from app.prompts import build_worker_prompt
from app.research_context import MCP_WORKER_INSTRUCTIONS, is_mcp_task
from app.result_parser import parse_worker_output

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

    def execute_task(
        self,
        task: Task,
        memory_entries: list[MemoryEntry] | None = None,
    ) -> TaskResult:
        """Send a task to the worker adapter and return a parsed result."""
        # Auto-detect MCP-related tasks and inject tool instructions
        mcp_instructions = MCP_WORKER_INSTRUCTIONS if is_mcp_task(task.description) else None
        prompt = build_worker_prompt(task, memory_entries, mcp_instructions=mcp_instructions)
        logger.info("Executing task %s via worker adapter", task.task_id)

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

        result = parse_worker_output(
            response.raw_output,
            task_id=task.task_id,
            worker_id=task.assigned_worker_id or "unknown",
        )
        logger.info(
            "Task %s result: status=%s, confidence=%.2f",
            task.task_id,
            result.status,
            result.confidence,
        )
        return result
