"""
Planner service — wraps the planner adapter with prompt building and output parsing.
"""

from __future__ import annotations

import logging
from typing import Any

from app.adapters.base import BaseAdapter
from app.models import (
    MemoryEntry,
    OrchestratorState,
    PlannerDecision,
    PlannerOutput,
    TaskResult,
)
from app.prompts import build_planner_prompt
from app.result_parser import parse_planner_output

logger = logging.getLogger("orchestrator.planner_service")


class PlannerService:
    """High-level service for interacting with the planner model."""

    def __init__(self, adapter: BaseAdapter, timeout: int = 180) -> None:
        self.adapter = adapter
        self.timeout = timeout

    def consult(
        self,
        state: OrchestratorState,
        new_results: list[TaskResult] | None = None,
        worker_ids: list[str] | None = None,
        research_context: str | None = None,
        mcp_problem_summary: str | None = None,
    ) -> PlannerOutput:
        """Call the planner with current state and get a decision."""
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
            len(response.raw_output),
            response.raw_output[:2000],
        )
        logger.info(
            "Planner decided: %s (reason: %s)",
            output.decision.value,
            output.reason[:100],
        )
        return output
