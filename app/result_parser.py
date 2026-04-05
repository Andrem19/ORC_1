"""
Parsing and validation of model outputs.

Handles:
- JSON extraction from potentially noisy model output
- Schema validation for planner and worker responses
- Normalization of partial/invalid responses
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.models import (
    PlannerDecision,
    PlannerOutput,
    TaskResult,
)

logger = logging.getLogger("orchestrator.parser")


def _extract_json_block(text: str) -> str | None:
    """Extract JSON from text that may contain markdown code fences or extra content."""
    # Try direct parse first
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Try to find ```json ... ``` block
    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return None


def parse_planner_output(raw: str) -> PlannerOutput:
    """Parse planner model output into a PlannerOutput."""
    json_str = _extract_json_block(raw)
    if json_str is None:
        logger.warning("Planner returned no parseable JSON, defaulting to wait")
        return PlannerOutput(
            decision=PlannerDecision.WAIT,
            reason="Failed to parse planner output as JSON",
            check_after_seconds=60,
        )

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Planner JSON invalid: %s", e)
        return PlannerOutput(
            decision=PlannerDecision.WAIT,
            reason=f"Invalid JSON from planner: {e}",
            check_after_seconds=60,
        )

    # Validate decision
    decision_str = data.get("decision", "wait")
    try:
        decision = PlannerDecision(decision_str)
    except ValueError:
        logger.warning("Unknown planner decision: %s, defaulting to wait", decision_str)
        decision = PlannerDecision.WAIT

    return PlannerOutput(
        decision=decision,
        target_worker_id=data.get("target_worker_id"),
        task_instruction=data.get("task_instruction", ""),
        reason=data.get("reason", ""),
        check_after_seconds=int(data.get("check_after_seconds", 300)),
        memory_update=data.get("memory_update", ""),
        should_finish=bool(data.get("should_finish", False)),
        final_summary=data.get("final_summary", ""),
        reassign_to_worker_id=data.get("reassign_to_worker_id"),
    )


def parse_worker_output(raw: str, task_id: str, worker_id: str) -> TaskResult:
    """Parse worker model output into a TaskResult."""
    json_str = _extract_json_block(raw)

    if json_str is None:
        logger.warning("Worker %s returned no parseable JSON for task %s", worker_id, task_id)
        return TaskResult(
            task_id=task_id,
            worker_id=worker_id,
            status="error",
            error="No parseable JSON in worker output",
            raw_output=raw[:500],
        )

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Worker JSON invalid: %s", e)
        return TaskResult(
            task_id=task_id,
            worker_id=worker_id,
            status="error",
            error=f"Invalid JSON from worker: {e}",
            raw_output=raw[:500],
        )

    status = data.get("status", "error")
    if status not in ("success", "error", "partial"):
        status = "error"

    return TaskResult(
        task_id=task_id,
        worker_id=worker_id,
        status=status,
        summary=str(data.get("summary", ""))[:1000],
        artifacts=data.get("artifacts", []),
        next_hint=str(data.get("next_hint", ""))[:500],
        confidence=float(data.get("confidence", 0.0)),
        error=str(data.get("error", ""))[:500],
        raw_output=raw[:2000],
        mcp_problems=[
            p for p in data.get("mcp_problems", [])
            if isinstance(p, dict)
        ],
    )


def is_duplicate_result(previous: TaskResult | None, current: TaskResult) -> bool:
    """Check if a result is a duplicate of the previous one."""
    if previous is None:
        return False
    return (
        previous.task_id == current.task_id
        and previous.worker_id == current.worker_id
        and previous.summary == current.summary
        and previous.status == current.status
        and previous.error == current.error
    )


def is_useless_result(result: TaskResult) -> bool:
    """Check if a result carries no useful information."""
    return result.is_empty or (result.status == "error" and not result.error)
