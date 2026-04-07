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


# ---------------------------------------------------------------------------
# Lenient JSON repair — handles common LLM output issues
# ---------------------------------------------------------------------------

# Matches a comma followed by optional whitespace then ] or }
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _repair_json(text: str) -> str:
    """Attempt lightweight repairs on JSON text that fails strict parsing.

    Handles the most common LLM JSON generation errors:
    - Trailing commas before } or ]
    """
    return _TRAILING_COMMA_RE.sub(r"\1", text)


def _extract_json_block(
    text: str,
    *,
    required_keys: set[str] | None = None,
    preferred_keys: set[str] | None = None,
    min_preferred_keys: int = 0,
) -> str | None:
    """Extract the best JSON object from mixed text.

    Candidate selection is schema-aware so nested JSON fragments do not
    outrank the actual top-level payload.
    """
    # Strip BOM and leading/trailing whitespace
    text = text.strip()
    if text.startswith('\ufeff'):
        text = text[1:].strip()

    if not text:
        return None

    best_candidate: str | None = None
    best_score = -1
    repaired = False
    for candidate in _iter_json_candidates(text):
        try:
            data = json.loads(candidate)
            usable_text = candidate
        except json.JSONDecodeError:
            # Try lenient parsing — strip trailing commas and retry
            repaired_text = _repair_json(candidate)
            try:
                data = json.loads(repaired_text)
                usable_text = repaired_text
                repaired = True
            except json.JSONDecodeError:
                continue

        score = _score_json_candidate(
            data,
            usable_text,
            required_keys=required_keys,
            preferred_keys=preferred_keys,
            min_preferred_keys=min_preferred_keys,
        )
        if score > best_score:
            best_candidate = usable_text
            best_score = score

    if best_candidate and repaired:
        logger.info("JSON repaired successfully (trailing comma stripping)")

    return best_candidate


def _iter_json_candidates(text: str) -> list[str]:
    """Return unique JSON-ish candidates ordered from most to least likely."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate.startswith('\ufeff'):
            candidate = candidate[1:].strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)

    _add(text)

    pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
    for match in re.finditer(pattern, text, re.DOTALL):
        _add(match.group(1))

    for candidate in _find_json_candidates(text):
        _add(candidate)

    return candidates


def _score_json_candidate(
    data: Any,
    candidate: str,
    *,
    required_keys: set[str] | None = None,
    preferred_keys: set[str] | None = None,
    min_preferred_keys: int = 0,
) -> int:
    """Return a score for a candidate JSON object or -1 if invalid for the schema."""
    if not isinstance(data, dict):
        return -1

    keys = set(data.keys())
    required = required_keys or set()
    preferred = preferred_keys or set()

    if required and not required.issubset(keys):
        return -1

    preferred_matches = len(keys & preferred) if preferred else 0
    if preferred and preferred_matches < min_preferred_keys:
        return -1

    # Favor candidates that satisfy more schema keys, then prefer longer payloads.
    return preferred_matches * 10000 + len(required & keys) * 1000 + len(candidate)


def _find_json_candidates(text: str) -> list[str]:
    """Find candidate JSON substrings by matching balanced braces.

    Returns candidates ordered from most likely to least likely:
    rightmost-close-brace pairs first (the final JSON object is usually
    the complete response when there's streaming preamble).
    """
    candidates: list[str] = []
    open_positions = [i for i, ch in enumerate(text) if ch == '{']
    close_positions = [i for i, ch in enumerate(text) if ch == '}']

    if not open_positions or not close_positions:
        return candidates

    # Try pairs from rightmost close brace backward
    for close_idx in reversed(close_positions):
        for open_idx in reversed(open_positions):
            if open_idx < close_idx:
                candidate = text[open_idx:close_idx + 1]
                if len(candidate) >= 2:
                    candidates.append(candidate)

    return candidates


def parse_planner_output(raw: str) -> PlannerOutput:
    """Parse planner model output into a PlannerOutput."""
    json_str = _extract_json_block(
        raw,
        required_keys={"decision"},
        preferred_keys={
            "decision",
            "target_worker_id",
            "task_instruction",
            "reason",
            "check_after_seconds",
            "memory_update",
            "should_finish",
            "final_summary",
            "reassign_to_worker_id",
        },
        min_preferred_keys=1,
    )
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
    json_str = _extract_json_block(
        raw,
        required_keys={"status"},
        preferred_keys={
            "status",
            "summary",
            "artifacts",
            "next_hint",
            "confidence",
            "error",
            "mcp_problems",
        },
        min_preferred_keys=1,
    )

    if json_str is None:
        logger.warning("Worker %s returned no parseable JSON for task %s", worker_id, task_id)
        return TaskResult(
            task_id=task_id,
            worker_id=worker_id,
            status="error",
            error="No parseable JSON in worker output",
            raw_output=raw,
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
            raw_output=raw,
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
        raw_output=raw,
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
    """Check if a result carries no useful information.

    A result is only useless if it has NO extractable information
    at all: empty summary, no artifacts, no error message, AND
    no meaningful raw output or confidence score.
    """
    has_content = (
        bool(result.summary and result.summary.strip())
        or bool(result.artifacts)
        or bool(result.error and result.error.strip())
    )
    if has_content:
        return False
    # No structured content — check raw_output and confidence
    has_raw = bool(result.raw_output and result.raw_output.strip())
    if has_raw and result.confidence >= 0.1:
        return False
    if result.status == "error" and not has_raw:
        return True
    if result.is_empty and result.confidence < 0.1 and not has_raw:
        return True
    return False


# ---------------------------------------------------------------------------
# Plan-mode parsers
# ---------------------------------------------------------------------------

def parse_plan_output(raw: str) -> dict[str, Any]:
    """Parse planner output containing a structured plan.

    Returns a dict with keys: plan_action, plan_version, plan_markdown,
    tasks, tasks_to_dispatch, anti_patterns_new, cumulative_summary,
    frozen_base, memory_update, check_after_seconds, should_finish.

    Falls back to plan_action="continue" on parse failure.
    """
    json_str = _extract_json_block(
        raw,
        required_keys={"plan_action", "tasks"},
        preferred_keys={
            "schema_version",
            "plan_action",
            "plan_version",
            "reason",
            "plan_markdown",
            "frozen_base",
            "baseline_run_id",
            "baseline_snapshot_ref",
            "baseline_metrics",
            "tasks",
            "anti_patterns_new",
            "cumulative_summary",
            "principles",
            "memory_update",
            "check_after_seconds",
            "should_finish",
        },
        min_preferred_keys=2,
    )
    if json_str is None:
        logger.warning("Planner returned no parseable plan JSON, defaulting to continue")
        return {
            "plan_action": "continue",
            "plan_version": 0,
            "reason": "Failed to parse planner output as JSON",
            "check_after_seconds": 60,
            "_parse_failed": True,
        }

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Planner plan JSON invalid: %s", e)
        return {
            "plan_action": "continue",
            "plan_version": 0,
            "reason": f"Invalid JSON from planner: {e}",
            "check_after_seconds": 60,
            "_parse_failed": True,
        }

    data["schema_version"] = int(data.get("schema_version", 1) or 1)

    # Normalize plan_action
    action = data.get("plan_action", "create")
    if action not in ("create", "update", "continue"):
        action = "create"
    data["plan_action"] = action

    # Ensure tasks is a list
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []
    data["tasks"] = tasks

    # Ensure anti_patterns_new is a list
    data["anti_patterns_new"] = data.get("anti_patterns_new", [])

    baseline_metrics = data.get("baseline_metrics", {})
    if not isinstance(baseline_metrics, dict):
        baseline_metrics = {}
    data["baseline_metrics"] = baseline_metrics

    for task in data["tasks"]:
        if not isinstance(task, dict):
            continue
        depends_on = task.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        task["depends_on"] = [int(dep) for dep in depends_on if isinstance(dep, int)]
        steps = task.get("steps", [])
        if isinstance(steps, list):
            normalized_steps: list[dict[str, Any]] = []
            for idx, step in enumerate(steps, 1):
                if not isinstance(step, dict):
                    continue
                normalized_steps.append({
                    "step_id": str(step.get("step_id", f"step_{idx}")),
                    "kind": str(step.get("kind", "work")),
                    "instruction": str(step.get("instruction", "")),
                    "tool_name": step.get("tool_name"),
                    "args": step.get("args", {}) if isinstance(step.get("args"), dict) else {},
                    "binds": step.get("binds", []) if isinstance(step.get("binds"), list) else [],
                    "decision_outputs": (
                        step.get("decision_outputs", [])
                        if isinstance(step.get("decision_outputs"), list) else []
                    ),
                    "notes": str(step.get("notes", "")),
                })
            task["steps"] = normalized_steps
        else:
            task["steps"] = []

        instructions = task.get("agent_instructions", [])
        if isinstance(instructions, list):
            task["agent_instructions"] = [str(step) for step in instructions]
        else:
            task["agent_instructions"] = []

    # Legacy compatibility for old planner outputs
    dispatch = data.get("tasks_to_dispatch", [])
    if isinstance(dispatch, list) and data["schema_version"] < 2:
        data["tasks_to_dispatch"] = [int(stage) for stage in dispatch if isinstance(stage, int)]
    else:
        data["tasks_to_dispatch"] = []

    return data


def parse_task_report(
    raw: str,
    task_id: str,
    worker_id: str,
    plan_version: int = 0,
) -> "TaskReport":
    """Parse worker output into a structured TaskReport.

    Falls back to a minimal error report on parse failure.
    """
    from app.plan_models import TaskReport

    json_str = _extract_json_block(
        raw,
        required_keys={"status"},
        preferred_keys={
            "status",
            "what_was_requested",
            "what_was_done",
            "results_table",
            "key_metrics",
            "artifacts",
            "verdict",
            "confidence",
            "error",
            "mcp_problems",
        },
        min_preferred_keys=3,
    )

    if json_str is None:
        logger.warning("Worker %s returned no parseable JSON for plan task %s", worker_id, task_id)
        return TaskReport(
            task_id=task_id,
            worker_id=worker_id,
            plan_version=plan_version,
            status="error",
            error="No parseable JSON in worker output",
            raw_output=raw,
        )

    try:
        data: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("Worker JSON invalid for plan task %s: %s", task_id, e)
        return TaskReport(
            task_id=task_id,
            worker_id=worker_id,
            plan_version=plan_version,
            status="error",
            error=f"Invalid JSON from worker: {e}",
            raw_output=raw,
        )

    status = data.get("status", "error")
    if status not in ("success", "error", "partial"):
        status = "error"

    results_table = data.get("results_table", [])
    if not isinstance(results_table, list):
        results_table = []

    verdict = str(data.get("verdict", "PENDING"))
    if verdict not in {"PROMOTE", "WATCHLIST", "REJECT", "PENDING", "BASELINE"}:
        verdict = "PENDING"

    return TaskReport(
        task_id=task_id,
        worker_id=worker_id,
        plan_version=plan_version,
        status=status,
        what_was_requested=str(data.get("what_was_requested", ""))[:2000],
        what_was_done=str(data.get("what_was_done", ""))[:2000],
        results_table=results_table,
        key_metrics=data.get("key_metrics", {}) if isinstance(data.get("key_metrics"), dict) else {},
        artifacts=data.get("artifacts", []) if isinstance(data.get("artifacts"), list) else [],
        confidence=float(data.get("confidence", 0.0)),
        verdict=verdict,
        error=str(data.get("error", ""))[:500],
        raw_output=raw,
        mcp_problems=[
            p for p in data.get("mcp_problems", []) if isinstance(p, dict)
        ],
    )
