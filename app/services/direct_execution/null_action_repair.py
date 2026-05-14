"""
Detect and repair repeated null-action tool calls from weak providers.

Some providers (notably MiniMax-M2.5) emit tool calls with ``action=null``
after receiving rich response data containing lists of completed jobs,
runs, or analyses.  Instead of picking an identifier from the response
and calling the tool with a concrete ``action='result'`` (or similar),
they re-emit the same null-action call, causing the tool loop to spin
until budget exhaustion.

This module detects the repeated-null pattern, extracts actionable
identifiers from prior tool responses in the transcript, and auto-repairs
the call to a sensible next-step action.

Threshold: after **2** consecutive null-action calls to the same tool
with rich list responses, the third call is auto-repaired.
"""

from __future__ import annotations

import json
from typing import Any

# Handle fields used by the generic fallback repair
_HANDLE_FIELDS = ("project_id", "job_id", "run_id", "snapshot_id", "operation_id", "branch_id")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _is_null_action(arguments: dict[str, Any]) -> bool:
    """Return True if the ``action`` field is null, empty, or absent."""
    action = arguments.get("action")
    if action is None:
        return True
    if isinstance(action, str) and not action.strip():
        return True
    return False


def _count_consecutive_null_action_results(
    transcript: list[dict[str, Any]],
    tool_name: str,
) -> int:
    """Count consecutive tool_result entries for *tool_name* whose original
    arguments had a null action, scanning backwards from the transcript end."""
    count = 0
    for entry in reversed(transcript):
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "tool_result":
            # Non-result entries (assistant_response, nudges) sit between
            # tool_result entries.  Skip them but keep scanning.
            continue
        if str(entry.get("tool") or "").strip() != tool_name:
            break
        orig = entry.get("original_arguments", entry.get("arguments", {}))
        if _is_null_action(orig if isinstance(orig, dict) else {}):
            count += 1
        else:
            break
    return count


# ---------------------------------------------------------------------------
# ID extraction from tool responses
# ---------------------------------------------------------------------------

def _deep_find_dict(payload: Any, key: str) -> Any:
    """Walk a nested dict looking for *key* at any depth.

    Returns the first match found via breadth-first search.
    """
    if not isinstance(payload, dict):
        return None
    queue = [payload]
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict):
            continue
        if key in current:
            return current[key]
        for v in current.values():
            if isinstance(v, dict):
                queue.append(v)
    return None


def _extract_completed_job_ids(payload: Any) -> list[str]:
    """Extract completed job_id values from a backtests_conditions list response.

    Handles multiple response nesting patterns:
    - data.jobs
    - data.data.jobs
    - payload.structuredContent.data.jobs
    """
    jobs = _deep_find_dict(payload, "jobs")
    if not isinstance(jobs, list):
        return []
    ids: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "").strip().lower()
        if status != "completed":
            continue
        job_id = str(job.get("job_id") or "").strip()
        if job_id:
            ids.append(job_id)
    return ids


def _extract_saved_run_ids(payload: Any) -> list[str]:
    """Extract saved run_id values from a backtests_runs list response."""
    runs = _deep_find_dict(payload, "saved_runs")
    if not isinstance(runs, list):
        return []
    ids: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = str(run.get("run_id") or "").strip()
        if run_id:
            ids.append(run_id)
    return ids


def _extract_completed_study_job_ids(payload: Any) -> list[str]:
    """Extract completed job_id values from a backtests_studies list response."""
    jobs = _deep_find_dict(payload, "jobs")
    if not isinstance(jobs, list):
        return []
    ids: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status") or "").strip().lower()
        if status not in ("completed", "succeeded"):
            continue
        job_id = str(job.get("job_id") or "").strip()
        if job_id:
            ids.append(job_id)
    return ids


def _extract_snapshot_ids(payload: Any) -> list[str]:
    """Extract snapshot_id values from a backtests_strategy list response.

    Handles both ``snapshots`` list and top-level ``snapshot_id`` field.
    """
    snapshots = _deep_find_dict(payload, "snapshots")
    if isinstance(snapshots, list):
        ids: list[str] = []
        for snap in snapshots:
            if not isinstance(snap, dict):
                continue
            sid = str(snap.get("snapshot_id") or "").strip()
            if sid:
                ids.append(sid)
        return ids
    # Fallback: top-level snapshot_id
    sid = _deep_find_dict(payload, "snapshot_id")
    if isinstance(sid, str) and sid.strip():
        return [sid.strip()]
    return []


# ---------------------------------------------------------------------------
# Tool-specific repair strategies
# ---------------------------------------------------------------------------

_TOOL_REPAIR_STRATEGIES: dict[str, list[dict[str, Any]]] = {
    "backtests_conditions": [
        {"action": "result", "id_field": "job_id", "extractor": _extract_completed_job_ids},
    ],
    "backtests_analysis": [
        {"action": "status", "id_field": "job_id", "extractor": _extract_completed_study_job_ids},
    ],
    "backtests_studies": [
        {"action": "result", "id_field": "job_id", "extractor": _extract_completed_study_job_ids},
    ],
    "backtests_runs": [
        {"action": "detail", "id_field": "run_id", "extractor": _extract_saved_run_ids},
    ],
    "backtests_strategy": [
        {"action": "inspect", "id_field": "snapshot_id", "extractor": _extract_snapshot_ids},
    ],
}

# After how many consecutive null-action results to trigger auto-repair
_REPAIR_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Main repair entry point
# ---------------------------------------------------------------------------

def repair_null_action_loop(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(repaired_arguments, repair_note)`` or ``(None, None)``.

    Conditions for repair:
    1. The current ``arguments`` have a null/empty ``action``.
    2. The transcript shows >= _REPAIR_THRESHOLD consecutive null-action
       results for the same tool.
    3. At least one of those results contains extractable completed IDs,
       OR the tool has a generic fallback action available.

    When all conditions are met, the function returns repaired arguments
    with a concrete ``action`` and the first extracted ID.
    """
    tool = str(tool_name or "").strip()
    if not tool:
        return None, None
    if not _is_null_action(arguments):
        return None, None

    consecutive = _count_consecutive_null_action_results(transcript, tool)
    if consecutive < _REPAIR_THRESHOLD:
        return None, None

    # Try tool-specific strategies first
    strategies = _TOOL_REPAIR_STRATEGIES.get(tool)
    if strategies:
        for strategy in strategies:
            extractor = strategy["extractor"]
            target_action = strategy["action"]
            id_field = strategy["id_field"]

            for entry in reversed(transcript):
                if not isinstance(entry, dict):
                    continue
                if entry.get("kind") != "tool_result":
                    continue
                if str(entry.get("tool") or "").strip() != tool:
                    continue
                payload = entry.get("payload")
                if not payload:
                    continue
                ids = extractor(payload)
                if ids:
                    repaired = dict(arguments)
                    repaired["action"] = target_action
                    repaired[id_field] = ids[0]
                    note = (
                        f"null-action loop repair: {tool} action=null → "
                        f"action='{target_action}', {id_field}='{ids[0]}' "
                        f"(consecutive_null_actions={consecutive})"
                    )
                    return repaired, note

    # Generic fallback: set action='inspect' for any tool that has handle
    # IDs already in the arguments (e.g. research_memory with project_id).
    repaired = dict(arguments)
    for field in _HANDLE_FIELDS:
        value = str(arguments.get(field) or "").strip()
        if value:
            repaired["action"] = "inspect"
            note = (
                f"null-action loop repair (generic): {tool} action=null → "
                f"action='inspect', {field}='{value}' "
                f"(consecutive_null_actions={consecutive})"
            )
            return repaired, note

    return None, None


__all__ = ["repair_null_action_loop"]
