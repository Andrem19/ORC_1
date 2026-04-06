"""
Helpers for keeping planner prompts inside deterministic size budgets.
"""

from __future__ import annotations

import json
from typing import Any

from app.plan_models import ResearchPlan, TaskReport
from app.plan_validation import PlanRepairRequest


def truncate_text(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    if limit <= 20:
        return text[:limit]
    return text[: limit - 15].rstrip() + "\n...[truncated]"


def compact_reports_for_revision(reports: list[TaskReport], *, max_reports: int = 5) -> str:
    lines: list[str] = []
    for report in reports[:max_reports]:
        lines.append(f"- stage plan_v{report.plan_version} | worker={report.worker_id} | status={report.status} | verdict={report.verdict}")
        if report.what_was_done:
            lines.append(f"  done: {truncate_text(report.what_was_done, 220)}")
        if report.key_metrics:
            metrics = ", ".join(f"{k}={v}" for k, v in list(report.key_metrics.items())[:6])
            lines.append(f"  metrics: {metrics}")
        if report.results_table:
            row = report.results_table[0]
            row_view = ", ".join(f"{k}={row.get(k)}" for k in list(row.keys())[:6])
            lines.append(f"  row0: {row_view}")
        if report.error:
            lines.append(f"  error: {truncate_text(report.error, 180)}")
    if len(reports) > max_reports:
        lines.append(f"- ... {len(reports) - max_reports} more reports omitted")
    return "\n".join(lines)


def compact_repair_context(
    repair_request: PlanRepairRequest,
    *,
    max_invalid_stage_chars: int = 6000,
) -> tuple[str, str]:
    """Return compact invalid-stage payload and valid-stage summary."""
    tasks = repair_request.invalid_plan_data.get("tasks", [])
    invalid_stage_numbers = {error.stage_number for error in repair_request.validation_errors if error.stage_number >= 0}

    invalid_tasks = [
        _compact_task(task)
        for task in tasks
        if isinstance(task, dict) and int(task.get("stage_number", -999)) in invalid_stage_numbers
    ]
    valid_tasks = [
        task for task in tasks
        if isinstance(task, dict) and int(task.get("stage_number", -999)) not in invalid_stage_numbers
    ]

    invalid_payload = json.dumps({"tasks": invalid_tasks}, ensure_ascii=False, indent=2)
    invalid_payload = truncate_json(invalid_payload, max_invalid_stage_chars)

    valid_summary_lines = [
        f"- stage {task.get('stage_number')} | {task.get('stage_name', '')}"
        for task in valid_tasks[:8]
    ]
    if len(valid_tasks) > 8:
        valid_summary_lines.append(f"- ... {len(valid_tasks) - 8} more valid stages omitted")

    return invalid_payload, "\n".join(valid_summary_lines)


def summarize_plan_for_markdown(plan: ResearchPlan | dict[str, Any], *, max_tasks: int = 5) -> str:
    """Build a concise markdown summary of a plan."""
    if isinstance(plan, dict):
        version = plan.get("plan_version", "?")
        tasks = plan.get("tasks", [])
        goal = plan.get("goal", "")
    else:
        version = plan.version
        tasks = plan.tasks
        goal = plan.goal

    parts = [f"# Plan v{version}", ""]
    if goal:
        parts.append(f"Goal: {truncate_text(str(goal), 160)}")
        parts.append("")
    parts.append("Stages:")
    for task in list(tasks)[:max_tasks]:
        if isinstance(task, dict):
            stage_number = task.get("stage_number")
            stage_name = task.get("stage_name", "")
            depends_on = task.get("depends_on", [])
        else:
            stage_number = task.stage_number
            stage_name = task.stage_name
            depends_on = task.depends_on
        parts.append(f"- ETAP {stage_number}: {stage_name} | depends_on={depends_on}")
    if len(tasks) > max_tasks:
        parts.append(f"- ... {len(tasks) - max_tasks} more stages omitted")
    return "\n".join(parts)


def truncate_json(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return truncate_text(text, limit)


def _compact_task(task: dict[str, Any]) -> dict[str, Any]:
    instructions = task.get("agent_instructions", [])
    steps = task.get("steps", [])
    compact = {
        "stage_number": task.get("stage_number"),
        "stage_name": task.get("stage_name", ""),
        "depends_on": task.get("depends_on", []),
        "theory": truncate_text(str(task.get("theory", "")), 220),
        "results_table_columns": task.get("results_table_columns", []),
        "decision_gates": task.get("decision_gates", [])[:3],
    }
    if steps:
        compact["steps"] = []
        for step in steps[:4]:
            if not isinstance(step, dict):
                continue
            compact["steps"].append({
                "step_id": step.get("step_id"),
                "kind": step.get("kind"),
                "instruction": truncate_text(str(step.get("instruction", "")), 180),
                "tool_name": step.get("tool_name"),
                "args": step.get("args", {}),
            })
        if len(steps) > 4:
            compact["steps"].append({"omitted": len(steps) - 4})
    else:
        compact["agent_instructions"] = [truncate_text(str(step), 220) for step in instructions[:4]]
        if len(instructions) > 4:
            compact["agent_instructions"].append(f"... {len(instructions) - 4} more instructions omitted")
    return compact
