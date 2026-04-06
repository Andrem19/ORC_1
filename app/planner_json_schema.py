"""
JSON schema builders for planner CLI structured output mode.
"""

from __future__ import annotations

import json


def build_plan_json_schema(max_tasks: int | None = None) -> str:
    """Return a compact JSON schema string for plan-mode planner output."""
    task_schema: dict[str, object] = {
        "type": "object",
        "required": [
            "stage_number",
            "stage_name",
            "depends_on",
            "steps",
        ],
        "properties": {
            "stage_number": {"type": "integer"},
            "stage_name": {"type": "string"},
            "theory": {"type": "string"},
            "depends_on": {"type": "array", "items": {"type": "integer"}},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["step_id", "kind", "instruction"],
                    "properties": {
                        "step_id": {"type": "string"},
                        "kind": {"type": "string", "enum": ["tool_call", "work", "decision", "record"]},
                        "instruction": {"type": "string"},
                        "tool_name": {"type": ["string", "null"]},
                        "args": {"type": "object"},
                        "binds": {"type": "array", "items": {"type": "string"}},
                        "decision_outputs": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"},
                    },
                    "additionalProperties": True,
                },
            },
            "results_table_columns": {"type": "array", "items": {"type": "string"}},
            "decision_gates": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": True,
    }
    tasks_schema: dict[str, object] = {"type": "array", "items": task_schema}
    if max_tasks is not None:
        tasks_schema["maxItems"] = max_tasks

    schema = {
        "type": "object",
        "required": ["schema_version", "plan_action", "plan_version", "tasks"],
        "properties": {
            "schema_version": {"type": "integer"},
            "plan_action": {"type": "string", "enum": ["create", "update", "continue"]},
            "plan_version": {"type": "integer"},
            "reason": {"type": "string"},
            "plan_markdown": {"type": "string"},
            "frozen_base": {"type": "string"},
            "baseline_run_id": {"type": "string"},
            "baseline_snapshot_ref": {"type": "string"},
            "baseline_metrics": {"type": "object"},
            "tasks": tasks_schema,
            "anti_patterns_new": {"type": "array", "items": {"type": "object"}},
            "cumulative_summary": {"type": "string"},
            "principles": {"type": "array", "items": {"type": "string"}},
            "memory_update": {"type": "string"},
            "check_after_seconds": {"type": "integer"},
            "should_finish": {"type": "boolean"},
        },
        "additionalProperties": True,
    }
    return json.dumps(schema, separators=(",", ":"))
