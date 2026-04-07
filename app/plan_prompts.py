"""
Prompt construction for plan-mode orchestrator.

Encodes the FOUNDATION plan pattern:
- Structured multi-task plans with theory, instructions, results tables, decision gates
- Cumulative knowledge base (anti-patterns, frozen base, incremental refinement)
- Workers return structured reports with filled results tables
"""

from __future__ import annotations

import json
import random
from typing import Any

from app.plan_models import ResearchPlan, TaskReport
from app.plan_prompt_budget import (
    compact_repair_context,
    compact_reports_for_revision,
    truncate_text,
)
from app.planner_context_builder import build_planner_context
from app.plan_validation import PlanRepairRequest


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

_PLAN_SCHEMA_BODY = """\
  "schema_version": {version},
  "plan_action": "create|update",
  "plan_version": 1,
  "reason": "why this plan or revision",
  "plan_markdown": "concise plan summary",
  "frozen_base": "active-signal-v1@1",
  "baseline_snapshot_ref": "active-signal-v1@1",
  "baseline_metrics": {{"net_pnl": 0.0, "sharpe": 0.0, "trades": 0, "max_drawdown_pct": 0.0}},
  "tasks": [
    {{
      "stage_number": 0,
      "stage_name": "Baseline",
      "theory": "Confirm baseline metrics and gather context",
      "depends_on": [],
      "steps": [
        {{
          "step_id": "baseline_run",
          "kind": "tool_call",
          "instruction": "Run baseline backtest",
          "tool_name": "backtests_runs",
          "args": {{"action": "start", "snapshot_id": "active-signal-v1", "version": "1"}}
        }}
      ]
    }},
    {{
      "stage_number": 1,
      "stage_name": "Feature A",
      "theory": "Test feature A independently",
      "depends_on": [0],
      "steps": []
    }},
    {{
      "stage_number": 2,
      "stage_name": "Feature B",
      "theory": "Test feature B independently (parallel with stage 1)",
      "depends_on": [0],
      "steps": []
    }},
    {{
      "stage_number": 3,
      "stage_name": "Integration",
      "theory": "Combine best features from parallel branches",
      "depends_on": [1, 2],
      "steps": []
    }}
  ],
  "cumulative_summary": "updated knowledge summary",
  "principles": ["principle 1", "principle 2"],
  "check_after_seconds": 300,
  "should_finish": false
}}"""


def _plan_schema(version: int) -> str:
    """Generate plan JSON schema with the given version number."""
    return "{" + _PLAN_SCHEMA_BODY.format(version=version)


# Single schema version for all plan operations (creation, revision, repair)
PLANNER_PLAN_SCHEMA = _plan_schema(4)

WORKER_REPORT_SCHEMA = """{
  "status": "success|error|partial",
  "what_was_requested": "brief summary of what was asked",
  "what_was_done": "detailed summary of what was actually done",
  "results_table": [
    {"run_id": "...", "net_pnl": 0, "trades": 0, "PF": 0.0, "WR": 0.0, "verdict": "PROMOTE|REJECT|WATCHLIST"}
  ],
  "key_metrics": {"metric_name": value},
  "artifacts": ["list of produced files or items"],
  "verdict": "PROMOTE|WATCHLIST|REJECT|PENDING",
  "confidence": 0.9,
  "error": "error message if any, empty otherwise",
  "mcp_problems": [{"tool_name": "...", "description": "...", "suggestion": "...", "severity": "low|medium|high"}]
}"""

MAX_CREATE_TASKS = 5


def _planner_context(research_context: str | None) -> str:
    return build_planner_context(research_context=research_context, baseline_bootstrap=None)


# ---------------------------------------------------------------------------
# Planner prompts
# ---------------------------------------------------------------------------

def build_plan_creation_prompt(
    goal: str,
    research_context: str | None = None,
    anti_patterns: list[dict[str, Any]] | None = None,
    cumulative_summary: str = "",
    worker_ids: list[str] | None = None,
    mcp_problem_summary: str | None = None,
    previous_plan_markdown: str | None = None,
    validation_warnings: list[dict[str, Any]] | None = None,
    planner_system_prompt: str = "",
    operator_directives: str = "",
    research_history: list[str] | None = None,
) -> str:
    """Build the prompt for creating a new research plan (plan_v1 or after full resolution).

    This is the main prompt that turns the planner into a plan author.
    """
    parts: list[str] = []

    if operator_directives:
        parts.append("## Operator Directives")
        parts.append(operator_directives)
        parts.append("")

    if planner_system_prompt:
        parts.append("## System Instructions")
        parts.append(planner_system_prompt)
        parts.append("")

    parts.append("## Goal")
    parts.append(truncate_text(goal, 500))
    parts.append("")

    context_text = _planner_context(research_context)
    if context_text:
        parts.append("## Context")
        parts.append(truncate_text(context_text, 4200))
        parts.append("")

    if cumulative_summary:
        parts.append("## Cumulative Knowledge")
        parts.append(truncate_text(cumulative_summary, 900))
        parts.append("")

    if anti_patterns:
        parts.append("## Anti-Patterns")
        for ap in anti_patterns:
            parts.append(
                f"- {ap.get('category', '?')}: "
                f"{truncate_text(str(ap.get('description', '')), 140)} "
                f"({ap.get('evidence_count', '?')} failures)"
            )
        parts.append("")

    if research_history:
        parts.append("## Research History (DO NOT repeat these approaches)")
        parts.append(
            "These approaches were already tested in previous plan versions. "
            "DO NOT propose them again unless you have a substantially different theory."
        )
        for line in research_history:
            parts.append(line)
        parts.append("")

    if previous_plan_markdown:
        parts.append("## Previous Plan")
        parts.append(truncate_text(previous_plan_markdown, 1200))
        parts.append("")

    if mcp_problem_summary:
        parts.append("## Known MCP Problems (avoid these mistakes)")
        parts.append(truncate_text(mcp_problem_summary, 900))
        parts.append("")

    if validation_warnings:
        parts.append("## Previous Plan Validation Warnings")
        parts.append(
            "The previous plan had these issues. Avoid repeating them:"
        )
        for w in validation_warnings[:5]:
            parts.append(
                f"- stage {w.get('stage_number', '?')} "
                f"{w.get('code', '?')}: {w.get('message', '')}"
            )
        parts.append("")

    if worker_ids:
        shuffled = list(worker_ids)
        random.shuffle(shuffled)
        parts.append("## Workers")
        parts.append(", ".join(shuffled))
        parts.append("")

    parts.append("## Requirements")
    parts.append(
        f"- Return at most {MAX_CREATE_TASKS} stages.\n"
        "- Prefer fewer, higher-value stages over exhaustive plans.\n"
        "- Use `depends_on` as the only execution contract.\n"
        "- Each stage must be executable with exact MCP calls and parameters.\n"
        "- Each stage must define `steps`, not `agent_instructions`.\n"
        "- `plan_markdown` must be concise and summarize the plan instead of duplicating every detail.\n"
        "- Stages with satisfied dependencies may run in parallel.\n"
        "- **Design parallel branches**: Create 2-3 independent investigation stages that depend\n"
        "  on the same parent stage. This allows multiple workers to run simultaneously.\n"
        "  Example: Stage 1 explores feature A, Stage 2 explores feature B — both depend on\n"
        "  Stage 0, so they run in PARALLEL. A later stage depends on [1, 2] to merge results.\n"
    )
    parts.append("")

    parts.append("## Execution Rules")
    parts.append(
        "- Use intra-stage refs for outputs of previous steps in the same stage: {{step:step_id.run_id}}, {{step:step_id.snapshot_ref}}, {{step:step_id.results_table[0].column}}\n"
        "- NEVER emit placeholders like <best_snapshot_id>, <run_id>, <v>, ellipses, or incomplete tool calls.\n"
        "- NEVER use tool aliases like snapshots(...), or backtests_runs(action='run').\n"
        "- You do NOT need to specify `symbol`, `anchor_timeframe`, or `execution_timeframe` in backtests_plan, backtests_runs, backtests_walkforward, or backtests_conditions args — the system auto-fills BTCUSDT/1h/5m.\n"
        "- Use concrete IDs only when already known from context.\n"
        "- Do NOT use symbolic cross-stage refs like {{stage:N.run_id}} — the orchestrator resolves those at dispatch time.\n"
        "- Define clear verdict criteria: PROMOTE, WATCHLIST, REJECT.\n"
        "- backtests_strategy(action='clone') uses `source_snapshot_id` (not snapshot_id).\n"
        "- backtests_runs(action='start') requires `version` — use the version from the snapshot ref (e.g. '1').\n"
    )
    parts.append("")

    parts.append("## Output")
    parts.append("Respond with JSON only matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")
    parts.append("")
    parts.append("## Intra-Stage Reference Contract")
    parts.append(
        "- Allowed intra-stage syntax: {{step:step_id.run_id}}, {{step:step_id.snapshot_ref}}, "
        "{{step:step_id.version}}, {{step:step_id.results_table[0].column_name}}\n"
        "- Do NOT use <run_id>, <v>, <snapshot_id>, or `...`\n"
        "- Use concrete IDs only when they are already known from the provided context"
    )
    parts.append("")
    parts.append("## Examples")
    parts.append("Valid step ref: backtests_runs(action='inspect', run_id='{{step:baseline_run.run_id}}', view='detail')")
    parts.append("Invalid: snapshots(action='fork', snapshot_ref='<run_id>')")
    parts.append("Invalid: backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail') — do NOT use cross-stage refs")
    parts.append("")
    parts.append("### Parallel Branch Example")
    parts.append(
        "Stage 0: Baseline (depends_on=[])\n"
        "Stage 1: Feature A — cf_regime_filter (depends_on=[0])\n"
        "Stage 2: Feature B — cf_volatility_adaptive (depends_on=[0])  ← runs IN PARALLEL with Stage 1\n"
        "Stage 3: Combine best features (depends_on=[1, 2])\n"
        "Stage 4: Robustness validation (depends_on=[3])"
    )

    return "\n".join(parts)


def build_plan_revision_prompt(
    goal: str,
    current_plan: ResearchPlan,
    reports: list[TaskReport],
    research_context: str | None = None,
    anti_patterns: list[dict[str, Any]] | None = None,
    worker_ids: list[str] | None = None,
    mcp_problem_summary: str | None = None,
    validation_warnings: list[dict[str, Any]] | None = None,
    planner_system_prompt: str = "",
    operator_directives: str = "",
    research_history: list[str] | None = None,
) -> str:
    """Build the prompt for revising a plan based on collected worker reports.

    The planner receives the current plan + all reports and generates the NEXT version.
    """
    parts: list[str] = []

    if operator_directives:
        parts.append("## Operator Directives")
        parts.append(operator_directives)
        parts.append("")

    if planner_system_prompt:
        parts.append("## System Instructions")
        parts.append(planner_system_prompt)
        parts.append("")

    parts.append("## Revision Context")
    parts.append(
        "Workers have completed tasks from the current plan "
        "and returned reports. Your job is to analyze the results and write the NEXT VERSION "
        "of the research plan (plan_v" + str(current_plan.version + 1) + ")."
    )
    parts.append("")

    parts.append("## Global Goal")
    parts.append(goal)
    parts.append("")

    context_text = _planner_context(research_context)
    if context_text:
        parts.append("## Context")
        parts.append(truncate_text(context_text, 3600))
        parts.append("")

    if current_plan.baseline_run_id or current_plan.baseline_metrics:
        parts.append("## Measured Baseline (source of truth)")
        if current_plan.baseline_snapshot_ref:
            parts.append(f"- Snapshot: {current_plan.baseline_snapshot_ref}")
        if current_plan.baseline_run_id:
            parts.append(f"- Run ID: {current_plan.baseline_run_id}")
        if current_plan.baseline_metrics:
            parts.append(f"- Metrics: {current_plan.baseline_metrics}")
        parts.append("")

    # Current plan
    parts.append(f"## Current Plan (v{current_plan.version})")
    if current_plan.plan_markdown:
        parts.append(truncate_text(current_plan.plan_markdown, 1600))
    else:
        parts.append(f"Goal: {current_plan.goal}")
        parts.append(f"Frozen base: {current_plan.frozen_base}")
    parts.append("")

    parts.append("## Worker Reports")
    parts.append(compact_reports_for_revision(reports))
    parts.append("")

    if research_history:
        parts.append("## Research History (DO NOT repeat these approaches)")
        parts.append(
            "These approaches were already tested in previous plan versions. "
            "DO NOT propose them again unless you have a substantially different theory."
        )
        for line in research_history:
            parts.append(line)
        parts.append("")

    # Anti-patterns
    if anti_patterns:
        parts.append("## Anti-Patterns (carry forward + add new ones if evidence exists)")
        for ap in anti_patterns:
            parts.append(
                f"- **{ap.get('category', '?')}**: {ap.get('description', '')} "
                f"({ap.get('evidence_count', '?')} failures)"
            )
        parts.append("")

    if mcp_problem_summary:
        parts.append("## Known MCP Problems")
        parts.append(mcp_problem_summary)
        parts.append("")

    if validation_warnings:
        parts.append("## Previous Plan Validation Warnings")
        parts.append(
            "The previous plan had these issues. Avoid repeating them:"
        )
        for w in validation_warnings[:5]:
            parts.append(
                f"- stage {w.get('stage_number', '?')} "
                f"{w.get('code', '?')}: {w.get('message', '')}"
            )
        parts.append("")

    if worker_ids:
        shuffled = list(worker_ids)
        random.shuffle(shuffled)
        parts.append("## Available Workers")
        parts.append(", ".join(shuffled))
        parts.append("")

    # Revision instructions
    parts.append("## Revision Instructions")
    parts.append(
        "Analyze the reports above and write plan_v" + str(current_plan.version + 1) + ":\n"
        "\n"
        "1. **Update verdicts**: For each completed stage, set PROMOTE / WATCHLIST / REJECT\n"
        "   based on the decision gates and actual results\n"
        "2. **Carry forward**: All PROMOTED and WATCHLIST items continue into the new plan\n"
        "3. **Add anti-patterns**: Any REJECTED approach with solid evidence goes to anti-patterns\n"
        "4. **New stages**: Add new investigation stages based on what was learned\n"
        "5. **Update cumulative summary**: Incorporate new findings\n"
        "6. **Frozen base**: Keep the same frozen base — NEVER modify it\n"
        "7. **Dependencies**: Use `depends_on` as the ONLY execution contract. "
        "Stages with the same depends_on WILL run simultaneously on separate workers. "
        "Always design a DAG with parallel branches, never a linear chain.\n"
        "8. **Future outputs**: Use {{step:step_id.run_id}} for earlier steps within the same stage. "
        "Do NOT use cross-stage refs like {{stage:N.run_id}} — the orchestrator resolves those at dispatch time.\n"
        "9. Emit schema v4 with `steps`, not free-form `agent_instructions`.\n"
        "10. **Concrete IDs**: Use concrete IDs (run_id, snapshot_id) only when they are already known "
        "from worker reports or context. Do not invent or guess IDs.\n"
        "11. **Parallel branches**: When adding new investigation stages, create 2-3 independent branches "
        "that depend on the same parent stage so workers can run in parallel.\n"
        "12. backtests_strategy(action='clone') uses `source_snapshot_id` (not snapshot_id).\n"
        "13. backtests_runs(action='start') requires `version`.\n"
    )
    parts.append("")

    parts.append("## Output")
    parts.append("Respond with JSON only matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")

    return "\n".join(parts)


def build_plan_repair_prompt(
    repair_request: PlanRepairRequest,
    research_context: str | None = None,
    worker_ids: list[str] | None = None,
    mcp_problem_summary: str | None = None,
    planner_system_prompt: str = "",
    operator_directives: str = "",
) -> str:
    """Build the prompt for repairing one invalid planner output."""
    invalid_payload, valid_summary = compact_repair_context(repair_request)
    parts: list[str] = []

    if operator_directives:
        parts.append("## Operator Directives")
        parts.append(
            "Apply these operator directives where they don't conflict with repair constraints:"
        )
        parts.append(operator_directives)
        parts.append("")

    if planner_system_prompt:
        parts.append("## System Instructions")
        parts.append(planner_system_prompt)
        parts.append("")

    parts.append("## Repair Task")
    parts.append(
        "Repair an invalid research plan. Patch only the broken parts and keep the overall intent."
    )
    parts.append("")
    parts.append("## Goal")
    parts.append(truncate_text(repair_request.goal, 500))
    parts.append("")

    context_text = _planner_context(research_context)
    if context_text:
        parts.append("## Context")
        parts.append(truncate_text(context_text, 3200))
        parts.append("")

    if mcp_problem_summary:
        parts.append("## Known MCP Problems")
        parts.append(mcp_problem_summary)
        parts.append("")

    if worker_ids:
        shuffled = list(worker_ids)
        random.shuffle(shuffled)
        parts.append("## Available Workers")
        parts.append(", ".join(shuffled))
        parts.append("")

    parts.append("## Repair Rules")
    parts.append(
        "- Preserve valid stages and the overall investigation direction\n"
        "- Only repair the invalid stages/instructions listed below\n"
        "- Use `depends_on` as the only dependency contract\n"
        "- **Preserve the parallel branch structure** — keep depends_on values of valid stages unchanged\n"
        "- Re-emit the full plan in schema v4 with `steps`\n"
        "- Use {{step:step_id.run_id}} for previous steps inside the same stage\n"
        "- Do NOT use cross-stage refs like {{stage:N.run_id}} — the orchestrator resolves those at dispatch time\n"
        "- Use concrete IDs only when they are already known from context\n"
        "- NEVER emit <...> placeholders, ellipses, or incomplete tool calls\n"
        "- Replace tool aliases with canonical facades from the MCP contract\n"
        f"- Keep the existing stage count; do not expand beyond {len(repair_request.invalid_plan_data.get('tasks', []))} stages\n"
        "- Respond with a COMPLETE corrected plan JSON, not a patch diff\n"
        "- backtests_runs(action='start') requires `version` — use '1' if not known from context\n"
        "- The instruction text in a step is informational; tool_name and args are what execute. Ensure tool_name and args are complete, even if instruction is brief"
    )
    parts.append("")

    parts.append("## Canonical Repair Guidance")
    parts.append("- Replace self stage refs with step refs inside the same stage")
    parts.append("- Replace `snapshots(...)` with `backtests_strategy(action='clone', ...)`")
    parts.append("- Replace `backtests_runs(action='run', ...)` with canonical `backtests_runs(action='start', ...)`")
    parts.append("- Use only the tool names, actions, and arg shapes from the MCP contract in Context")
    parts.append("")

    parts.append("## Validation Errors To Fix")
    for error in repair_request.validation_errors:
        parts.append(
            f"- stage {error.stage_number} | code={error.code} | {error.message} | text={error.offending_text[:200]}"
        )
    parts.append("")

    if valid_summary:
        parts.append("## Valid Stage Summary")
        parts.append(valid_summary)
        parts.append("")

    parts.append("## Invalid Stage Fragments")
    parts.append("```json")
    parts.append(invalid_payload)
    parts.append("```")
    parts.append("")

    parts.append("## Output")
    parts.append("Respond with JSON only matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")
    parts.append("")
    parts.append("`plan_markdown` must be concise and summarize the corrected plan rather than duplicating every task detail.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Worker prompts
# ---------------------------------------------------------------------------

def build_plan_task_prompt(
    stage_number: int,
    stage_name: str,
    theory: str,
    agent_instructions: list[str],
    steps: list[Any] | None = None,
    results_table_columns: list[str] | None = None,
    plan_version: int = 0,
    mcp_instructions: str | None = None,
    dependency_reports: list[TaskReport] | None = None,
) -> str:
    """Build the prompt for a worker executing a single plan task.

    This replaces the thin build_worker_prompt() with a structured task
    that includes theory context and specific result requirements.
    """
    parts: list[str] = []

    parts.append(f"# ETAP {stage_number}: {stage_name}")
    if plan_version:
        parts.append(f"(from plan_v{plan_version})")
    parts.append("")

    if theory:
        parts.append("## Theory")
        parts.append(theory)
        parts.append("")

    # Previous stage results from dependency reports
    if dependency_reports:
        parts.append("## Previous Stage Results")
        parts.append(
            "The following reports are from earlier completed stages that this stage depends on. "
            "Use concrete IDs from these reports (e.g. run_id, snapshot_id) when needed."
        )
        parts.append("")
        for dep_report in dependency_reports:
            parts.append(f"### Stage Report (task_id={dep_report.task_id})")
            if dep_report.verdict:
                parts.append(f"- Verdict: {dep_report.verdict}")
            if dep_report.results_table and isinstance(dep_report.results_table[0], dict):
                cols = list(dep_report.results_table[0].keys())
                if cols:
                    parts.append("| " + " | ".join(cols) + " |")
                    parts.append("| " + " | ".join("---" for _ in cols) + " |")
                    for row in dep_report.results_table:
                        parts.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
                    parts.append("")
            if dep_report.key_metrics:
                parts.append(f"- Key metrics: {dep_report.key_metrics}")
            if dep_report.artifacts:
                parts.append(f"- Artifacts: {', '.join(str(a) for a in dep_report.artifacts)}")
            parts.append("")

    parts.append("## Workflow")
    parts.append("Execute steps in order. Resolve `{{step:...}}` using outputs from earlier steps in this ETAP. Do not guess missing values.")
    parts.append("")
    if steps:
        for i, step in enumerate(steps, 1):
            step_id = getattr(step, "step_id", "") or step.get("step_id", f"step_{i}")
            kind = getattr(step, "kind", "") or step.get("kind", "work")
            instruction = getattr(step, "instruction", "") or step.get("instruction", "")
            tool_name = getattr(step, "tool_name", None) if not isinstance(step, dict) else step.get("tool_name")
            args = getattr(step, "args", {}) if not isinstance(step, dict) else step.get("args", {})
            binds = getattr(step, "binds", []) if not isinstance(step, dict) else step.get("binds", [])
            decision_outputs = getattr(step, "decision_outputs", []) if not isinstance(step, dict) else step.get("decision_outputs", [])
            notes = getattr(step, "notes", "") if not isinstance(step, dict) else step.get("notes", "")
            parts.append(f"{i}. [{step_id}] {kind}")
            if instruction:
                parts.append(f"   - instruction: {instruction}")
            if tool_name:
                parts.append(f"   - tool_name: {tool_name}")
                parts.append(f"   - args: {json.dumps(args, ensure_ascii=False)}")
            if binds:
                parts.append(f"   - binds: {', '.join(str(x) for x in binds)}")
            if decision_outputs:
                parts.append(f"   - decision_outputs: {', '.join(str(x) for x in decision_outputs)}")
            if notes:
                parts.append(f"   - notes: {notes}")
    else:
        for i, step in enumerate(agent_instructions, 1):
            parts.append(f"{i}. {step}")
    parts.append("")

    # Results table to fill
    if results_table_columns:
        parts.append("## Results Table")
        parts.append(
            "You MUST fill in this results table. Report one row per experiment/run. "
            "Each row must have values for ALL columns."
        )
        parts.append("")
        cols = results_table_columns
        parts.append("| " + " | ".join(cols) + " |")
        parts.append("| " + " | ".join("---" for _ in cols) + " |")
        parts.append("")

    # MCP instructions
    if mcp_instructions:
        parts.append(mcp_instructions)
        parts.append("")

    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(WORKER_REPORT_SCHEMA)
    parts.append("```")
    parts.append("")
    parts.append(
        "CRITICAL:\n"
        "- Respond with JSON ONLY. Do NOT include preamble text, markdown commentary, or code fences.\n"
        "- The first non-whitespace character must be `{` and the last must be `}`.\n"
        "- `what_was_done` must be a DETAILED description of everything you did\n"
        "- `results_table` must contain one row per experiment/run with ALL columns filled\n"
        "- `key_metrics` must include the most important numbers from your work\n"
        "- `verdict` must be your honest assessment: PROMOTE if results clearly pass decision gates, "
        "REJECT if they clearly fail, WATCHLIST if uncertain\n"
    )
    parts.append(
        "INFRASTRUCTURE TOLERANCE:\n"
        "- If this is an infrastructure validation stage (Stage 0 / ETAP 0), "
        "async operations like datasets_sync or data refresh may time out — "
        "this is EXPECTED and should NOT cause a REJECT verdict.\n"
        "- A PROMOTE verdict is appropriate if core tools (backtests_runs, "
        "features_catalog, models_registry) are accessible, even if sync "
        "or refresh operations time out.\n"
        "- Snapshot count mismatches or stale caches are not errors — "
        "note them but still report PROMOTE if the research pipeline is functional.\n"
    )

    return "\n".join(parts)
