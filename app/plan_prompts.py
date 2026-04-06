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
from app.plan_validation import PlanRepairRequest


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

PLANNER_PLAN_SCHEMA = """{
  "schema_version": 2,
  "plan_action": "create|update",
  "plan_version": 1,
  "reason": "why this plan or revision",
  "plan_markdown": "the full structured plan document as a single markdown string",
  "frozen_base": "immutable reference identifier",
  "baseline_run_id": "measured baseline run id if known",
  "baseline_snapshot_ref": "snapshot_id@version for measured baseline",
  "baseline_metrics": {"net_pnl": 0.0, "sharpe": 0.0, "trades": 0, "max_drawdown_pct": 0.0},
  "tasks": [
    {
      "stage_number": 0,
      "stage_name": "Stage name",
      "theory": "2-5 sentences: why this hypothesis is worth testing",
      "depends_on": [],
      "agent_instructions": ["specific step 1 with exact tool call", "specific step 2 using {{stage:0.run_id}} if needed"],
      "results_table_columns": ["run_id", "net_pnl", "trades", "PF", "WR", "max_DD", "verdict"],
      "decision_gates": [
        {"metric": "pnl", "threshold": 0, "comparator": "gt", "verdict_pass": "PROMOTE", "verdict_fail": "REJECT"}
      ]
    }
  ],
  "anti_patterns_new": [
    {"category": "approach name", "description": "what was tried", "evidence_count": 3, "evidence_summary": "key results"}
  ],
  "cumulative_summary": "updated knowledge summary carrying forward from all previous plans",
  "principles": ["principle 1", "principle 2"],
  "memory_update": "brief note to remember",
  "check_after_seconds": 300,
  "should_finish": false
}"""

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
) -> str:
    """Build the prompt for creating a new research plan (plan_v1 or after full resolution).

    This is the main prompt that turns the planner into a plan author.
    """
    parts: list[str] = []

    parts.append("## Your Role")
    parts.append(
        "You are the research director. Your job is to write a STRUCTURED RESEARCH PLAN "
        "that will be executed by worker agents. The plan must contain MULTIPLE numbered "
        "stages (ETAPs), each with theory, specific instructions, results table schema, "
        "and quantitative decision gates."
    )
    parts.append("")

    parts.append("## Global Goal")
    parts.append(goal)
    parts.append("")

    # Research context (MCP state, baseline metrics, available tools)
    if research_context:
        parts.append(research_context)
        parts.append("")

    # Cumulative knowledge from all previous plans
    if cumulative_summary:
        parts.append("## Cumulative Knowledge (from all previous plan versions)")
        parts.append(cumulative_summary)
        parts.append("")

    # Anti-patterns — what NOT to do
    if anti_patterns:
        parts.append("## Anti-Patterns (DO NOT repeat these approaches)")
        parts.append(
            "These approaches have been PROVEN to fail. Never include them in your plan:"
        )
        parts.append("")
        for ap in anti_patterns:
            parts.append(
                f"- **{ap.get('category', '?')}**: {ap.get('description', '')} "
                f"({ap.get('evidence_count', '?')} failures — {ap.get('evidence_summary', '')})"
            )
        parts.append("")

    # Previous plan (for revision context)
    if previous_plan_markdown:
        parts.append("## Previous Plan")
        parts.append(previous_plan_markdown)
        parts.append("")

    # Known MCP problems
    if mcp_problem_summary:
        parts.append("## Known MCP Problems (avoid these mistakes)")
        parts.append(mcp_problem_summary)
        parts.append("")

    # Workers available
    if worker_ids:
        shuffled = list(worker_ids)
        random.shuffle(shuffled)
        parts.append("## Available Workers")
        parts.append(", ".join(shuffled))
        parts.append("Distribute tasks across workers for parallelism.")
        parts.append("")

    # Plan structure instructions
    parts.append("## Plan Structure Requirements")
    parts.append(
        "Your plan MUST follow this structure:\n"
        "\n"
        "1. **Header**: Frozen base reference (immutable), goal of this version\n"
        "2. **Principles**: 3-5 rules that govern this plan version\n"
        "3. **Cumulative Summary**: What has been proven so far (carry forward + update)\n"
        "4. **Anti-Patterns**: What categorically does NOT work (add new ones if evidence exists)\n"
        "5. **Numbered Stages (ETAP 0, ETAP 1, ...)**, each with:\n"
        "   - **Theory**: 2-5 sentences on why this hypothesis is worth testing\n"
        "   - **Depends On**: list of prerequisite stage numbers that must resolve first\n"
        "   - **Agent Instructions**: Numbered, SPECIFIC steps with EXACT MCP tool calls,\n"
        "     snapshot IDs, parameters. Workers need exact tool_name(action='...', param=value) syntax.\n"
        "   - **Results Table Columns**: Define what metrics the worker must report\n"
        "   - **Decision Gates**: Quantitative accept/reject criteria\n"
        "6. **Dependencies / Parallelism**: Stages with satisfied dependencies may run in parallel\n"
    )
    parts.append("")

    parts.append("## Key Principles for Good Plans")
    parts.append(
        "- Each stage should be a SUBSTANTIAL task (not a single tool call, but a coherent investigation)\n"
        "- Give workers EXACT parameters: snapshot IDs, feature names, timeframes, thresholds\n"
        "- Use symbolic refs ONLY for future outputs: {{stage:N.run_id}}, {{stage:N.snapshot_ref}}, {{stage:N.results_table[0].column}}\n"
        "- NEVER emit placeholders like <best_snapshot_id>, ellipses, or incomplete tool calls\n"
        "- Include CODE SNIPPETS for signal logic when applicable\n"
        "- Define clear VERDICT criteria: PROMOTE (confirmed improvement), WATCHLIST (promising but needs more evidence), REJECT (failed)\n"
        "- Group related hypotheses into families\n"
        "- Never repeat approaches from the anti-patterns list\n"
        "- Reference the frozen base explicitly when comparing results\n"
    )
    parts.append("")
    parts.append(
        "## Decision Gate Best Practices\n"
        "- NEVER use comparator='eq' with threshold=1.0 for tool_success_rate — "
        "async operations like datasets_sync time out regularly.\n"
        "- Use comparator='gte' with threshold=0.7 or higher for success rates.\n"
        "- Infrastructure stages (Stage 0) should have permissive gates: "
        "if the core research pipeline is functional, PROMOTE.\n"
    )
    parts.append("")

    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")
    parts.append("")
    parts.append(
        "IMPORTANT: The `plan_markdown` field must contain the FULL plan document as "
        "a single string (use \\n for line breaks). This is saved as plan_vN.md for "
        "future reference. The `tasks` array provides structured metadata for dispatch. "
        "Each task MUST include explicit `depends_on` stage numbers. A task with "
        "`depends_on: []` is ready immediately. Tasks whose dependencies are satisfied "
        "may run in parallel."
    )
    parts.append("")
    parts.append("## Symbolic Reference Contract")
    parts.append(
        "- Allowed future-output syntax: {{stage:N.run_id}}, {{stage:N.snapshot_id}}, "
        "{{stage:N.version}}, {{stage:N.snapshot_ref}}, {{stage:N.results_table[0].column_name}}\n"
        "- Symbolic refs MUST point to an earlier dependency listed in `depends_on`\n"
        "- Do NOT use <run_id>, <v>, <snapshot_id>, or `...`\n"
        "- Use concrete IDs only when they are already known from the provided context"
    )
    parts.append("")
    parts.append("## Examples")
    parts.append("Valid: backtests_runs(action='inspect', run_id='{{stage:0.run_id}}', view='detail')")
    parts.append("Invalid: backtests_runs(action='inspect', run_id='<run_id>', view='detail')")

    return "\n".join(parts)


def build_plan_revision_prompt(
    goal: str,
    current_plan: ResearchPlan,
    reports: list[TaskReport],
    research_context: str | None = None,
    anti_patterns: list[dict[str, Any]] | None = None,
    worker_ids: list[str] | None = None,
    mcp_problem_summary: str | None = None,
) -> str:
    """Build the prompt for revising a plan based on collected worker reports.

    The planner receives the current plan + all reports and generates the NEXT version.
    """
    parts: list[str] = []

    parts.append("## Your Role")
    parts.append(
        "You are the research director. Workers have completed tasks from the current plan "
        "and returned reports. Your job is to analyze the results and write the NEXT VERSION "
        "of the research plan (plan_v" + str(current_plan.version + 1) + ")."
    )
    parts.append("")

    parts.append("## Global Goal")
    parts.append(goal)
    parts.append("")

    if research_context:
        parts.append(research_context)
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
        parts.append(current_plan.plan_markdown)
    else:
        parts.append(f"Goal: {current_plan.goal}")
        parts.append(f"Frozen base: {current_plan.frozen_base}")
    parts.append("")

    # Worker reports
    parts.append("## Worker Reports")
    parts.append(
        "These are the results from workers who executed tasks in the current plan:"
    )
    parts.append("")
    for report in reports:
        parts.append(f"### Task Report (stage from plan_v{report.plan_version})")
        parts.append(f"- **Worker**: {report.worker_id}")
        parts.append(f"- **Status**: {report.status}")
        parts.append(f"- **Confidence**: {report.confidence}")
        parts.append(f"- **Verdict**: {report.verdict}")
        parts.append("")
        if report.what_was_requested:
            parts.append(f"**What was requested**: {report.what_was_requested}")
            parts.append("")
        if report.what_was_done:
            parts.append(f"**What was done**: {report.what_was_done}")
            parts.append("")
        if report.results_table:
            parts.append("**Results Table**:")
            # Render as markdown table
            if report.results_table:
                cols = list(report.results_table[0].keys())
                parts.append("| " + " | ".join(cols) + " |")
                parts.append("| " + " | ".join("---" for _ in cols) + " |")
                for row in report.results_table:
                    parts.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
            parts.append("")
        if report.key_metrics:
            parts.append("**Key Metrics**:")
            for k, v in report.key_metrics.items():
                parts.append(f"- {k}: {v}")
            parts.append("")
        if report.error:
            parts.append(f"**Error**: {report.error}")
            parts.append("")
        if report.artifacts:
            parts.append(f"**Artifacts**: {', '.join(report.artifacts)}")
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
        "7. **Dependencies**: Use `depends_on` as the only execution contract. "
        "Stages whose dependencies are resolved may run in parallel.\n"
        "8. **Future outputs**: Use symbolic refs like {{stage:0.run_id}} instead of "
        "inventing future run IDs or version placeholders.\n"
    )
    parts.append("")

    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")

    return "\n".join(parts)


def build_plan_repair_prompt(
    repair_request: PlanRepairRequest,
    research_context: str | None = None,
    worker_ids: list[str] | None = None,
    mcp_problem_summary: str | None = None,
) -> str:
    """Build the prompt for repairing one invalid planner output."""
    parts: list[str] = []

    parts.append("## Your Role")
    parts.append(
        "You are repairing an INVALID research plan. Keep the plan structure and intent, "
        "but patch only the invalid parts so the orchestrator can execute it."
    )
    parts.append("")
    parts.append("## Global Goal")
    parts.append(repair_request.goal)
    parts.append("")

    if research_context:
        parts.append(research_context)
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
        "- Use symbolic refs like {{stage:N.run_id}} for future outputs\n"
        "- NEVER emit <...> placeholders, ellipses, or incomplete tool calls\n"
        "- Respond with a COMPLETE corrected plan JSON, not a patch diff"
    )
    parts.append("")

    parts.append("## Validation Errors To Fix")
    for error in repair_request.validation_errors:
        parts.append(
            f"- stage {error.stage_number} | code={error.code} | {error.message} | text={error.offending_text[:200]}"
        )
    parts.append("")

    parts.append("## Invalid Plan Payload")
    parts.append("```json")
    parts.append(json.dumps(repair_request.invalid_plan_data, ensure_ascii=False, indent=2))
    parts.append("```")
    parts.append("")

    parts.append("## Required Output Format")
    parts.append("Respond ONLY with a JSON object matching this schema:")
    parts.append("```json")
    parts.append(PLANNER_PLAN_SCHEMA)
    parts.append("```")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Worker prompts
# ---------------------------------------------------------------------------

def build_plan_task_prompt(
    stage_number: int,
    stage_name: str,
    theory: str,
    agent_instructions: list[str],
    results_table_columns: list[str] | None = None,
    plan_version: int = 0,
    mcp_instructions: str | None = None,
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

    parts.append("## Instructions")
    parts.append("Execute these steps in order:")
    parts.append("")
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
