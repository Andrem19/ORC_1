"""
Planner context builder for plan-mode prompts.
"""

from __future__ import annotations

from typing import Any

from app.planner_contract import summarize_templates
from app.research_context import MCP_TOOL_CATALOG


def build_planner_context(
    *,
    research_context: str | None,
    baseline_bootstrap: dict[str, Any] | None = None,
) -> str:
    sections: list[str] = []

    sections.append(MCP_TOOL_CATALOG)
    sections.append("")
    sections.append(summarize_templates())

    if baseline_bootstrap:
        sections.append("")
        sections.append("## Baseline Bootstrap")
        snapshot_id = baseline_bootstrap.get("baseline_snapshot_id")
        version = baseline_bootstrap.get("baseline_version")
        symbol = baseline_bootstrap.get("symbol")
        anchor = baseline_bootstrap.get("anchor_timeframe")
        execution = baseline_bootstrap.get("execution_timeframe")
        if snapshot_id is not None:
            sections.append(f"- snapshot_id: {snapshot_id}")
        if version is not None:
            sections.append(f"- version: {version}")
        if symbol is not None:
            sections.append(f"- symbol: {symbol}")
        if anchor is not None:
            sections.append(f"- anchor_timeframe: {anchor}")
        if execution is not None:
            sections.append(f"- execution_timeframe: {execution}")

    if research_context:
        sections.append("")
        sections.append("## Live Research Context")
        sections.append(research_context)

    return "\n".join(section for section in sections if section)
