"""
Diffing helpers for MCP catalog snapshots.
"""

from __future__ import annotations

import json

from app.services.mcp_catalog.models import McpCatalogDiff, McpCatalogSnapshot


def diff_catalog_snapshots(
    previous: McpCatalogSnapshot | None,
    current: McpCatalogSnapshot,
) -> McpCatalogDiff:
    if previous is None:
        return McpCatalogDiff(
            previous_hash="",
            current_hash=current.schema_hash,
            added_tools=current.tool_names(),
            removed_tools=[],
            changed_tools=[],
            unchanged=False,
            summary=f"initial catalog loaded with {len(current.tools)} tools",
        )
    prev_map = {tool.name: tool.contract_signature() for tool in previous.tools}
    curr_map = {tool.name: tool.contract_signature() for tool in current.tools}
    prev_names = set(prev_map)
    curr_names = set(curr_map)
    added = sorted(curr_names - prev_names)
    removed = sorted(prev_names - curr_names)
    changed = sorted(
        name
        for name in (prev_names & curr_names)
        if json.dumps(prev_map[name], ensure_ascii=False, sort_keys=True)
        != json.dumps(curr_map[name], ensure_ascii=False, sort_keys=True)
    )
    unchanged = not added and not removed and not changed and previous.schema_hash == current.schema_hash
    summary_bits: list[str] = []
    if added:
        summary_bits.append(f"added={len(added)}")
    if removed:
        summary_bits.append(f"removed={len(removed)}")
    if changed:
        summary_bits.append(f"changed={len(changed)}")
    if not summary_bits:
        summary_bits.append("no contract changes")
    return McpCatalogDiff(
        previous_hash=previous.schema_hash,
        current_hash=current.schema_hash,
        added_tools=added,
        removed_tools=removed,
        changed_tools=changed,
        unchanged=unchanged,
        summary=", ".join(summary_bits),
    )


__all__ = ["diff_catalog_snapshots"]
