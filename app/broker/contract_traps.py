"""
Deterministic broker-side rewrites for known dev_space1 contract traps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.execution_models import ToolPolicy


@dataclass(frozen=True)
class AppliedToolRewrite:
    tool_name: str
    reason_code: str
    original_arguments: dict[str, Any]
    rewritten_arguments: dict[str, Any]
    summary: str


def apply_known_contract_traps(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    policy: ToolPolicy,
    supports_wait: bool,
) -> tuple[dict[str, Any], list[AppliedToolRewrite]]:
    normalized = dict(arguments)
    rewrites: list[AppliedToolRewrite] = []

    level = str(normalized.get("level", "") or "").strip().lower()
    if tool_name == "research_search" and level == "compact":
        rewritten = dict(normalized)
        rewritten["level"] = "normal"
        rewrites.append(
            AppliedToolRewrite(
                tool_name=tool_name,
                reason_code="research_search_compact_level_rewritten",
                original_arguments=dict(normalized),
                rewritten_arguments=dict(rewritten),
                summary="research_search level='compact' is advertised by schema/docs but rejected by runtime; broker rewrote it to level='normal'.",
            )
        )
        normalized = rewritten

    wait = str(normalized.get("wait", "") or "").strip().lower()
    if supports_wait and policy.expensive and wait == "completed":
        rewritten = dict(normalized)
        rewritten["wait"] = "started"
        rewrites.append(
            AppliedToolRewrite(
                tool_name=tool_name,
                reason_code="expensive_async_wait_completed_rewritten",
                original_arguments=dict(normalized),
                rewritten_arguments=dict(rewritten),
                summary=f"{tool_name} wait='completed' is blocked for expensive async tools; broker rewrote it to wait='started'.",
            )
        )
        normalized = rewritten

    return normalized, rewrites


__all__ = ["AppliedToolRewrite", "apply_known_contract_traps"]
