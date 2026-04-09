"""
Cheap broker-side semantic guards for known dev_space1 tool contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.broker.contract_traps import AppliedToolRewrite


@dataclass(frozen=True)
class SemanticContractError(ValueError):
    tool_name: str
    reason_code: str
    message: str

    def __str__(self) -> str:
        return self.message


def apply_semantic_contract_rules(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[AppliedToolRewrite]]:
    normalized = dict(arguments)
    rewrites: list[AppliedToolRewrite] = []

    if tool_name in {"research_search", "research_map"}:
        project_id = str(normalized.get("project_id", "") or "").strip()
        if not project_id:
            raise SemanticContractError(
                tool_name=tool_name,
                reason_code=f"{tool_name}_project_id_required",
                message=f"{tool_name} requires a non-empty project_id; broker rejected the call locally.",
            )

    if tool_name in {"events", "events_sync"}:
        family = str(normalized.get("family", "") or "").strip().lower()
        symbol = str(normalized.get("symbol", "") or "").strip()
        if family == "funding" and not symbol:
            raise SemanticContractError(
                tool_name=tool_name,
                reason_code=f"{tool_name}_funding_symbol_required",
                message=f"{tool_name} for family='funding' requires symbol; broker rejected the call locally.",
            )

    if tool_name == "features_cleanup":
        scope = str(normalized.get("scope", "") or "").strip()
        if scope == "registry":
            rewritten = dict(normalized)
            rewritten["scope"] = "features_only"
            rewrites.append(
                AppliedToolRewrite(
                    tool_name=tool_name,
                    reason_code="features_cleanup_registry_scope_rewritten",
                    original_arguments=dict(normalized),
                    rewritten_arguments=dict(rewritten),
                    summary="features_cleanup scope='registry' is not accepted by runtime; broker rewrote it to scope='features_only'.",
                )
            )
            normalized = rewritten

    if tool_name == "features_analytics":
        action = str(normalized.get("action", "") or "").strip().lower()
        if action in {"analytics", "heatmap", "render", "portability"}:
            selectors = ("feature_name", "feature", "column_name", "column", "name")
            if not any(str(normalized.get(field, "") or "").strip() for field in selectors):
                raise SemanticContractError(
                    tool_name=tool_name,
                    reason_code="features_analytics_feature_selector_required",
                    message=(
                        "features_analytics requires one concrete feature selector for "
                        "analytics/heatmap/render/portability; broker rejected the call locally."
                    ),
                )

    return normalized, rewrites


__all__ = ["SemanticContractError", "apply_semantic_contract_rules"]
