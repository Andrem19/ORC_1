"""
Generic runtime profile derivation for direct slices.
"""

from __future__ import annotations

from app.services.mcp_catalog.models import McpCatalogSnapshot

_RESEARCH_SETUP_TOOLS = frozenset({"research_project", "research_map", "research_memory"})
_RESEARCH_SETUP_PREREQUISITES: list[str] = []
_RESEARCH_SETUP_REQUIRED_FACTS = [
    "research.project_id",
    "research.baseline_configured",
    "research.atlas_defined",
    "research.invariants_recorded",
    "research.naming_recorded",
]
_RESEARCH_SHORTLIST_PREREQUISITES: list[str] = []
_RESEARCH_SHORTLIST_REQUIRED_FACTS = [
    "research.project_id",
    "research.shortlist_families",
    "research.novelty_justification_present",
]
_BACKTESTS_STABILITY_PREREQUISITES: list[str] = []
_BACKTESTS_STABILITY_REQUIRED_FACTS: list[str] = []
_BACKTESTS_INTEGRATION_PREREQUISITES: list[str] = []
_BACKTESTS_INTEGRATION_REQUIRED_FACTS: list[str] = []
_BACKTESTS_CANNIBALIZATION_PREREQUISITES: list[str] = []
_STRICT_RUNTIME_CONTRACTS: dict[str, tuple[list[str], list[str], str]] = {
    "catalog_contract_probe": ([], ["features_catalog.scopes"], "fact_based"),
    "research_setup": (list(_RESEARCH_SETUP_PREREQUISITES), list(_RESEARCH_SETUP_REQUIRED_FACTS), "fact_based"),
    "research_shortlist": (list(_RESEARCH_SHORTLIST_PREREQUISITES), list(_RESEARCH_SHORTLIST_REQUIRED_FACTS), "fact_based"),
    "backtests_stability_analysis": (
        list(_BACKTESTS_STABILITY_PREREQUISITES),
        list(_BACKTESTS_STABILITY_REQUIRED_FACTS),
        "none",
    ),
    "backtests_integration_analysis": (
        list(_BACKTESTS_INTEGRATION_PREREQUISITES),
        list(_BACKTESTS_INTEGRATION_REQUIRED_FACTS),
        "none",
    ),
    "backtests_cannibalization_analysis": (
        list(_BACKTESTS_CANNIBALIZATION_PREREQUISITES),
        [],
        "none",
    ),
}
_SHORTLIST_MARKERS = (
    "shortlist",
    "short-list",
    "novelty",
    "new class",
    "new family",
    "not duplicate",
    "not a duplicate",
    "history v1-v12",
    "v1-v12",
    "first wave",
    "wave 1",
)
_RESEARCH_CONTEXT_TOOLS = frozenset({"research_map", "research_memory", "research_record"})
_BACKTESTS_ANALYSIS_TOOLS = frozenset({"backtests_conditions", "backtests_analysis", "backtests_runs"})


def derive_runtime_slice_metadata(
    *,
    allowed_tools: list[str] | set[str],
    catalog_snapshot: McpCatalogSnapshot | None,
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | None = None,
    policy_tags: list[str] | None = None,
) -> tuple[str, list[str], str]:
    if catalog_snapshot is None:
        return "generic_read", [], "generic_salvage"

    tool_names = [str(item).strip() for item in allowed_tools if str(item).strip()]
    tools = [catalog_snapshot.get_tool(tool_name) for tool_name in tool_names]
    live_tools = [tool for tool in tools if tool is not None]
    if not live_tools:
        return "generic_read", [], "generic_salvage"
    live_tool_name_set = {tool.name for tool in live_tools if str(tool.name or "").strip()}

    all_discovery = all(tool.supports_discovery and tool.side_effects == "read_only" for tool in live_tools)
    any_terminal_write = any(tool.supports_terminal_write for tool in live_tools)
    any_polling = any(tool.supports_polling or tool.async_like for tool in live_tools)
    any_catalog_like = any(
        "scope" in tool.fields and ("catalog" in tool.description.lower() or "timeframe" in tool.fields)
        for tool in live_tools
    )
    any_project_handles = any("project_id" in tool.accepted_handle_fields for tool in live_tools)
    any_result_handles = any(
        field in {"job_id", "run_id", "operation_id", "snapshot_id"}
        for tool in live_tools
        for field in tool.accepted_handle_fields + tool.produced_handle_fields
    )
    any_research_memory_write = any(_is_research_memory_write_tool(tool.name, tool.family, tool.supports_terminal_write) for tool in live_tools)
    any_research_mapping = any(str(tool.name or "").strip() in {"research_project", "research_map"} for tool in live_tools)

    if any_catalog_like and all_discovery:
        return "catalog_contract_probe", ["features_catalog.scopes"], "fact_based"
    backtests_profile = _derive_backtests_runtime_profile(
        tool_names=live_tool_name_set,
        title=title,
        objective=objective,
        success_criteria=success_criteria or [],
        policy_tags=policy_tags or [],
    )
    if backtests_profile:
        required_facts = {
            "backtests_stability_analysis": list(_BACKTESTS_STABILITY_REQUIRED_FACTS),
            "backtests_integration_analysis": list(_BACKTESTS_INTEGRATION_REQUIRED_FACTS),
            "backtests_cannibalization_analysis": [],
        }[backtests_profile]
        return backtests_profile, required_facts, "none"
    if any_research_memory_write and any_research_mapping and _looks_like_research_shortlist_intent(
        title=title,
        objective=objective,
        success_criteria=success_criteria or [],
        policy_tags=policy_tags or [],
    ):
        return "research_shortlist", list(_RESEARCH_SHORTLIST_REQUIRED_FACTS), "fact_based"
    if _RESEARCH_SETUP_TOOLS.issubset(live_tool_name_set):
        return "research_setup", list(_RESEARCH_SETUP_REQUIRED_FACTS), "fact_based"
    if any_project_handles and all_discovery:
        return "research_memory", [], "generic_salvage"
    if any_polling and any_result_handles:
        return "async_observe", [], "generic_salvage"
    if any_terminal_write and _supports_research_write_result(live_tools):
        return "write_result", [], "none"
    if any(tool.side_effects == "mutating" for tool in live_tools):
        return "generic_mutation", [], "none"
    return "generic_read", [], "generic_salvage"


def resolve_runtime_slice_metadata(
    *,
    runtime_profile: str,
    required_output_facts: list[str] | None,
    finalization_mode: str,
    allowed_tools: list[str] | set[str],
    catalog_snapshot: McpCatalogSnapshot | None,
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | None = None,
    policy_tags: list[str] | None = None,
) -> tuple[str, list[str], str]:
    return _resolve_runtime_slice_metadata(
        runtime_profile=runtime_profile,
        required_output_facts=required_output_facts,
        required_prerequisite_facts=None,
        finalization_mode=finalization_mode,
        allowed_tools=allowed_tools,
        catalog_snapshot=catalog_snapshot,
        title=title,
        objective=objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    )[:3]


def resolve_runtime_slice_metadata_with_prerequisites(
    *,
    runtime_profile: str,
    required_output_facts: list[str] | None,
    required_prerequisite_facts: list[str] | None,
    finalization_mode: str,
    allowed_tools: list[str] | set[str],
    catalog_snapshot: McpCatalogSnapshot | None,
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | None = None,
    policy_tags: list[str] | None = None,
) -> tuple[str, list[str], list[str], str]:
    return _resolve_runtime_slice_metadata(
        runtime_profile=runtime_profile,
        required_output_facts=required_output_facts,
        required_prerequisite_facts=required_prerequisite_facts,
        finalization_mode=finalization_mode,
        allowed_tools=allowed_tools,
        catalog_snapshot=catalog_snapshot,
        title=title,
        objective=objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    )


def derive_slice_acceptance_policy(
    *,
    runtime_profile: str,
    allowed_tools: list[str] | set[str],
    policy_tags: list[str] | None = None,
) -> dict[str, bool | str]:
    tool_names = {str(item).strip() for item in allowed_tools if str(item).strip()}
    normalized_tags = {
        str(item or "").strip().lower()
        for item in (policy_tags or [])
        if str(item or "").strip()
    }
    is_backtests_execution = "backtests_runs" in tool_names and "backtests_analysis" not in tool_names
    return {
        "dependency_unblock_mode": "advisory_only" if {"advisory_only", "advisory_unblock"} & normalized_tags else "accepted_only",
        "watchlist_allows_unblock": "watchlist_allows_unblock" in normalized_tags,
        "requires_mutating_evidence": str(runtime_profile or "").strip() == "write_result",
        "requires_persisted_artifact": is_backtests_execution,
        "requires_live_handle_validation": is_backtests_execution,
    }


def _resolve_runtime_slice_metadata(
    *,
    runtime_profile: str,
    required_output_facts: list[str] | None,
    required_prerequisite_facts: list[str] | None,
    finalization_mode: str,
    allowed_tools: list[str] | set[str],
    catalog_snapshot: McpCatalogSnapshot | None,
    title: str = "",
    objective: str = "",
    success_criteria: list[str] | None = None,
    policy_tags: list[str] | None = None,
) -> tuple[str, list[str], list[str], str]:
    derived_profile, derived_required_facts, derived_finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=allowed_tools,
        catalog_snapshot=catalog_snapshot,
        title=title,
        objective=objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    )
    explicit_profile = str(runtime_profile or "").strip()
    strict_profile = derived_profile if derived_profile in _STRICT_RUNTIME_CONTRACTS else explicit_profile if explicit_profile in _STRICT_RUNTIME_CONTRACTS else ""
    explicit_facts = [str(item).strip() for item in list(required_output_facts or []) if str(item).strip()]
    explicit_prerequisites = [str(item).strip() for item in list(required_prerequisite_facts or []) if str(item).strip()]
    if strict_profile:
        strict_prerequisites, strict_required_facts, strict_finalization_mode = _STRICT_RUNTIME_CONTRACTS[strict_profile]
        return (
            strict_profile,
            list(explicit_facts or strict_required_facts),
            list(explicit_prerequisites or strict_prerequisites),
            str(strict_finalization_mode),
        )
    profile = derived_profile or explicit_profile
    facts = explicit_facts or list(derived_required_facts)
    prerequisites = explicit_prerequisites or _default_prerequisites_for_profile(profile, allowed_tools=allowed_tools)
    finalization = derived_finalization_mode or str(finalization_mode or "").strip()
    return profile, facts, prerequisites, finalization


def _default_prerequisites_for_profile(profile: str, *, allowed_tools: list[str] | set[str]) -> list[str]:
    return []


_BACKTESTS_PROFILE_TAGS: dict[str, str] = {
    "cannibalization": "backtests_cannibalization_analysis",
    "cannibalization_analysis": "backtests_cannibalization_analysis",
    "ownership": "backtests_cannibalization_analysis",
    "integration": "backtests_integration_analysis",
    "integration_analysis": "backtests_integration_analysis",
    "stability": "backtests_stability_analysis",
    "stability_analysis": "backtests_stability_analysis",
}


def _derive_backtests_runtime_profile(
    *,
    tool_names: set[str],
    title: str,
    objective: str,
    success_criteria: list[str],
    policy_tags: list[str],
) -> str:
    del title, objective, success_criteria
    if "research_memory" not in tool_names:
        return ""
    if not (tool_names & _BACKTESTS_ANALYSIS_TOOLS):
        return ""
    normalized_tags = [str(item or "").strip().lower() for item in (policy_tags or []) if str(item or "").strip()]
    for tag in normalized_tags:
        profile = _BACKTESTS_PROFILE_TAGS.get(tag)
        if profile:
            return profile
    return ""


def _looks_like_research_shortlist_intent(
    *,
    title: str,
    objective: str,
    success_criteria: list[str],
    policy_tags: list[str],
) -> bool:
    normalized_tags = {str(item or "").strip().lower() for item in policy_tags if str(item or "").strip()}
    if "hypothesis_formation" in normalized_tags:
        return True
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (title, objective, *(success_criteria or []))
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in _SHORTLIST_MARKERS)


def _is_research_memory_write_tool(name: str, family: str, supports_terminal_write: bool) -> bool:
    normalized_name = str(name or "").strip()
    normalized_family = str(family or "").strip()
    return supports_terminal_write and (
        normalized_name in {"research_memory", "research_record"}
        or (normalized_family == "research_memory" and normalized_name not in {"research_project", "research_map"})
    )


def _supports_research_write_result(live_tools: list[object]) -> bool:
    live_tool_names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in live_tools
        if str(getattr(tool, "name", "") or "").strip()
    }
    if not live_tool_names or any(not _is_research_family_tool(name) for name in live_tool_names):
        return False
    terminal_write_names = {
        str(getattr(tool, "name", "") or "").strip()
        for tool in live_tools
        if bool(getattr(tool, "supports_terminal_write", False))
    }
    if not terminal_write_names:
        return False
    return all(_is_research_family_tool(name) for name in terminal_write_names)


def _is_research_family_tool(name: str) -> bool:
    normalized = str(name or "").strip()
    return normalized.startswith("research_")


__all__ = [
    "derive_runtime_slice_metadata",
    "resolve_runtime_slice_metadata",
    "resolve_runtime_slice_metadata_with_prerequisites",
]
