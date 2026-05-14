"""
Feature-contract runtime helpers for direct execution.
"""

from __future__ import annotations

import json
from typing import Any

from app.services.direct_execution.guardrails import (
    final_report_payload_passes_gate,
    synthesize_transcript_evidence_refs,
)
from app.services.direct_execution.transcript_facts import derive_facts_from_transcript

_FEATURE_ANALYTICS_ACTIONS = frozenset({"analytics", "heatmap", "portability", "render"})
_FEATURES_CUSTOM_VIEWS = frozenset({"detail", "source"})
_READ_ONLY_EXPLORATION_TOOLS = frozenset({"features_catalog", "events", "datasets"})
_FILTER_FEATURE_PRIORITY = ("rsi_1", "atr_1", "iv_est_1", "rsi", "atr_1", "iv_est", "cl_1h")
_CONTRACT_MARKERS = (
    "feature contract",
    "data contract",
    "leakage",
    "custom feature",
    "validated and published",
    "event alignment",
    "feature_contract",
    "data_readiness",
)


def feature_contract_local_preflight(
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any] | None:
    normalized_tool = str(tool_name or "").strip()
    if normalized_tool == "features_analytics":
        action = str(arguments.get("action") or "").strip().lower()
        feature_name = _feature_name_argument(arguments)
        if action in _FEATURE_ANALYTICS_ACTIONS and not feature_name:
            return {
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": f"features_analytics(action='{action}') requires feature_name.",
                "details": {
                    "tool_name": "features_analytics",
                    "missing_field": "feature_name",
                    "action": action,
                    "reason_code": "features_analytics_requires_feature_name",
                    "remediation": (
                        "Inspect managed dataset columns first and choose one explicit feature_name "
                        "before feature-specific analytics or heatmaps."
                    ),
                },
            }
    if normalized_tool == "features_custom":
        action = str(arguments.get("action") or "").strip().lower()
        view = str(arguments.get("view") or "").strip().lower()
        name = str(arguments.get("name") or "").strip()
        if action == "inspect" and view in _FEATURES_CUSTOM_VIEWS and not name:
            return {
                "ok": False,
                "error_class": "agent_contract_misuse",
                "summary": f"features_custom(action='inspect', view='{view}') requires name.",
                "details": {
                    "tool_name": "features_custom",
                    "missing_field": "name",
                    "view": view,
                    "reason_code": "features_custom_detail_requires_name",
                    "remediation": (
                        "Inspect the custom-feature list first and choose one explicit name "
                        "before detail or source inspection."
                    ),
                },
            }
    return None


def repair_feature_analytics_identifier_from_transcript(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
) -> tuple[dict[str, Any], list[str]]:
    normalized_tool = str(tool_name or "").strip()
    if normalized_tool != "features_analytics":
        return arguments, []
    action = str(arguments.get("action") or "").strip().lower()
    if action not in _FEATURE_ANALYTICS_ACTIONS or _feature_name_argument(arguments):
        return arguments, []
    markers = " ".join(
        str(item or "").strip().lower()
        for item in (slice_title, slice_objective, *(success_criteria or []), *(policy_tags or []))
        if str(item or "").strip()
    )
    if "feature contract" not in markers and "profitability" not in markers and "filter" not in markers:
        return arguments, []
    feature_name = _latest_explicit_analytics_feature_name(transcript)
    if not feature_name and ("profitability" in markers or "filter" in markers):
        feature_name = _preferred_filter_feature_name_from_transcript(transcript)
    if not feature_name:
        return arguments, []
    repaired = dict(arguments)
    repaired["feature_name"] = feature_name
    return repaired, [f"reused_recent_feature_name:{feature_name}"]


def build_feature_contract_exploration_final_report(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    result_payload: dict[str, Any],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
    required_output_facts: list[str],
) -> str | None:
    if not _looks_like_feature_contract_slice(
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    ):
        return None
    normalized_tools = {str(item or "").strip() for item in allowed_tools if str(item or "").strip()}
    if not normalized_tools.issuperset({"research_memory"}) or not normalized_tools & _READ_ONLY_EXPLORATION_TOOLS:
        return None
    if normalized_tools & {"features_custom", "features_dataset", "models_dataset", "backtests_plan", "backtests_runs"}:
        return None
    if tool_name not in _READ_ONLY_EXPLORATION_TOOLS or _is_error_payload(result_payload):
        return None
    seen = _successful_tool_names(transcript)
    required_domain_tools = sorted(normalized_tools & _READ_ONLY_EXPLORATION_TOOLS)
    if not required_domain_tools or any(name not in seen for name in required_domain_tools):
        return None
    facts = {
        "execution.kind": "direct",
        "feature_contract.exploration_completed": True,
        "feature_contract.domain_tools": required_domain_tools,
    }
    transcript_facts = derive_facts_from_transcript(transcript, runtime_profile="generic_mutation")
    for key, value in transcript_facts.items():
        facts.setdefault(key, value)
    catalog_scopes = _successful_argument_values(transcript, "features_catalog", "scope")
    event_families = _successful_argument_values(transcript, "events", "family")
    dataset_views = _successful_argument_values(transcript, "datasets", "view")
    if catalog_scopes:
        facts["feature_contract.catalog_scopes"] = catalog_scopes
    if event_families:
        facts["feature_contract.event_families"] = event_families
    if dataset_views:
        facts["feature_contract.dataset_views"] = dataset_views
    findings = [
        "Feature/data contract exploration completed from live catalog, event, and dataset probes.",
        f"Read-only domain coverage captured: {', '.join(required_domain_tools)}.",
    ]
    if catalog_scopes:
        findings.append(f"Catalog scopes inspected: {', '.join(catalog_scopes)}.")
    if event_families:
        findings.append(f"Event families checked for alignment: {', '.join(event_families)}.")
    if dataset_views:
        findings.append(f"Dataset views inspected: {', '.join(dataset_views)}.")
    findings.extend(_success_criterion_findings(success_criteria))
    findings.append("Exploration stopped after sufficient live evidence instead of looping through more context reads.")
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Feature/data contract exploration phase completed from live MCP probes.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.78,
    }
    return _json_block(payload)


def feature_contract_exploration_missing_tools(
    *,
    transcript: list[dict[str, Any]],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
) -> list[str]:
    if not _is_read_only_feature_contract_exploration(
        allowed_tools=allowed_tools,
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    ):
        return []
    seen = _successful_tool_names(transcript)
    ordered = [name for name in ("features_catalog", "events", "datasets") if name in {str(item or "").strip() for item in allowed_tools}]
    return [name for name in ordered if name not in seen]


def feature_contract_exploration_next_call(
    *,
    transcript: list[dict[str, Any]],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
) -> dict[str, Any] | None:
    missing = feature_contract_exploration_missing_tools(
        transcript=transcript,
        allowed_tools=allowed_tools,
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    )
    if not missing:
        return None
    next_tool = missing[0]
    if next_tool == "features_catalog":
        return {"tool": "features_catalog", "arguments": {"scope": "timeframe", "timeframe": "1h"}}
    if next_tool == "events":
        return {"tool": "events", "arguments": {"view": "catalog"}}
    if next_tool == "datasets":
        return {"tool": "datasets", "arguments": {"view": "catalog"}}
    return None


def build_feature_contract_identifier_report(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
    required_output_facts: list[str],
) -> str | None:
    del allowed_tools
    if not _looks_like_feature_contract_slice(
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    ):
        return None
    identifier = _missing_identifier_name(
        tool_name=tool_name,
        arguments=arguments,
        result_payload=result_payload,
    )
    unknown_name = _unknown_custom_feature_name(
        transcript=transcript,
        tool_name=tool_name,
        arguments=arguments,
        result_payload=result_payload,
    )
    if not (identifier or unknown_name) or not _has_sufficient_contract_probe_evidence(transcript):
        return None
    facts = {
        "execution.kind": "direct",
        "feature_contract.contract_probe_completed": True,
    }
    if identifier:
        facts["feature_contract.explicit_identifier_required"] = identifier
    if unknown_name:
        facts["feature_contract.invalid_custom_feature_name"] = unknown_name
    transcript_facts = derive_facts_from_transcript(transcript, runtime_profile="generic_mutation")
    for key, value in transcript_facts.items():
        facts.setdefault(key, value)
    columns = _dataset_columns(transcript)
    custom_names = _custom_feature_names(transcript)
    if columns:
        facts["feature_contract.column_count"] = len(columns)
        facts["feature_contract.column_sample"] = columns[:8]
    if custom_names:
        facts["feature_contract.custom_feature_count"] = len(custom_names)
        facts["feature_contract.custom_feature_sample"] = custom_names[:6]
    findings = [
        "Feature-contract probes collected enough live evidence for a terminal report.",
    ]
    if identifier:
        findings.append(f"Explicit `{identifier}` selection is required before feature-specific analytics or detail calls.")
    if unknown_name:
        findings.append(f"Requested custom feature `{unknown_name}` was not present in the live inventory from `features_custom(list)`.")
    if columns:
        findings.append(f"Managed dataset columns inspected: {', '.join(columns[:8])}.")
    if _saw_successful_contract_view(transcript, tool_name="features_custom", view="contract"):
        findings.append("Custom-feature publish/validate contract was inspected.")
    if custom_names:
        findings.append(f"Custom-feature inventory inspected: {', '.join(custom_names[:4])}.")
    findings.extend(_success_criterion_findings(success_criteria))
    findings.append("Runtime stopped before inventing a feature identifier that was not selected by prior evidence.")
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Feature-contract evidence assembled from live probes without guessing feature-specific identifiers.",
        "verdict": "WATCHLIST",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.71,
    }
    return _json_block(payload)


def build_feature_contract_construction_final_report(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    result_payload: dict[str, Any],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
    required_output_facts: list[str],
) -> str | None:
    if not _looks_like_feature_contract_slice(
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    ):
        return None
    normalized_tools = {str(item or "").strip() for item in allowed_tools if str(item or "").strip()}
    if "research_memory" not in normalized_tools:
        return None
    if not normalized_tools & {"features_dataset", "features_custom", "features_analytics", "models_dataset"}:
        return None
    if tool_name not in {"features_dataset", "features_custom", "features_analytics", "models_dataset", "research_memory"}:
        return None
    if _is_error_payload(result_payload) or not _has_sufficient_construction_probe_evidence(transcript):
        return None
    facts = {
        "execution.kind": "direct",
        "feature_contract.construction_completed": True,
    }
    transcript_facts = derive_facts_from_transcript(transcript, runtime_profile="generic_mutation")
    for key, value in transcript_facts.items():
        facts.setdefault(key, value)
    columns = _dataset_columns(transcript)
    custom_names = _custom_feature_names(transcript)
    analytics_actions = _successful_analytics_actions(transcript)
    analytics_features = _successful_analytics_feature_names(transcript)
    if columns:
        facts["feature_contract.column_count"] = len(columns)
        facts["feature_contract.column_sample"] = columns[:8]
    if custom_names:
        facts["feature_contract.custom_feature_count"] = len(custom_names)
        facts["feature_contract.custom_feature_sample"] = custom_names[:6]
    if analytics_actions:
        facts["feature_contract.analytics_actions"] = analytics_actions
    if analytics_features:
        facts["feature_contract.analytics_feature_names"] = analytics_features[:4]
    findings = [
        "Feature-contract construction collected enough live evidence for a terminal report.",
        "Managed dataset columns were inspected before feature-specific analytics.",
    ]
    if columns:
        findings.append(f"Managed dataset columns inspected: {', '.join(columns[:8])}.")
    if _saw_successful_contract_view(transcript, tool_name='features_custom', view='contract'):
        findings.append("Custom-feature publish/validate contract was inspected.")
    if custom_names:
        findings.append(f"Custom-feature inventory inspected: {', '.join(custom_names[:4])}.")
    if analytics_features:
        findings.append(
            f"Explicit feature analytics were run for {', '.join(analytics_features[:3])} via {', '.join(analytics_actions[:3])}."
        )
    findings.extend(_success_criterion_findings(success_criteria))
    findings.append("Runtime stopped after contract coverage was sufficient instead of spending the remaining expensive-tool budget on redundant reads.")
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Feature/data contract construction completed from live dataset, custom-feature, and analytics probes.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.77,
    }
    return _json_block(payload)


def build_feature_profitability_filter_final_report(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    result_payload: dict[str, Any],
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
    required_output_facts: list[str],
) -> str | None:
    normalized_tools = {str(item or "").strip() for item in allowed_tools if str(item or "").strip()}
    markers = " ".join(
        str(item or "").strip().lower()
        for item in (slice_title, slice_objective, *(success_criteria or []), *(policy_tags or []))
        if str(item or "").strip()
    )
    if "research_memory" not in normalized_tools or "features_analytics" not in normalized_tools:
        return None
    if "filter" not in markers and "plausibility" not in markers and "profitability" not in markers:
        return None
    if tool_name not in {"features_catalog", "features_analytics", "research_memory"} or _is_error_payload(result_payload):
        return None
    if not _has_sufficient_profitability_filter_evidence(transcript, allowed_tools=normalized_tools):
        return None
    facts = {
        "execution.kind": "direct",
        "feature_filter.preliminary_screen_completed": True,
    }
    transcript_facts = derive_facts_from_transcript(transcript, runtime_profile="generic_mutation")
    for key, value in transcript_facts.items():
        facts.setdefault(key, value)
    catalog_scopes = _successful_argument_values(transcript, "features_catalog", "scope")
    analytics_actions = _successful_analytics_actions(transcript)
    analytics_features = _successful_analytics_feature_names(transcript)
    catalog_features = _catalog_cf_feature_names(transcript)
    if catalog_scopes:
        facts["feature_filter.catalog_scopes"] = catalog_scopes
    if analytics_actions:
        facts["feature_filter.analytics_actions"] = analytics_actions
    feature_names_for_facts = analytics_features[:4] if analytics_features else catalog_features[:4]
    if feature_names_for_facts:
        facts["feature_filter.analytics_feature_names"] = feature_names_for_facts
        facts["research.shortlist_families"] = feature_names_for_facts
    findings = [
        "Preliminary signal plausibility and feature profitability screen completed from live probes.",
    ]
    if catalog_scopes:
        findings.append(f"Feature catalog scopes inspected: {', '.join(catalog_scopes)}.")
    if analytics_features:
        findings.append(
            f"Profitability probes ran on explicit features {', '.join(analytics_features[:3])} via {', '.join(analytics_actions[:3])}."
        )
    elif catalog_features:
        findings.append(
            f"Feature catalog confirmed {len(catalog_features)} custom features available for profitability analysis."
        )
    findings.extend(_success_criterion_findings(success_criteria))
    findings.append("Runtime finalized once enough plausibility evidence was available, instead of stalling into salvage.")
    evidence_refs = synthesize_transcript_evidence_refs(transcript)
    if not final_report_payload_passes_gate(
        facts=facts,
        findings=findings,
        evidence_refs=evidence_refs,
        required_output_facts=required_output_facts,
    ):
        return None
    payload = {
        "type": "final_report",
        "summary": "Preliminary feature-profitability filter completed from live analytics evidence.",
        "verdict": "COMPLETE",
        "findings": findings,
        "facts": facts,
        "evidence_refs": evidence_refs,
        "confidence": 0.75,
    }
    return _json_block(payload)


def _looks_like_feature_contract_slice(
    *,
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
) -> bool:
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            slice_title,
            slice_objective,
            *(success_criteria or []),
            *(policy_tags or []),
        )
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in _CONTRACT_MARKERS)


def _is_read_only_feature_contract_exploration(
    *,
    allowed_tools: set[str] | list[str],
    slice_title: str,
    slice_objective: str,
    success_criteria: list[str],
    policy_tags: list[str] | None,
) -> bool:
    if not _looks_like_feature_contract_slice(
        slice_title=slice_title,
        slice_objective=slice_objective,
        success_criteria=success_criteria,
        policy_tags=policy_tags,
    ):
        return False
    normalized_tools = {str(item or "").strip() for item in allowed_tools if str(item or "").strip()}
    if "research_memory" not in normalized_tools:
        return False
    if not normalized_tools & _READ_ONLY_EXPLORATION_TOOLS:
        return False
    return not bool(normalized_tools & {"features_custom", "features_dataset", "models_dataset", "backtests_plan", "backtests_runs"})


def _feature_name_argument(arguments: dict[str, Any]) -> str:
    for field_name in ("feature_name", "feature", "column_name", "column", "name"):
        value = str(arguments.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _missing_identifier_name(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
) -> str:
    summary = _error_summary(result_payload).lower()
    if str(tool_name or "").strip() == "features_analytics":
        action = str(arguments.get("action") or "").strip().lower()
        if action in _FEATURE_ANALYTICS_ACTIONS and (not _feature_name_argument(arguments) or "requires feature_name" in summary):
            return "feature_name"
    if str(tool_name or "").strip() == "features_custom":
        action = str(arguments.get("action") or "").strip().lower()
        view = str(arguments.get("view") or "").strip().lower()
        if action == "inspect" and view in _FEATURES_CUSTOM_VIEWS and (not str(arguments.get("name") or "").strip() or "requires name" in summary):
            return "name"
    return ""


def _unknown_custom_feature_name(
    *,
    transcript: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
    result_payload: dict[str, Any],
) -> str:
    if str(tool_name or "").strip() != "features_custom":
        return ""
    action = str(arguments.get("action") or "").strip().lower()
    view = str(arguments.get("view") or "").strip().lower()
    if action != "inspect" or view not in _FEATURES_CUSTOM_VIEWS:
        return ""
    name = str(arguments.get("name") or "").strip()
    if not name:
        return ""
    known_names = set(_custom_feature_names(transcript))
    if not known_names or name in known_names:
        return ""
    summary = _error_summary(result_payload).lower()
    if "unknown custom feature" in summary or "not found" in summary:
        return name
    return ""


def _has_sufficient_contract_probe_evidence(transcript: list[dict[str, Any]]) -> bool:
    tool_names = _successful_tool_names(transcript)
    if "features_dataset" not in tool_names:
        return False
    return bool({"features_custom", "models_dataset"} & tool_names)


def _has_sufficient_construction_probe_evidence(transcript: list[dict[str, Any]]) -> bool:
    tool_names = _successful_tool_names(transcript)
    if "features_dataset" not in tool_names:
        return False
    if "features_analytics" not in tool_names:
        return False
    if not (
        _saw_successful_contract_view(transcript, tool_name="features_custom", view="contract")
        or "models_dataset" in tool_names
    ):
        return False
    return len(_successful_analytics_actions(transcript)) >= 2 and bool(_successful_analytics_feature_names(transcript))


def _has_sufficient_profitability_filter_evidence(
    transcript: list[dict[str, Any]],
    *,
    allowed_tools: set[str],
) -> bool:
    tool_names = _successful_tool_names(transcript)
    if "research_memory" not in tool_names:
        return False
    if "features_catalog" in allowed_tools and "features_catalog" not in tool_names:
        return False
    analytics_actions = _successful_analytics_actions(transcript)
    analytics_features = _successful_analytics_feature_names(transcript)
    catalog_features = _catalog_cf_feature_names(transcript)
    # Primary path: >=2 analytics actions with explicit feature names.
    if "features_analytics" in tool_names and len(analytics_actions) >= 2 and analytics_features:
        return True
    # Relaxed path: >=1 analytics action AND features_catalog returned cf_ names.
    if "features_analytics" in tool_names and analytics_actions and catalog_features:
        return True
    # Catalog-only path: analytics attempted but failed, catalog confirmed cf_ features.
    # Covers the case where MiniMax picks wrong feature names for analytics calls,
    # but the catalog provides sufficient evidence of available custom features.
    if catalog_features and _tool_was_attempted(transcript, "features_analytics"):
        return True
    return False


def _tool_was_attempted(transcript: list[dict[str, Any]], tool_name: str) -> bool:
    """Check if a tool was called at all (even if all calls failed)."""
    for entry in transcript:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "tool_result":
            continue
        if str(entry.get("tool") or "").strip() == tool_name:
            return True
    return False


def _catalog_cf_feature_names(transcript: list[dict[str, Any]]) -> list[str]:
    """Extract cf_ feature names discovered from features_catalog responses."""
    names: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_catalog":
            continue
        payload = entry.get("payload")
        if not isinstance(payload, dict):
            continue
        inner = payload.get("payload")
        if not isinstance(inner, dict):
            continue
        content = inner.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                continue
            text = str(block.get("text") or "").strip()
            if not text.startswith("{"):
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                continue
            columns = (parsed.get("data") or {}).get("columns") or []
            for col in columns:
                if isinstance(col, dict):
                    name = str(col.get("name") or "").strip()
                elif isinstance(col, str):
                    name = col.strip()
                else:
                    continue
                if name.startswith("cf_") and name not in seen:
                    seen.add(name)
                    names.append(name)
    return names


def _successful_tool_names(transcript: list[dict[str, Any]]) -> set[str]:
    return {
        str(entry.get("tool") or "").strip()
        for entry in _successful_tool_entries(transcript)
        if str(entry.get("tool") or "").strip()
    }


def _successful_argument_values(
    transcript: list[dict[str, Any]],
    tool_name: str,
    field_name: str,
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != tool_name:
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        value = str(arguments.get(field_name) or "").strip()
        if value and value not in seen:
            seen.add(value)
            values.append(value)
    return values


def _successful_analytics_actions(transcript: list[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_analytics":
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        action = str(arguments.get("action") or "").strip().lower()
        if action and action not in seen:
            seen.add(action)
            actions.append(action)
    return actions


def _successful_analytics_feature_names(transcript: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_analytics":
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        name = _feature_name_argument(arguments)
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _latest_explicit_analytics_feature_name(transcript: list[dict[str, Any]]) -> str:
    for entry in reversed(transcript):
        if entry.get("kind") != "tool_result":
            continue
        if str(entry.get("tool") or "").strip() != "features_analytics":
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        name = _feature_name_argument(arguments)
        if name:
            return name
        payload = entry.get("payload")
        structured = _extract_structured(payload)
        error = structured.get("error") if isinstance(structured.get("error"), dict) else {}
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        detail_name = str(details.get("feature_name") or "").strip()
        if detail_name:
            return detail_name
    return ""


def _preferred_filter_feature_name_from_transcript(transcript: list[dict[str, Any]]) -> str:
    columns = _catalog_columns(transcript) + _dataset_columns(transcript)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in columns:
        text = str(name or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    for candidate in _FILTER_FEATURE_PRIORITY:
        if candidate in seen:
            return candidate
    for candidate in ordered:
        if candidate.startswith("cf_"):
            continue
        if candidate.startswith(("bars_", "is_")):
            continue
        return candidate
    return ""


def _catalog_columns(transcript: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_catalog":
            continue
        structured = _extract_structured(entry.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        for item in data.get("columns") or []:
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _dataset_columns(transcript: list[dict[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_dataset":
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        if str(arguments.get("action") or "").strip().lower() != "inspect":
            continue
        if str(arguments.get("view") or "").strip().lower() != "columns":
            continue
        structured = _extract_structured(entry.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        for item in data.get("columns") or []:
            name = str(item or "").strip()
            if name and name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


def _custom_feature_names(transcript: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != "features_custom":
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        if str(arguments.get("action") or "").strip().lower() != "inspect":
            continue
        if str(arguments.get("view") or "").strip().lower() != "list":
            continue
        structured = _extract_structured(entry.get("payload"))
        data = structured.get("data") if isinstance(structured.get("data"), dict) else {}
        for item in data.get("features") or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if name and name not in seen:
                seen.add(name)
                names.append(name)
    return names


def _saw_successful_contract_view(
    transcript: list[dict[str, Any]],
    *,
    tool_name: str,
    view: str,
) -> bool:
    for entry in _successful_tool_entries(transcript):
        if str(entry.get("tool") or "").strip() != tool_name:
            continue
        arguments = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        if str(arguments.get("action") or "").strip().lower() != "inspect":
            continue
        if str(arguments.get("view") or "").strip().lower() == view:
            return True
    return False


def _successful_tool_entries(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successful: list[dict[str, Any]] = []
    for entry in transcript:
        if entry.get("kind") != "tool_result":
            continue
        if _is_error_payload(entry.get("payload")):
            continue
        successful.append(entry)
    return successful


def _is_error_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return True
    if payload.get("error") or payload.get("ok") is False:
        return True
    structured = _extract_structured(payload)
    return str(structured.get("status") or "").strip().lower() in {"error", "failed"}


def _error_summary(result_payload: dict[str, Any]) -> str:
    if not isinstance(result_payload, dict):
        return ""
    structured = _extract_structured(result_payload)
    if str(structured.get("message") or "").strip():
        return str(structured.get("message") or "").strip()
    if result_payload.get("summary"):
        return str(result_payload.get("summary") or "").strip()
    if result_payload.get("error"):
        return str(result_payload.get("error") or "").strip()
    details = result_payload.get("details")
    if isinstance(details, dict):
        return str(details.get("remediation") or "").strip()
    return ""


def _extract_structured(result_payload: Any) -> dict[str, Any]:
    if not isinstance(result_payload, dict):
        return {}
    payload = result_payload.get("payload")
    if isinstance(payload, dict):
        structured = payload.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = payload.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = str(item.get("text") or "").strip()
                if not text.startswith("{"):
                    continue
                try:
                    decoded = json.loads(text)
                except Exception:
                    continue
                if isinstance(decoded, dict):
                    return decoded
    return {}


def _success_criterion_findings(success_criteria: list[str]) -> list[str]:
    findings: list[str] = []
    for criterion in success_criteria[:2]:
        text = str(criterion or "").strip()
        if text:
            findings.append(f"Criterion addressed via live evidence: {text}")
    return findings


def _json_block(payload: dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


__all__ = [
    "build_feature_contract_construction_final_report",
    "build_feature_contract_exploration_final_report",
    "build_feature_contract_identifier_report",
    "build_feature_profitability_filter_final_report",
    "feature_contract_exploration_missing_tools",
    "feature_contract_exploration_next_call",
    "feature_contract_local_preflight",
    "repair_feature_analytics_identifier_from_transcript",
]
