"""Build typed slice acceptance contracts from compiled slice metadata."""

from __future__ import annotations

from typing import Any

from app.execution_models import PlanSlice
from app.services.direct_execution.acceptance.contracts import AcceptanceContract, normalize_contract
from app.services.direct_execution.acceptance.states import normalize_dependency_unblock_mode

_BACKTEST_CONTRADICTIONS = [
    "no saved/active runs",
    "cannot evaluate",
    "lack standalone backtest runs",
    "no standalone backtest",
]


def build_acceptance_contract(slice_obj: PlanSlice) -> AcceptanceContract:
    existing = getattr(slice_obj, "acceptance_contract", None)
    if isinstance(existing, dict) and existing.get("kind"):
        return normalize_contract(existing)

    kind = derive_acceptance_kind(slice_obj)
    allow_advisory = normalize_dependency_unblock_mode(getattr(slice_obj, "dependency_unblock_mode", "")) == "advisory_only"
    required_predicates = _required_predicates(kind)
    return AcceptanceContract(
        kind=kind,
        mode="strict",
        required_subjects=_required_subjects(kind),
        required_predicates=required_predicates,
        mcp_proof_calls=_proof_call_templates(kind),
        forbid_contradictions=_forbidden_contradictions(kind),
        allow_advisory_unblock=allow_advisory,
        acceptance_requirements=_acceptance_requirements(slice_obj, kind=kind),
    )


def derive_acceptance_kind(slice_obj: PlanSlice) -> str:
    runtime_profile = str(getattr(slice_obj, "runtime_profile", "") or "").strip()
    tools = {str(item).strip() for item in getattr(slice_obj, "allowed_tools", []) if str(item).strip()}
    tags = {str(item).strip().lower() for item in getattr(slice_obj, "policy_tags", []) if str(item).strip()}
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            getattr(slice_obj, "title", ""),
            getattr(slice_obj, "objective", ""),
            *(getattr(slice_obj, "success_criteria", []) or []),
            *(getattr(slice_obj, "evidence_requirements", []) or []),
            *tags,
        )
        if str(item or "").strip()
    )
    if runtime_profile == "write_result":
        return "write_result"
    if runtime_profile == "research_setup" or {"research_project", "research_map", "research_memory"}.issubset(tools):
        return "research_setup"
    if runtime_profile == "research_shortlist" or "hypothesis_formation" in tags or "shortlist" in haystack:
        return "research_shortlist_write"
    if "model_training" in runtime_profile or "models_train" in tools or "model training" in haystack:
        return "model_training"
    if "backtests_integration" in runtime_profile or "integration" in tags:
        return "integration_backtest"
    if "backtests_stability" in runtime_profile or "stability" in tags:
        return "condition_stability_analysis"
    if "backtests_runs" in tools and "backtests_analysis" not in tools:
        return "standalone_backtests"
    if "feature_profitability_filter" in tags or "profitability" in haystack:
        return "feature_profitability_filter"
    if "features_custom" in tools and any(word in haystack for word in ("publish", "construction", "build custom", "construct")):
        return "feature_contract_construction"
    if tools & {"features_custom", "features_dataset", "features_catalog", "events", "datasets"}:
        return "feature_contract_exploration"
    if normalize_dependency_unblock_mode(getattr(slice_obj, "dependency_unblock_mode", "")) == "advisory_only":
        return "generic_read_advisory"
    return "generic_read_strict"


def _required_subjects(kind: str) -> str:
    return {
        "standalone_backtests": "each_surviving_candidate",
        "integration_backtest": "each_referenced_run",
        "condition_stability_analysis": "each_referenced_run",
        "model_training": "dataset_and_model_version",
        "write_result": "persisted_research_node",
        "research_shortlist_write": "persisted_research_node",
        "research_setup": "persisted_research_setup_nodes",
        "feature_contract_construction": "published_feature_and_dataset_if_required",
        "feature_profitability_filter": "explicit_candidate_set",
    }.get(kind, "none")


def _required_predicates(kind: str) -> list[str]:
    mapping = {
        "standalone_backtests": [
            "candidate_set_non_empty",
            "each_candidate_has_run_id",
            "each_run_exists",
            "each_run_completed",
            "each_run_has_metrics",
            "each_run_matches_snapshot_symbol_timeframes",
        ],
        "integration_backtest": ["run_set_non_empty", "each_run_proof_pass"],
        "condition_stability_analysis": ["run_set_non_empty", "each_run_proof_pass"],
        "write_result": ["mutating_tool_call_present", "research_node_proof_pass"],
        "research_shortlist_write": ["mutating_tool_call_present", "research_node_proof_pass"],
        "research_setup": ["research_project_present", "research_setup_facts_present"],
        "feature_contract_construction": ["feature_subject_present", "feature_proof_pass"],
        "feature_profitability_filter": ["candidate_decision_set_present", "domain_tool_evidence_present"],
        "model_training": ["model_dataset_proof_pass", "model_registry_proof_pass"],
        "generic_read_strict": ["domain_tool_evidence_present"],
    }
    return list(mapping.get(kind, []))


def _proof_call_templates(kind: str) -> list[dict[str, Any]]:
    if kind in {"standalone_backtests", "integration_backtest", "condition_stability_analysis"}:
        return [{"tool": "backtests_runs", "arguments": {"action": "inspect", "view": "acceptance_proof"}}]
    if kind in {"write_result", "research_shortlist_write", "research_setup"}:
        return [{"tool": "research_memory", "arguments": {"action": "prove"}}]
    if kind == "feature_contract_construction":
        return [
            {"tool": "features_custom", "arguments": {"action": "inspect", "view": "acceptance_proof"}},
            {"tool": "features_dataset", "arguments": {"action": "inspect", "view": "acceptance_proof"}},
        ]
    if kind == "model_training":
        return [
            {"tool": "models_dataset", "arguments": {"action": "inspect", "view": "acceptance_proof"}},
            {"tool": "models_registry", "arguments": {"action": "inspect", "view": "acceptance_proof"}},
        ]
    return []


def _forbidden_contradictions(kind: str) -> list[str]:
    if kind in {"standalone_backtests", "integration_backtest", "condition_stability_analysis"}:
        return list(_BACKTEST_CONTRADICTIONS)
    return []


def _acceptance_requirements(slice_obj: PlanSlice, *, kind: str) -> dict[str, Any]:
    requirements: dict[str, Any] = {
        "slice_id": getattr(slice_obj, "slice_id", ""),
        "contract_kind": kind,
    }
    if kind in {"standalone_backtests", "integration_backtest", "condition_stability_analysis"}:
        requirements.update(
            {
                "requires_completed": True,
                "requires_persisted_artifacts": True,
                "requires_metrics": True,
            }
        )
    if kind == "feature_contract_construction":
        requirements["require_dataset_activation"] = _mentions_dataset_activation(slice_obj)
    return requirements


def _mentions_dataset_activation(slice_obj: PlanSlice) -> bool:
    haystack = " ".join(
        str(item or "").strip().lower()
        for item in (
            getattr(slice_obj, "title", ""),
            getattr(slice_obj, "objective", ""),
            *(getattr(slice_obj, "success_criteria", []) or []),
        )
        if str(item or "").strip()
    )
    return any(marker in haystack for marker in ("dataset activation", "active dataset", "activate", "build dataset"))


__all__ = ["build_acceptance_contract", "derive_acceptance_kind"]
