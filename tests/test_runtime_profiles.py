from __future__ import annotations

from app.services.direct_execution.runtime_profiles import (
    derive_runtime_slice_metadata,
    resolve_runtime_slice_metadata_with_prerequisites,
)
from app.services.mcp_catalog.models import McpCatalogSnapshot, McpToolSpec


def _snapshot_with_tools(*tools: McpToolSpec) -> McpCatalogSnapshot:
    return McpCatalogSnapshot(
        server_name="dev_space1",
        endpoint_url="http://127.0.0.1:8766/mcp",
        schema_hash="hash",
        fetched_at="2026-04-11T00:00:00Z",
        tools=list(tools),
    )


def test_research_setup_toolset_gets_strict_runtime_profile() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="research_project",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "branch_id"],
        ),
        McpToolSpec(
            name="research_map",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "branch_id"],
        ),
    )

    profile, required_facts, finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=["research_project", "research_map", "research_memory"],
        catalog_snapshot=snapshot,
    )

    assert profile == "research_setup"
    assert required_facts == [
        "research.project_id",
        "research.baseline_configured",
        "research.atlas_defined",
        "research.invariants_recorded",
        "research.naming_recorded",
    ]
    assert finalization_mode == "fact_based"

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile=profile,
        required_output_facts=required_facts,
        required_prerequisite_facts=None,
        finalization_mode=finalization_mode,
        allowed_tools=["research_project", "research_map", "research_memory"],
        catalog_snapshot=snapshot,
    )

    assert resolved == (
        "research_setup",
        [
            "research.project_id",
            "research.baseline_configured",
            "research.atlas_defined",
            "research.invariants_recorded",
            "research.naming_recorded",
        ],
        [],
        "fact_based",
    )


def test_research_shortlist_intent_gets_strict_runtime_profile() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="research_project",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "branch_id"],
        ),
        McpToolSpec(
            name="research_map",
            side_effects="read_only",
            supports_terminal_write=False,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
        McpToolSpec(
            name="research_record",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "operation_id"],
        ),
    )

    profile, required_facts, finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=["research_project", "research_map", "research_record"],
        catalog_snapshot=snapshot,
        title="Form first-wave shortlist",
        objective="Assemble new signal families with novelty justification versus history v1-v12",
        success_criteria=["Shortlist exists"],
        policy_tags=["hypothesis_formation"],
    )

    assert profile == "research_shortlist"
    assert required_facts == [
        "research.project_id",
        "research.shortlist_families",
        "research.novelty_justification_present",
    ]
    assert finalization_mode == "fact_based"

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile=profile,
        required_output_facts=required_facts,
        required_prerequisite_facts=None,
        finalization_mode=finalization_mode,
        allowed_tools=["research_project", "research_map", "research_record"],
        catalog_snapshot=snapshot,
        title="Form first-wave shortlist",
        objective="Assemble new signal families with novelty justification versus history v1-v12",
        success_criteria=["Shortlist exists"],
        policy_tags=["hypothesis_formation"],
    )

    assert resolved == (
        "research_shortlist",
        [
            "research.project_id",
            "research.shortlist_families",
            "research.novelty_justification_present",
        ],
        [],
        "fact_based",
    )


def test_mixed_feature_contract_toolset_stays_generic_mutation() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="features_custom",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["operation_id"],
            produced_handle_fields=["operation_id"],
        ),
        McpToolSpec(
            name="features_dataset",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["operation_id"],
            produced_handle_fields=["operation_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "memory_node_id"],
        ),
    )

    profile, required_facts, finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=["features_custom", "features_dataset", "research_memory"],
        catalog_snapshot=snapshot,
        title="Data contract and feature contract for each new family",
        objective="Validate custom feature publication and leakage checks before tests.",
        success_criteria=["Custom features validated and published."],
        policy_tags=["feature_contract"],
    )

    assert profile == "generic_mutation"
    assert required_facts == []
    assert finalization_mode == "none"

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile=profile,
        required_output_facts=required_facts,
        required_prerequisite_facts=None,
        finalization_mode=finalization_mode,
        allowed_tools=["features_custom", "features_dataset", "research_memory"],
        catalog_snapshot=snapshot,
        title="Data contract and feature contract for each new family",
        objective="Validate custom feature publication and leakage checks before tests.",
        success_criteria=["Custom features validated and published."],
        policy_tags=["feature_contract"],
    )

    assert resolved == (
        "generic_mutation",
        [],
        [],
        "none",
    )


def test_backtests_stability_slice_gets_strict_runtime_profile() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="backtests_conditions",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["job_id", "snapshot_id"],
            produced_handle_fields=["job_id"],
        ),
        McpToolSpec(
            name="backtests_analysis",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["run_id"],
            produced_handle_fields=["run_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
    )

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile="",
        required_output_facts=None,
        required_prerequisite_facts=None,
        finalization_mode="",
        allowed_tools=["backtests_conditions", "backtests_analysis", "research_memory"],
        catalog_snapshot=snapshot,
        title="Stability и condition analysis",
        objective="Проверить stability и condition behavior для shortlist кандидатов.",
        success_criteria=["Condition stability measured."],
        policy_tags=["stability", "analysis"],
    )

    assert resolved == (
        "backtests_stability_analysis",
        [],
        [],
        "none",
    )


def test_backtests_profile_ignores_success_criteria_substrings_without_policy_tag() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="backtests_analysis",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["run_id"],
            produced_handle_fields=["run_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
    )

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile="",
        required_output_facts=None,
        required_prerequisite_facts=None,
        finalization_mode="",
        allowed_tools=["backtests_analysis", "research_memory"],
        catalog_snapshot=snapshot,
        title="Check stability",
        objective="Measure regime concentration.",
        success_criteria=["stable enough for integration testing"],
        policy_tags=["stability", "gate"],
    )

    assert resolved[0] == "backtests_stability_analysis"


def test_backtests_profile_requires_policy_tag_for_integration() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="backtests_analysis",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["run_id"],
            produced_handle_fields=["run_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
    )

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile="",
        required_output_facts=None,
        required_prerequisite_facts=None,
        finalization_mode="",
        allowed_tools=["backtests_analysis", "research_memory"],
        catalog_snapshot=snapshot,
        title="Integration checks",
        objective="Compare matched OOS runs.",
        success_criteria=["integration testing complete"],
        policy_tags=["integration"],
    )

    assert resolved[0] == "backtests_integration_analysis"


def test_backtests_integration_slice_gets_strict_runtime_profile() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="backtests_runs",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["run_id", "snapshot_id"],
            produced_handle_fields=["run_id"],
        ),
        McpToolSpec(
            name="backtests_analysis",
            side_effects="mutating",
            supports_polling=True,
            accepted_handle_fields=["run_id"],
            produced_handle_fields=["run_id"],
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id"],
        ),
    )

    resolved = resolve_runtime_slice_metadata_with_prerequisites(
        runtime_profile="",
        required_output_facts=None,
        required_prerequisite_facts=None,
        finalization_mode="",
        allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
        catalog_snapshot=snapshot,
        title="Интеграция с active-signal-v1@1",
        objective="Проверить реальную ценность каждого surviving candidate поверх неизменной базы через интеграционные бэктесты.",
        success_criteria=["Есть полная интеграционная картина."],
        policy_tags=["integration", "analysis"],
    )

    assert resolved == (
        "backtests_integration_analysis",
        [],
        [],
        "none",
    )


def test_mixed_read_exploration_toolset_does_not_collapse_to_write_result() -> None:
    snapshot = _snapshot_with_tools(
        McpToolSpec(
            name="features_catalog",
            side_effects="read_only",
            supports_terminal_write=False,
            supports_discovery=True,
            fields=["scope", "timeframe"],
            description="Inspect the managed feature catalog.",
        ),
        McpToolSpec(
            name="events",
            side_effects="read_only",
            supports_terminal_write=False,
            supports_discovery=True,
            fields=["family", "symbol"],
            description="Inspect local normalized event stores.",
        ),
        McpToolSpec(
            name="datasets",
            side_effects="read_only",
            supports_terminal_write=False,
            supports_discovery=True,
            fields=["view", "dataset_id"],
            description="Inspect dataset catalog and details.",
        ),
        McpToolSpec(
            name="research_memory",
            side_effects="mutating",
            supports_terminal_write=True,
            accepted_handle_fields=["project_id"],
            produced_handle_fields=["project_id", "memory_node_id"],
        ),
    )

    profile, required_facts, finalization_mode = derive_runtime_slice_metadata(
        allowed_tools=["features_catalog", "events", "datasets", "research_memory"],
        catalog_snapshot=snapshot,
        title="Data contract and feature contract exploration",
        objective="Explore data coverage, alignment, and feature contracts before recording findings.",
        success_criteria=["Exploration phase complete"],
        policy_tags=["feature_contract", "data_readiness"],
    )

    assert profile == "generic_mutation"
    assert required_facts == []
    assert finalization_mode == "none"
