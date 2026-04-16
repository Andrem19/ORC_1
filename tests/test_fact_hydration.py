from __future__ import annotations

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts


def test_hydrate_final_report_facts_derives_research_aliases_and_hypothesis_refs() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_2",
            title="Shortlist",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_record"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "project_id": "proj_1",
                "research.shortlist_families": [
                    "Cross-Market Correlation Signals",
                    "Order Flow Imbalance Signals",
                ],
            },
            findings=["Shortlist stored in structured facts."],
            evidence_refs=["node_1", "note_2"],
        ),
        required_output_facts=["research.shortlist_families", "research.hypothesis_refs"],
    )

    assert readiness.facts["research.project_id"] == "proj_1"
    assert readiness.facts["research.shortlist_families"] == [
        "Cross-Market Correlation Signals",
        "Order Flow Imbalance Signals",
    ]
    assert readiness.facts["research.hypothesis_refs"] == ["node_1"]
    assert readiness.missing_required_facts == []


def test_hydrate_final_report_facts_reports_missing_required_facts() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_2",
            title="Shortlist",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_record"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a2",
            action_type="final_report",
            summary="done",
            facts={"project_id": "proj_1"},
            findings=[],
            evidence_refs=[],
        ),
        required_output_facts=["research.shortlist_families", "research.hypothesis_refs"],
    )

    assert readiness.missing_required_facts == ["research.shortlist_families", "research.hypothesis_refs"]


def test_hydrate_final_report_facts_uses_valid_signal_types_as_shortlist() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_2",
            title="Shortlist",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_record"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a3",
            action_type="final_report",
            summary="done",
            facts={"project_id": "proj_1", "valid_signal_types": ["price_momentum", "order_flow"]},
            findings=[],
            evidence_refs=[],
        ),
        required_output_facts=["research.shortlist_families"],
    )

    assert readiness.facts["research.shortlist_families"] == ["price_momentum", "order_flow"]
    assert readiness.missing_required_facts == []


def test_hydrate_final_report_facts_uses_inherited_facts_for_required_fields() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_3",
            title="Contracts",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["features_catalog"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a4",
            action_type="final_report",
            summary="done",
            facts={"research.shortlist_families": ["momentum", "funding"]},
            findings=[],
            evidence_refs=[],
        ),
        required_output_facts=["research.shortlist_families"],
        inherited_facts={"research.project_id": "proj_inherited"},
    )

    assert readiness.facts["research.project_id"] == "proj_inherited"
    assert readiness.facts["research.shortlist_families"] == ["momentum", "funding"]
    assert readiness.missing_required_facts == []


def test_hydrate_final_report_facts_extracts_project_id_from_prefixed_inherited_key() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_3",
            title="Contracts",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["features_catalog"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a5",
            action_type="final_report",
            summary="done",
            facts={"research.shortlist_families": ["momentum"]},
            findings=[],
            evidence_refs=[],
        ),
        required_output_facts=["research.shortlist_families"],
        inherited_facts={"compiled_plan_v1_stage_1.project_id": "proj_prefixed"},
    )

    assert readiness.facts["research.project_id"] == "proj_prefixed"
    assert readiness.missing_required_facts == []


def test_hydrate_final_report_facts_prefers_non_transient_project_id() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project", "research_map", "research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a5b",
            action_type="final_report",
            summary="done",
            facts={
                "research.project_id": "c56e9c06968d4eb09317dc1d2f0914e9",
                "project_id": "v1-cycle-invariants-7750a80d",
            },
            findings=[],
            evidence_refs=["transcript:1:research_project"],
        ),
        required_output_facts=[],
    )

    assert readiness.facts["research.project_id"] == "v1-cycle-invariants-7750a80d"
    assert readiness.missing_required_facts == []


def test_research_shortlist_hydration_does_not_backfill_shortlist_from_findings() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_2",
            title="Shortlist",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_record"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            runtime_profile="research_shortlist",
        ),
        action=WorkerAction(
            action_id="a6",
            action_type="final_report",
            summary="done",
            facts={"project_id": "proj_1"},
            findings=[
                "Funding Dislocation - Novelty: Uses cross-exchange funding stress.",
                "Expiry Proximity - Novelty: Uses expiry timing.",
            ],
            evidence_refs=["node_1"],
        ),
        required_output_facts=[
            "research.shortlist_families",
            "research.novelty_justification_present",
        ],
    )

    assert readiness.facts["research.project_id"] == "proj_1"
    assert "research.shortlist_families" not in readiness.facts
    assert readiness.missing_required_facts == [
        "research.shortlist_families",
        "research.novelty_justification_present",
    ]


def test_generic_hydration_does_not_backfill_shortlist_from_findings() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a7",
            action_type="final_report",
            summary="done",
            facts={"project_id": "proj_1"},
            findings=[
                "Baseline explicitly configured.",
                "Naming convention recorded.",
            ],
            evidence_refs=["note_1"],
        ),
        required_output_facts=[],
    )

    assert readiness.facts["research.project_id"] == "proj_1"
    assert "research.shortlist_families" not in readiness.facts


def test_hydrate_final_report_facts_canonicalizes_backtests_handoff_outputs() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_6",
            title="Stability",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_conditions", "backtests_analysis", "research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            runtime_profile="backtests_stability_analysis",
        ),
        action=WorkerAction(
            action_id="a8",
            action_type="final_report",
            summary="done",
            facts={
                "feature_long_job": "cond-f07199b451c1",
                "feature_short_job": "cond-8972f7822bb2",
                "diagnostics_run": "20260411-193208-40dbd831",
            },
            findings=[],
            evidence_refs=["analysis-e83ae8371a57"],
        ),
        required_output_facts=[],
    )

    assert readiness.facts["backtests.candidate_handles"] == {
        "feature_long_job": "cond-f07199b451c1",
        "feature_short_job": "cond-8972f7822bb2",
    }
    assert "analysis-e83ae8371a57" in readiness.facts["backtests.analysis_refs"]
    assert readiness.missing_required_facts == []


def test_hydrate_final_report_facts_canonicalizes_integration_aliases_and_sanitizes_transport_refs() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_7",
            title="Integration",
            hypothesis="h",
            objective="Integrate surviving candidates over the base.",
            success_criteria=[],
            allowed_tools=["backtests_plan", "backtests_runs", "backtests_analysis", "research_memory"],
            evidence_requirements=[],
            policy_tags=["integration"],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            runtime_profile="backtests_integration_analysis",
        ),
        action=WorkerAction(
            action_id="a8b",
            action_type="final_report",
            summary="done",
            facts={
                "request_id": 1,
                "server_session_id": "0f116f24435a4d9ebe19eb11105e183f",
                "run_id": "3",
                "integration_handles": {
                    "feature_long": "20260413-143910-3db43a14",
                    "feature_short": "20260413-144002-e2f7c911",
                },
                "integration_refs": [
                    "analysis-18dfbc658a2a-rm-feature-long",
                    "3",
                    "0f116f24435a4d9ebe19eb11105e183f",
                ],
                "direct.created_ids": [
                    "1",
                    "0f116f24435a4d9ebe19eb11105e183f",
                    "node-valid",
                    "research-project-valid",
                    "branch-valid",
                ],
            },
            findings=[],
            evidence_refs=[
                "1",
                "0f116f24435a4d9ebe19eb11105e183f",
                "analysis-18dfbc658a2a-rm-feature-long",
                "node-valid",
            ],
        ),
        required_output_facts=[],
    )

    assert readiness.facts["backtests.integration_handles"] == {
        "feature_long": "20260413-143910-3db43a14",
        "feature_short": "20260413-144002-e2f7c911",
    }
    assert readiness.facts["backtests.integration_refs"] == ["analysis-18dfbc658a2a-rm-feature-long"]
    assert readiness.facts["direct.created_ids"] == ["node-valid", "research-project-valid", "branch-valid"]
    assert "request_id" not in readiness.facts
    assert "server_session_id" not in readiness.facts
    assert "run_id" not in readiness.facts
    assert readiness.evidence_refs == ["analysis-18dfbc658a2a-rm-feature-long", "node-valid"]
    assert readiness.missing_required_facts == []


# ---------- project_id rescue from created_ids / evidence_refs ----------


def test_hydrate_rescues_project_id_from_created_ids() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project", "research_map", "research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "direct.successful_tool_names": ["research_project", "research_memory"],
                "direct.created_ids": ["cycle-invariants-v1-95eb1ec8", "node-abc123"],
                "research.baseline_configured": True,
            },
        ),
        required_output_facts=["research.project_id"],
    )

    assert readiness.facts["research.project_id"] == "cycle-invariants-v1-95eb1ec8"


def test_hydrate_skips_node_ids_in_project_rescue() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "direct.successful_tool_names": ["research_project"],
                "direct.created_ids": ["node-abc123", "dim-xyz789", "branch-deadbeef"],
            },
        ),
        required_output_facts=[],
    )

    assert "research.project_id" not in readiness.facts


def test_hydrate_skips_rescue_without_research_project() -> None:
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "direct.successful_tool_names": ["research_memory"],
                "direct.created_ids": ["cycle-invariants-v1-95eb1ec8"],
            },
        ),
        required_output_facts=[],
    )

    assert "research.project_id" not in readiness.facts


# ---------- transcript reference rejection ----------


def test_hydrate_rejects_transcript_ref_as_project_id() -> None:
    """MiniMax puts transcript:2:research_project as project_id — must be rejected."""
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "research.project_id": "transcript:2:research_project",
                "project_id": "transcript:2:research_project",
                "direct.successful_tool_names": ["research_project"],
                "direct.created_ids": ["active-signal-cycle-78c740d5", "node-abc"],
            },
        ),
        required_output_facts=[],
    )

    assert readiness.facts["research.project_id"] == "active-signal-cycle-78c740d5"
    assert readiness.facts["project_id"] == "active-signal-cycle-78c740d5"


def test_hydrate_keeps_valid_project_id_over_transcript_ref() -> None:
    """When both a real and a transcript ref project_id exist, real one wins."""
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "research.project_id": "transcript:2:research_project",
                "project_id": "real-project-123",
            },
        ),
        required_output_facts=[],
    )

    assert readiness.facts["research.project_id"] == "real-project-123"


def test_hydrate_no_project_id_when_only_transcript_refs() -> None:
    """When only transcript refs exist and no rescue source, project_id is absent."""
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_memory"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "project_id": "transcript:4:research_memory",
            },
        ),
        required_output_facts=[],
    )

    assert "research.project_id" not in readiness.facts


def test_hydrate_rescue_skips_transcript_refs_in_evidence() -> None:
    """Rescue should skip transcript:* entries in evidence_refs."""
    readiness = hydrate_final_report_facts(
        slice_obj=PlanSlice(
            slice_id="stage_1",
            title="Setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        action=WorkerAction(
            action_id="a1",
            action_type="final_report",
            summary="done",
            facts={
                "direct.successful_tool_names": ["research_project"],
                "direct.supported_evidence_refs": [
                    "transcript:2:research_project",
                    "transcript:4:research_project",
                    "active-signal-cycle-78c740d5",
                ],
            },
        ),
        required_output_facts=[],
    )

    assert readiness.facts["research.project_id"] == "active-signal-cycle-78c740d5"
