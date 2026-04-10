from __future__ import annotations

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.fact_hydration import hydrate_final_report_facts


def test_hydrate_final_report_facts_derives_research_aliases_and_shortlist() -> None:
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
            facts={"project_id": "proj_1"},
            findings=[
                "Cross-Market Correlation Signals - Novelty: Uses cross-market features.",
                "Order Flow Imbalance Signals - Novelty: Uses order flow.",
            ],
            evidence_refs=["node_1", "note_2"],
        ),
        required_output_facts=["research.project_id", "research.shortlist_families", "research.hypothesis_refs"],
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
        required_output_facts=["research.project_id", "research.shortlist_families", "research.hypothesis_refs"],
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
        required_output_facts=["research.project_id", "research.shortlist_families"],
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
        required_output_facts=["research.project_id", "research.shortlist_families"],
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
        required_output_facts=["research.project_id", "research.shortlist_families"],
        inherited_facts={"compiled_plan_v1_stage_1.project_id": "proj_prefixed"},
    )

    assert readiness.facts["research.project_id"] == "proj_prefixed"
    assert readiness.missing_required_facts == []
