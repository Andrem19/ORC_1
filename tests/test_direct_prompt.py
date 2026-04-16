from __future__ import annotations

from app.services.direct_execution.backtests_protocol import (
    augment_allowed_tools_for_backtests,
    build_backtests_zero_tool_nudge,
)
from app.services.direct_execution.prompt import build_direct_slice_prompt

from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _base_prompt_kwargs(**overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "s1",
            "title": "t",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": [],
            "evidence_requirements": [],
        },
        baseline_bootstrap={
            "baseline_snapshot_id": "active-signal-v1",
            "baseline_version": 1,
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "execution_timeframe": "5m",
        },
        known_facts={},
        recent_turn_summaries=[],
        checkpoint_summary="",
        max_tool_calls=8,
        max_expensive_tool_calls=2,
        catalog_snapshot=make_catalog_snapshot(),
    )
    kwargs.update(overrides)
    return kwargs


def test_direct_prompt_includes_generated_required_field_hints() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_map", "research_search"],
        )
    )

    assert "research_map: Inspect or advance the research atlas." in prompt
    assert "research_map: required fields -> project_id" in prompt
    assert "research_search: required fields -> query" in prompt


def test_direct_prompt_includes_enum_hints() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["experiments_read", "system_bootstrap"],
        )
    )

    assert "experiments_read.view: allowed values -> json, list, text" in prompt
    assert "system_bootstrap.view: allowed values -> detail, raw, summary" in prompt


def test_direct_prompt_includes_policy_locked_field_description() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_catalog"],
        )
    )

    assert "features_catalog.timeframe: policy-locked timeframe token." in prompt


def test_direct_prompt_includes_bootstrap_hint_when_available() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_map"],
        )
    )

    assert "research_map.inspect is the default read path for atlas state." in prompt


def test_known_facts_are_compacted_without_recursive_prefix_chains() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_project"],
            slice_payload={
                "slice_id": "s2",
                "title": "t",
                "hypothesis": "h",
                "objective": "o",
                "success_criteria": [],
                "evidence_requirements": [],
                "depends_on": ["stage_6"],
            },
            known_facts={
                "stage_6.stage_5.stage_4.project_id": "proj_1",
                "stage_3.stage_2.stage_1.shortlist": ["momentum"],
            },
        )
    )
    assert "stage_6.project_id = proj_1" in prompt
    assert "stage_6.stage_5" not in prompt
    assert "stage_3.shortlist" in prompt


def test_direct_prompt_includes_research_setup_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_project", "research_map", "research_memory"],
        )
    )

    assert "Research setup protocol:" in prompt
    assert "research_project(action='set_baseline'" in prompt
    assert "create or explicitly open the cycle project" in prompt
    assert "research_map(action='define'" in prompt
    assert "research_memory(action='create'" in prompt
    assert "research.baseline_configured" in prompt


def test_direct_prompt_includes_research_shortlist_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_project", "research_map", "research_record"],
            slice_payload={
                "slice_id": "s3",
                "title": "Form shortlist",
                "hypothesis": "h",
                "objective": "Assemble a shortlist of novel signal families versus v1-v12.",
                "success_criteria": ["Shortlist exists"],
                "evidence_requirements": [],
                "policy_tags": ["hypothesis_formation"],
                "runtime_profile": "research_shortlist",
            },
            required_output_facts=[
                "research.project_id",
                "research.memory_node_id",
                "research.shortlist_families",
                "research.novelty_justification_present",
            ],
        )
    )

    assert "Research shortlist protocol:" in prompt
    assert "Read/map calls do not complete this slice" in prompt
    assert "record.metadata.shortlist_families" in prompt
    assert "record.content.candidates=[{family, why_new, relative_to}]" in prompt
    assert "execute the milestone write instead of returning a checkpoint" in prompt
    assert "terminal verdict MUST be COMPLETE" in prompt


def test_direct_prompt_includes_feature_contract_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_custom", "features_dataset", "features_analytics", "models_dataset", "research_memory"],
            slice_payload={
                "slice_id": "s4",
                "title": "Data contract and feature contract",
                "hypothesis": "h",
                "objective": "Define a precise data contract, validate leakage control, and publish custom features.",
                "success_criteria": [
                    "Custom features validated and published.",
                    "Leakage verified for every contract.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["data_readiness", "feature_contract"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "Feature contract protocol:" in prompt
    assert "`research_memory.search` is context recovery only" in prompt
    assert "features_dataset(action='inspect', view='columns'" in prompt
    assert "features_custom(action='inspect', view='contract')" in prompt
    assert "Do not guess `feature_name`" in prompt
    assert "requires one explicit `feature_name`" in prompt
    assert "requires one explicit `name`" in prompt


def test_direct_prompt_includes_feature_contract_exploration_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_catalog", "events", "datasets", "research_memory"],
            slice_payload={
                "slice_id": "s5",
                "title": "Data contract and feature contract exploration",
                "hypothesis": "h",
                "objective": "Explore managed features, datasets, and event coverage for leakage-safe feature contracts.",
                "success_criteria": [
                    "Event alignment verified for event hypotheses.",
                    "Exploration phase complete.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["data_readiness", "feature_contract"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "Feature contract protocol:" in prompt
    assert "features_catalog(scope='timeframe'" in prompt
    assert "events(view='catalog')" in prompt
    assert "datasets(view='catalog')" in prompt
    assert "Mixed outcomes are allowed" in prompt
    assert "Do not use WATCHLIST just because some families are blocked" in prompt


def test_direct_prompt_strict_acceptance_requires_complete_example() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_catalog", "events", "datasets", "research_memory"],
            slice_payload={
                "slice_id": "s5b",
                "title": "Strict exploration",
                "hypothesis": "h",
                "objective": "Classify every shortlisted family with live evidence.",
                "success_criteria": ["Every family classified."],
                "evidence_requirements": [],
                "dependency_unblock_mode": "accepted_only",
                "watchlist_allows_unblock": False,
                "policy_tags": ["data_readiness", "feature_contract"],
            },
        )
    )

    assert "This slice is acceptance-strict" in prompt
    assert "downstream work will not unblock on WATCHLIST" in prompt
    assert '"verdict":"COMPLETE"' in prompt


def test_direct_prompt_includes_generic_mixed_domain_protocol_for_read_only_probe() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_analytics", "features_catalog", "research_memory"],
            slice_payload={
                "slice_id": "s6",
                "title": "Quick plausibility filter",
                "hypothesis": "h",
                "objective": "Use feature profitability and plausibility checks to filter weak hypotheses before backtests.",
                "success_criteria": [
                    "Only hypotheses with preliminary meaning remain.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["cheap_first", "filter"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "Mixed-domain protocol:" in prompt
    assert "After 2 successful non-research probes" in prompt
    assert "Do not bounce back to `research_memory.search`" in prompt


def test_direct_prompt_includes_backtests_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_plan", "backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s7",
                "title": "Stability analysis of candidates",
                "hypothesis": "h",
                "objective": "Run stability backtests for shortlisted signals.",
                "success_criteria": [
                    "Stable signals pass condition analysis.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["backtesting"],
                "runtime_profile": "backtests_stability_analysis",
            },
        )
    )

    assert "Backtests protocol:" in prompt
    assert "First live action: call backtests_plan" in prompt
    assert "Start `backtests_runs(action='start', ...)` only after a successful `backtests_plan(...)`" in prompt
    assert "Do not loop on research_memory" in prompt


def test_direct_prompt_standalone_backtests_gets_inspect_first() -> None:
    """Standalone backtest slices should get snapshot-inspect-first guidance."""
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_plan", "backtests_runs", "backtests_strategy", "research_memory"],
            slice_payload={
                "slice_id": "s7",
                "title": "Standalone backtests новых сигналов",
                "hypothesis": "h",
                "objective": "Запустить честные standalone бэктесты для shortlist кандидатов.",
                "success_criteria": [
                    "Есть shortlist standalone-кандидатов.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["backtesting"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "Backtests protocol:" in prompt
    # Standalone slices should NOT have hardcoded baseline first action
    assert "First live action: call backtests_plan" not in prompt
    # Should have snapshot inspection guidance instead
    assert "backtests_strategy(action='inspect'" in prompt
    assert "research_memory(action='create'" in prompt


def test_backtests_prompt_does_not_reference_unavailable_backtests_plan() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s7b",
                "title": "Integration",
                "hypothesis": "h",
                "objective": "Integrate surviving candidates over the baseline.",
                "success_criteria": ["integration"],
                "evidence_requirements": [],
                "policy_tags": ["integration"],
                "runtime_profile": "backtests_integration_analysis",
            },
        )
    )

    assert "Backtests protocol:" in prompt
    assert "First live action: call backtests_plan" not in prompt
    assert "backtests_plan(" not in prompt
    assert "Use only the approved backtests tools listed above" in prompt


def test_backtests_allowed_tools_auto_add_read_only_plan_when_live() -> None:
    effective = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_runs", "research_memory"},
        catalog_snapshot=make_catalog_snapshot(),
        runtime_profile="backtests_integration_analysis",
        title="Integration",
        objective="Integrate surviving candidates over the baseline.",
        success_criteria=["integration"],
        policy_tags=["integration"],
    )

    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=sorted(effective),
            slice_payload={
                "slice_id": "s7c",
                "title": "Integration",
                "hypothesis": "h",
                "objective": "Integrate surviving candidates over the baseline.",
                "success_criteria": ["integration"],
                "evidence_requirements": [],
                "policy_tags": ["integration"],
                "runtime_profile": "backtests_integration_analysis",
            },
        )
    )

    assert "backtests_plan" in effective
    assert "First live action: call backtests_plan" in prompt


def test_backtests_zero_tool_nudge_is_allowed_tool_aware() -> None:
    with_plan = build_backtests_zero_tool_nudge(
        allowed_tools={"backtests_plan", "backtests_runs"},
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1},
    )
    without_plan = build_backtests_zero_tool_nudge(
        allowed_tools={"backtests_runs"},
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1},
    )

    assert "backtests_plan(" in with_plan
    assert "backtests_plan(" not in without_plan
    assert "do not start a duplicate baseline run" in without_plan


def test_direct_prompt_includes_backtests_analysis_protocol() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_conditions", "backtests_analysis", "research_memory"],
            slice_payload={
                "slice_id": "s8",
                "title": "Stability и condition analysis",
                "hypothesis": "h",
                "objective": "Проверить stability и condition behavior для shortlist кандидатов.",
                "success_criteria": [
                    "Condition stability measured.",
                    "Weak regimes identified.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["stability", "analysis"],
                "runtime_profile": "generic_read",
            },
        )
    )

    assert "Backtests analysis protocol:" in prompt
    assert "`research_memory.search` is context recovery only" in prompt
    assert "switch immediately to `backtests_conditions`" in prompt


def test_backtests_analysis_protocol_includes_run_id_guidance_when_conditions_present() -> None:
    """Slices with backtests_conditions must get run_id guidance."""
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_conditions", "backtests_analysis", "backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s8b",
                "title": "Stability и condition analysis",
                "hypothesis": "h",
                "objective": "Condition behavior analysis",
                "success_criteria": ["Condition stability measured."],
                "evidence_requirements": [],
                "policy_tags": ["stability"],
                "runtime_profile": "backtests_stability_analysis",
            },
        )
    )

    assert "CRITICAL" in prompt
    assert "backtests_conditions(action='run')" in prompt
    assert "requires an explicit `run_id`" in prompt
    assert "saved_runs[].run_id" in prompt
    assert "Do NOT call `backtests_runs(action='start')`" in prompt


def test_backtests_analysis_protocol_omits_run_id_guidance_without_conditions() -> None:
    """Slices without backtests_conditions should NOT get run_id guidance."""
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_analysis", "backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s8c",
                "title": "Standalone backtests analysis",
                "hypothesis": "h",
                "objective": "Analyze backtest results",
                "success_criteria": ["Analysis complete."],
                "evidence_requirements": [],
                "policy_tags": ["backtesting"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "CRITICAL" not in prompt
    assert "requires an explicit `run_id`" not in prompt


def test_direct_prompt_prioritizes_dependency_backtests_handoff_facts_over_low_signal_telemetry() -> None:
    known_facts = {
        "stage_6.feature_long_job": "cond-f07199b451c1",
        "stage_6.feature_short_job": "cond-8972f7822bb2",
        "stage_6.diagnostics_run": "20260411-193208-40dbd831",
        "stage_6.direct.created_ids": ["1", "ddc3dce8bbc14437877f5a1f045b3d2b"],
        "stage_6.direct.successful_tool_names": ["research_memory", "backtests_conditions"],
        "stage_5.direct.provider": "lmstudio",
    }
    for index in range(20):
        known_facts[f"prior.stage_{index}.direct.created_ids"] = [str(index)]

    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            provider="lmstudio",
            allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
            slice_payload={
                "slice_id": "s9",
                "title": "Integration",
                "hypothesis": "h",
                "objective": "Integrate surviving candidates over the base.",
                "success_criteria": ["integration"],
                "evidence_requirements": [],
                "depends_on": ["stage_6"],
            },
            required_prerequisite_facts=[
                "research.project_id",
                "backtests.candidate_handles",
                "backtests.analysis_refs",
            ],
            known_facts=known_facts,
        )
    )

    assert "stage_6.feature_long_job = cond-f07199b451c1" in prompt
    assert "stage_6.feature_short_job = cond-8972f7822bb2" in prompt
    assert "stage_6.diagnostics_run = 20260411-193208-40dbd831" in prompt
    assert "stage_6.direct.created_ids" not in prompt
    assert "stage_6.direct.successful_tool_names" not in prompt


def test_backtests_analysis_prompt_includes_layer_compare_recovery_guidance() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_conditions", "backtests_analysis", "research_memory"],
            slice_payload={
                "slice_id": "s10",
                "title": "Stability и condition analysis",
                "hypothesis": "h",
                "objective": "Проверить stability и condition behavior для shortlist кандидатов.",
                "success_criteria": [
                    "Condition stability measured.",
                    "Weak regimes identified.",
                ],
                "evidence_requirements": [],
                "policy_tags": ["stability", "analysis"],
                "runtime_profile": "generic_read",
            },
        )
    )

    assert "Layer compare failure recovery:" in prompt
    assert "layer_compare" in prompt
    assert "incompatible execution profiles" in prompt
    assert "do NOT retry it" in prompt
    assert "Never spend more than one attempt on layer_compare" in prompt


def test_prompt_surfaces_missing_downstream_prerequisites() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s11",
                "title": "Generic mutation",
                "hypothesis": "h",
                "objective": "Run backtests",
                "success_criteria": [],
                "evidence_requirements": [],
                "runtime_profile": "generic_mutation",
            },
            known_facts={
                "direct.missing_downstream_prerequisites": [
                    "backtests.candidate_handles",
                    "backtests.analysis_refs",
                ],
            },
        )
    )

    assert "Downstream slices need these facts from you:" in prompt
    assert "`backtests.candidate_handles`" in prompt
    assert "`backtests.analysis_refs`" in prompt
    assert "Your final_report.facts MUST include these before completion." in prompt


def test_prompt_no_downstream_section_when_not_missing() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_memory"],
            slice_payload={
                "slice_id": "s12",
                "title": "Summary",
                "hypothesis": "h",
                "objective": "Summarize results",
                "success_criteria": [],
                "evidence_requirements": [],
            },
            known_facts={},
        )
    )

    assert "Downstream slices need these facts from you:" not in prompt


def test_backtests_protocol_includes_layer_compare_line_when_analysis_available() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_analysis", "backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s13",
                "title": "Standalone backtests новых сигналов",
                "hypothesis": "h",
                "objective": "Запустить честные standalone бэктесты для shortlist кандидатов.",
                "success_criteria": ["Есть shortlist standalone-кандидатов."],
                "evidence_requirements": [],
                "policy_tags": ["backtesting"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "layer_compare requires compatible runs" in prompt
    assert "fall back to individual diagnostics" in prompt


def test_backtests_protocol_excludes_layer_compare_when_analysis_missing() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["backtests_runs", "research_memory"],
            slice_payload={
                "slice_id": "s14",
                "title": "Standalone backtests",
                "hypothesis": "h",
                "objective": "Run standalone backtests.",
                "success_criteria": ["Backtests complete."],
                "evidence_requirements": [],
                "policy_tags": ["backtesting"],
                "runtime_profile": "generic_mutation",
            },
        )
    )

    assert "layer_compare requires compatible runs" not in prompt


# ---------- Fix 3a: Research setup metadata in prompt ----------


def test_research_setup_prompt_includes_metadata_fields() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_project", "research_map", "research_memory"],
            slice_payload={
                "slice_id": "s1",
                "title": "Research setup",
                "hypothesis": "h",
                "objective": "Set up research project",
                "success_criteria": [],
                "evidence_requirements": [],
            },
        )
    )
    assert "metadata.invariants" in prompt
    assert "metadata.naming_convention" in prompt
    assert "research.invariants_recorded=True" in prompt
    assert "research.naming_recorded=True" in prompt


def test_non_research_setup_prompt_no_metadata_instructions() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["features_custom", "features_dataset"],
            slice_payload={
                "slice_id": "s1",
                "title": "Feature construction",
                "hypothesis": "h",
                "objective": "Build feature",
                "success_criteria": [],
                "evidence_requirements": [],
            },
        )
    )
    assert "Research setup protocol:" not in prompt

