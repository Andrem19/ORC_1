from __future__ import annotations

from app.execution_models import BaselineRef
from app.raw_plan_compiler import compile_semantic_raw_plan
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan, SemanticStage
from app.services.mcp_catalog.models import McpCatalogSnapshot, McpToolSpec
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _document() -> RawPlanDocument:
    return RawPlanDocument(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_1",
        title="Plan v1",
        version_label="v1",
        normalized_text="# Plan v1",
        parse_confidence=0.75,
    )


def _semantic_plan() -> SemanticRawPlan:
    return SemanticRawPlan(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_1",
        source_title="Plan v1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Setup stage",
                objective="Open research memory",
                actions=["Open project"],
                success_criteria=["Project opened"],
                tool_hints=["research_memory"],
                policy_tags=["setup"],
            ),
            SemanticStage(
                stage_id="stage_2",
                title="Feature stage",
                objective="Validate feature contract",
                actions=["Check features"],
                success_criteria=["Contract inspected"],
                tool_hints=["feature_contract"],
                required=False,
            ),
            SemanticStage(
                stage_id="stage_3",
                title="Backtest stage",
                objective="Run backtest",
                actions=["Start backtest"],
                success_criteria=["Backtest started"],
                tool_hints=["backtesting"],
                parallelizable=True,
            ),
            SemanticStage(
                stage_id="stage_4",
                title="Final stage",
                objective="Summarize results",
                actions=["Write final verdict"],
                success_criteria=["Final verdict ready"],
                tool_hints=["finalization"],
            ),
        ],
    )


def test_compile_semantic_raw_plan_batches_by_three_slices() -> None:
    sequence = compile_semantic_raw_plan(
        _document(),
        _semantic_plan(),
        semantic_method="llm",
        catalog_snapshot=make_catalog_snapshot(),
    )

    assert sequence.report.compile_status == "compiled"
    # backtesting hint (6 tools) gets split → 5 stages total
    assert sequence.report.stage_count == 5
    assert sequence.report.compiled_plan_count == 2
    assert len(sequence.plans[0].slices) == 3
    assert len(sequence.plans[1].slices) == 2


def test_compile_semantic_raw_plan_inferrs_tools_budgets_and_optional_tag() -> None:
    sequence = compile_semantic_raw_plan(
        _document(),
        _semantic_plan(),
        semantic_method="llm",
        catalog_snapshot=make_catalog_snapshot(),
    )

    setup_slice = sequence.plans[0].slices[0]
    feature_slice = sequence.plans[0].slices[1]
    backtest_exploration = sequence.plans[0].slices[2]

    assert "research_project" in setup_slice.allowed_tools
    assert feature_slice.max_turns == 36
    assert feature_slice.max_tool_calls == 30
    assert "optional_candidate" in feature_slice.policy_tags
    # stage_3 is split: part1 (exploration) has backtests_plan, backtests_conditions, backtests_analysis
    assert "backtests_plan" in backtest_exploration.allowed_tools
    # part1 is not parallelizable (sub-stages are always sequential)
    assert backtest_exploration.parallel_slot == 1


def test_compile_semantic_raw_plan_auto_adds_research_record_for_documentation_stage() -> None:
    semantic = _semantic_plan()
    semantic.stages[0] = SemanticStage(
        stage_id="stage_1",
        title="Setup stage",
        objective="Document methodology and record postmortem rules",
        actions=["Open project", "Record methodology milestone"],
        success_criteria=["Postmortem documented"],
        tool_hints=["research_project"],
        policy_tags=["setup"],
    )

    sequence = compile_semantic_raw_plan(
        _document(),
        semantic,
        semantic_method="llm",
        catalog_snapshot=make_catalog_snapshot(),
    )

    setup_slice = sequence.plans[0].slices[0]
    assert "research_record" in setup_slice.allowed_tools
    assert setup_slice.max_expensive_calls >= 2
    assert any("Auto-added research_record" in warning for warning in sequence.report.warnings)


def test_compile_semantic_raw_plan_does_not_split_small_stages() -> None:
    """Regression guard: stages with <=5 resolved tools must NOT be split."""
    semantic = SemanticRawPlan(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_nosplit",
        source_title="No split plan",
        goal="Test",
        baseline_ref=BaselineRef(snapshot_id="snap_1", version=1),
        global_constraints=[],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Setup",
                objective="Setup",
                actions=["Do it"],
                success_criteria=["Done"],
                tool_hints=["research_memory"],  # 4 tools
            ),
            SemanticStage(
                stage_id="stage_2",
                title="Features",
                objective="Check features",
                actions=["Check"],
                success_criteria=["Checked"],
                tool_hints=["feature_contract"],  # 4 tools
                depends_on=["stage_1"],
            ),
            SemanticStage(
                stage_id="stage_3",
                title="Finalize",
                objective="Summarize",
                actions=["Summarize"],
                success_criteria=["Summarized"],
                tool_hints=["finalization"],  # 3 tools
                depends_on=["stage_2"],
            ),
        ],
    )
    sequence = compile_semantic_raw_plan(
        _document(),
        semantic,
        semantic_method="llm",
        catalog_snapshot=make_catalog_snapshot(),
    )
    assert sequence.report.stage_count == 3
    all_ids = [s.slice_id for p in sequence.plans for s in p.slices]
    assert not any("_part1" in sid for sid in all_ids)
    assert not any("_part2" in sid for sid in all_ids)


def test_compile_semantic_raw_plan_assigns_research_shortlist_profile() -> None:
    semantic = SemanticRawPlan(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_shortlist",
        source_title="Shortlist plan",
        goal="Find new signal classes",
        baseline_ref=BaselineRef(snapshot_id="snap_1", version=1),
        global_constraints=[],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Form first-wave shortlist",
                objective="Assemble novel signal families and explain why they are not duplicates of v1-v12.",
                actions=["Inspect research memory", "Write shortlist milestone"],
                success_criteria=["Shortlist exists", "Each family has novelty justification"],
                tool_hints=["research"],
                policy_tags=["hypothesis_formation"],
            ),
        ],
    )

    sequence = compile_semantic_raw_plan(
        _document(),
        semantic,
        semantic_method="llm",
        catalog_snapshot=make_catalog_snapshot(),
    )

    shortlist_slice = sequence.plans[0].slices[0]
    assert shortlist_slice.runtime_profile == "research_shortlist"
    assert shortlist_slice.required_output_facts == [
        "research.project_id",
        "research.shortlist_families",
        "research.novelty_justification_present",
    ]
    assert shortlist_slice.required_prerequisite_facts == []
    assert shortlist_slice.finalization_mode == "fact_based"


def test_compile_semantic_raw_plan_fails_when_upstream_does_not_produce_required_prerequisite_fact() -> None:
    semantic = SemanticRawPlan(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_invalid",
        source_title="Invalid plan",
        goal="Validate prerequisite invariant",
        baseline_ref=BaselineRef(snapshot_id="snap_1", version=1),
        global_constraints=[],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Write result",
                objective="Record setup result",
                actions=["Write result"],
                success_criteria=["Project recorded"],
                tool_hints=["research"],
                policy_tags=["setup"],
            ),
            SemanticStage(
                stage_id="stage_2",
                title="Integration analysis",
                objective="Run integration checks",
                actions=["Inspect candidate handles"],
                success_criteria=["Integration testing complete"],
                tool_hints=["backtests_analysis", "research_memory"],
                policy_tags=["integration"],
                depends_on=["stage_1"],
            ),
        ],
    )

    snapshot = McpCatalogSnapshot(
        server_name="dev_space1",
        endpoint_url="http://127.0.0.1:8766/mcp",
        schema_hash="hash_custom",
        fetched_at="2026-04-14T00:00:00Z",
        tools=[
            McpToolSpec(
                name="research_project",
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
                produced_handle_fields=["project_id", "memory_node_id"],
            ),
            McpToolSpec(
                name="backtests_analysis",
                side_effects="mutating",
                supports_polling=True,
                accepted_handle_fields=["run_id"],
                produced_handle_fields=["run_id"],
            ),
        ],
    )

    sequence = compile_semantic_raw_plan(
        _document(),
        semantic,
        semantic_method="llm",
        catalog_snapshot=snapshot,
    )

    # With ID prerequisites removed (MCP handles IDs), the plan compiles successfully.
    # Backtests integration no longer requires candidate_handles from upstream.
    assert sequence.report.compile_status == "compiled"
