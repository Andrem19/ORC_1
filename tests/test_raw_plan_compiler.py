from __future__ import annotations

from app.execution_models import BaselineRef
from app.raw_plan_compiler import compile_semantic_raw_plan
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan, SemanticStage


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
    sequence = compile_semantic_raw_plan(_document(), _semantic_plan(), semantic_method="llm")

    assert sequence.report.compile_status == "compiled"
    # backtesting hint (6 tools) gets split → 5 stages total
    assert sequence.report.stage_count == 5
    assert sequence.report.compiled_plan_count == 2
    assert len(sequence.plans[0].slices) == 3
    assert len(sequence.plans[1].slices) == 2


def test_compile_semantic_raw_plan_inferrs_tools_budgets_and_optional_tag() -> None:
    sequence = compile_semantic_raw_plan(_document(), _semantic_plan(), semantic_method="llm")

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

    sequence = compile_semantic_raw_plan(_document(), semantic, semantic_method="llm")

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
    sequence = compile_semantic_raw_plan(_document(), semantic, semantic_method="llm")
    assert sequence.report.stage_count == 3
    all_ids = [s.slice_id for p in sequence.plans for s in p.slices]
    assert not any("_part1" in sid for sid in all_ids)
    assert not any("_part2" in sid for sid in all_ids)
