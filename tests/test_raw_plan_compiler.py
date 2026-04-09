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
    assert sequence.report.stage_count == 4
    assert sequence.report.compiled_plan_count == 2
    assert len(sequence.plans[0].slices) == 3
    assert len(sequence.plans[1].slices) == 1


def test_compile_semantic_raw_plan_inferrs_tools_budgets_and_optional_tag() -> None:
    sequence = compile_semantic_raw_plan(_document(), _semantic_plan(), semantic_method="llm")

    setup_slice = sequence.plans[0].slices[0]
    feature_slice = sequence.plans[0].slices[1]
    backtest_slice = sequence.plans[0].slices[2]

    assert "research_project" in setup_slice.allowed_tools
    assert feature_slice.max_turns == 6
    assert "optional_candidate" in feature_slice.policy_tags
    assert "backtests_runs" in backtest_slice.allowed_tools
    assert backtest_slice.parallel_slot == 3
