"""Tests for stage splitting logic in raw_plan_compiler."""

from __future__ import annotations

from app.execution_models import BaselineRef
from app.raw_plan_compiler import (
    _classify_split_tools,
    _expand_stages,
    _maybe_split_stage,
    compile_semantic_raw_plan,
)
from app.raw_plan_models import RawPlanDocument, SemanticRawPlan, SemanticStage
from app.services.mcp_catalog.classifier import build_family_tool_map
from tests.mcp_catalog_fixtures import make_catalog_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SNAPSHOT = make_catalog_snapshot()
_TOOL_SET = _SNAPSHOT.tool_name_set()
_FAMILY_MAP = build_family_tool_map(_SNAPSHOT)


def _document() -> RawPlanDocument:
    return RawPlanDocument(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_split",
        title="Plan v1",
        version_label="v1",
        normalized_text="# Plan v1",
        parse_confidence=0.75,
    )


def _stage(**overrides) -> SemanticStage:
    defaults = dict(
        stage_id="stage_test",
        title="Test stage",
        objective="Test objective",
        actions=["Do something"],
        success_criteria=["Done"],
        tool_hints=["research_memory"],
        policy_tags=["setup"],
    )
    defaults.update(overrides)
    return SemanticStage(**defaults)


def _big_stage(stage_id: str = "stage_3", **overrides) -> SemanticStage:
    """Stage with 10 tools — mimics real stage_3 (data contract)."""
    defaults = dict(
        stage_id=stage_id,
        title="Data contract and feature contract for each new family",
        objective="Fix exact data and feature contracts per hypothesis",
        actions=[
            "Create feature contracts",
            "Validate and publish custom features",
            "Prepare model-backed feature training loops",
            "Check event alignment",
            "Document feature interpretation",
        ],
        success_criteria=[
            "Clean data/feature contract per hypothesis",
            "Leakage checked per contract",
            "Custom features validated and published",
            "Event alignment checked for event-hypotheses",
        ],
        tool_hints=[
            "datasets", "datasets_preview", "features_catalog",
            "features_dataset", "features_custom", "features_analytics",
            "events", "events_sync", "models_dataset", "research_record",
        ],
        policy_tags=["data_contract", "leakage_guard"],
        depends_on=["stage_2"],
        required=True,
        parallelizable=True,
    )
    defaults.update(overrides)
    return SemanticStage(**defaults)


# ---------------------------------------------------------------------------
# _classify_split_tools
# ---------------------------------------------------------------------------


class TestClassifySplitTools:
    def test_returns_none_below_threshold(self) -> None:
        tools = ["research_project", "research_map", "research_record"]
        assert _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT) is None

    def test_returns_none_at_exact_threshold(self) -> None:
        tools = ["research_project", "research_map", "research_record", "datasets", "features_catalog"]
        assert _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT) is None

    def test_returns_none_all_exploration(self) -> None:
        tools = ["datasets", "features_catalog", "events", "research_project", "research_map", "features_analytics"]
        assert _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT) is None

    def test_returns_none_all_construction(self) -> None:
        tools = ["features_dataset", "features_custom", "events_sync", "models_dataset",
                 "backtests_strategy", "backtests_runs"]
        assert _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT) is None

    def test_splits_mixed_tools(self) -> None:
        tools = [
            "datasets", "features_catalog", "events",  # exploration
            "features_dataset", "features_custom", "events_sync",  # construction
            "research_record",  # both
        ]
        result = _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT)
        assert result is not None
        exploration, construction = result
        assert "research_record" in exploration
        assert "research_record" in construction
        assert "datasets" in exploration
        assert "features_dataset" in construction
        assert "datasets" not in construction
        assert "features_dataset" not in exploration

    def test_big_stage_produces_split(self) -> None:
        tools = [
            "datasets", "datasets_preview", "features_catalog",
            "features_dataset", "features_custom", "features_analytics",
            "events", "events_sync", "models_dataset", "research_record",
        ]
        result = _classify_split_tools(tools, catalog_snapshot=_SNAPSHOT)
        assert result is not None
        exploration, construction = result
        assert len(exploration) >= 4
        assert len(construction) >= 4
        assert "research_record" in exploration
        assert "research_record" in construction


# ---------------------------------------------------------------------------
# _maybe_split_stage
# ---------------------------------------------------------------------------


class TestMaybeSplitStage:
    def test_returns_single_for_small_stage(self) -> None:
        stage = _stage(tool_hints=["research_memory"])  # 4 tools
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert len(result) == 1
        assert result[0].stage_id == "stage_test"

    def test_returns_single_for_exploration_only(self) -> None:
        stage = _stage(tool_hints=["datasets", "datasets_preview", "features_catalog",
                                   "events", "research_project", "research_map"])
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert len(result) == 1

    def test_splits_big_stage_into_two(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert len(result) == 2
        assert result[0].stage_id == "stage_3_part1"
        assert result[1].stage_id == "stage_3_part2"

    def test_part1_keeps_original_depends_on(self) -> None:
        stage = _big_stage(depends_on=["stage_2"])
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert result[0].depends_on == ["stage_2"]

    def test_part2_depends_on_part1(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert result[1].depends_on == ["stage_3_part1"]

    def test_sub_stages_not_parallelizable(self) -> None:
        stage = _big_stage(parallelizable=True)
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert result[0].parallelizable is False
        assert result[1].parallelizable is False

    def test_part1_gets_exploration_tools(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        part1_tools = result[0].tool_hints
        assert "datasets" in part1_tools
        assert "features_catalog" in part1_tools
        assert "events" in part1_tools
        assert "research_record" in part1_tools
        # construction tools NOT in part1
        assert "features_dataset" not in part1_tools
        assert "features_custom" not in part1_tools

    def test_part2_gets_construction_tools(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        part2_tools = result[1].tool_hints
        assert "features_dataset" in part2_tools
        assert "features_custom" in part2_tools
        assert "models_dataset" in part2_tools
        assert "research_record" in part2_tools
        # exploration tools NOT in part2
        assert "datasets" not in part2_tools
        assert "features_catalog" not in part2_tools

    def test_part1_has_extra_success_criterion(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert "Exploration phase complete" in result[0].success_criteria
        assert "Exploration phase complete" not in result[1].success_criteria

    def test_titles_indicate_phase(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert "(exploration)" in result[0].title
        assert "(construction)" in result[1].title

    def test_inherits_objective_and_actions(self) -> None:
        stage = _big_stage()
        result = _maybe_split_stage(stage, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert result[0].objective == stage.objective
        assert result[1].objective == stage.objective
        assert result[0].actions == stage.actions
        assert result[1].actions == stage.actions


# ---------------------------------------------------------------------------
# _expand_stages
# ---------------------------------------------------------------------------


class TestExpandStages:
    def test_no_split_returns_same_count(self) -> None:
        stages = [
            _stage(stage_id="s1", tool_hints=["research_memory"]),   # 4 tools
            _stage(stage_id="s2", tool_hints=["feature_contract"]),  # 4 tools
            _stage(stage_id="s3", tool_hints=["finalization"]),      # 3 tools
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert len(result) == 3
        assert [s.stage_id for s in result] == ["s1", "s2", "s3"]

    def test_expand_one_big_stage(self) -> None:
        stages = [
            _stage(stage_id="s1", tool_hints=["research_memory"]),
            _stage(stage_id="s2", tool_hints=["research_memory"]),
            _big_stage(stage_id="s3", depends_on=["s2"]),
            _stage(stage_id="s4", tool_hints=["analysis"], depends_on=["s3"]),
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        assert len(result) == 5
        ids = [s.stage_id for s in result]
        assert "s3_part1" in ids
        assert "s3_part2" in ids

    def test_remaps_downstream_depends_on(self) -> None:
        stages = [
            _stage(stage_id="s1", tool_hints=["research_memory"]),
            _stage(stage_id="s2", tool_hints=["research_memory"], depends_on=["s1"]),
            _big_stage(stage_id="s3", depends_on=["s2"]),
            _stage(stage_id="s4", tool_hints=["analysis"], depends_on=["s3"]),
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        s4 = next(s for s in result if s.stage_id == "s4")
        assert s4.depends_on == ["s3_part2"]

    def test_part1_keeps_original_dependency(self) -> None:
        stages = [
            _stage(stage_id="s1", tool_hints=["research_memory"]),
            _big_stage(stage_id="s3", depends_on=["s1"]),
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        part1 = next(s for s in result if s.stage_id == "s3_part1")
        assert part1.depends_on == ["s1"]

    def test_part2_depends_on_part1(self) -> None:
        stages = [
            _big_stage(stage_id="s3", depends_on=["s1"]),
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        part2 = next(s for s in result if s.stage_id == "s3_part2")
        assert part2.depends_on == ["s3_part1"]

    def test_non_split_stage_depends_on_non_split(self) -> None:
        stages = [
            _stage(stage_id="s1", tool_hints=["research_memory"]),
            _stage(stage_id="s2", tool_hints=["analysis"], depends_on=["s1"]),
        ]
        result = _expand_stages(stages, catalog_snapshot=_SNAPSHOT, tool_name_set=_TOOL_SET, family_map=_FAMILY_MAP)
        s2 = next(s for s in result if s.stage_id == "s2")
        assert s2.depends_on == ["s1"]  # unchanged


# ---------------------------------------------------------------------------
# Integration: compile_semantic_raw_plan with splitting
# ---------------------------------------------------------------------------


class TestCompileWithSplitting:
    def _semantic_with_big_stage(self) -> SemanticRawPlan:
        return SemanticRawPlan(
            source_file="raw_plans/plan_v1.md",
            source_hash="hash_split",
            source_title="Plan with big stage",
            goal="Test splitting",
            baseline_ref=BaselineRef(snapshot_id="snap_1", version=1),
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
                    title="Hypothesis stage",
                    objective="Form hypotheses",
                    actions=["Create hypothesis"],
                    success_criteria=["Hypotheses formed"],
                    tool_hints=["research_memory"],
                    policy_tags=["hypothesis_formation"],
                    depends_on=["stage_1"],
                ),
                _big_stage(stage_id="stage_3", depends_on=["stage_2"]),
                SemanticStage(
                    stage_id="stage_4",
                    title="Quick filter stage",
                    objective="Filter hypotheses",
                    actions=["Run analytics"],
                    success_criteria=["Filtered"],
                    tool_hints=["features_analytics", "features_catalog", "datasets_preview", "research_record"],
                    policy_tags=["filter_gate"],
                    depends_on=["stage_3"],
                ),
            ],
        )

    def test_stage_count_includes_sub_stages(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        # 4 original stages → stage_3 splits → 5 compiled stages
        assert seq.report.stage_count == 5

    def test_produces_correct_batch_count(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        # 5 stages → batches of 3: [1,2,3a] [3b,4]
        assert seq.report.compiled_plan_count == 2

    def test_slice_ids_include_part1_part2(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        all_ids = [s.slice_id for p in seq.plans for s in p.slices]
        assert "compiled_plan_v1_stage_3_part1" in all_ids
        assert "compiled_plan_v1_stage_3_part2" in all_ids

    def test_part1_has_exploration_tools(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        all_slices = [s for p in seq.plans for s in p.slices]
        part1 = next(s for s in all_slices if s.slice_id.endswith("stage_3_part1"))
        assert "datasets" in part1.allowed_tools
        assert "features_catalog" in part1.allowed_tools
        assert "research_record" in part1.allowed_tools
        assert "features_dataset" not in part1.allowed_tools

    def test_part2_has_construction_tools(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        all_slices = [s for p in seq.plans for s in p.slices]
        part2 = next(s for s in all_slices if s.slice_id.endswith("stage_3_part2"))
        assert "features_dataset" in part2.allowed_tools
        assert "features_custom" in part2.allowed_tools
        assert "research_record" in part2.allowed_tools
        assert "datasets" not in part2.allowed_tools

    def test_dependency_wiring(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        all_slices = [s for p in seq.plans for s in p.slices]
        part1 = next(s for s in all_slices if s.slice_id.endswith("stage_3_part1"))
        part2 = next(s for s in all_slices if s.slice_id.endswith("stage_3_part2"))
        stage4 = next(s for s in all_slices if s.slice_id.endswith("stage_4"))

        # part1 depends on stage_2
        assert "compiled_plan_v1_stage_2" in part1.depends_on
        # part2 depends on part1
        assert "compiled_plan_v1_stage_3_part1" in part2.depends_on
        # stage_4 depends on part2 (remapped from stage_3)
        assert "compiled_plan_v1_stage_3_part2" in stage4.depends_on

    def test_budget_normalization_applied(self) -> None:
        seq = compile_semantic_raw_plan(
            _document(),
            self._semantic_with_big_stage(),
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        all_slices = [s for p in seq.plans for s in p.slices]
        part1 = next(s for s in all_slices if s.slice_id.endswith("stage_3_part1"))
        assert part1.budget_scale_applied == 6
        assert part1.max_tool_calls > 0

    def test_no_split_for_small_stages(self) -> None:
        """Regression: stages with <=5 tools are NOT split."""
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
        seq = compile_semantic_raw_plan(
            _document(),
            semantic,
            semantic_method="llm",
            catalog_snapshot=_SNAPSHOT,
        )
        assert seq.report.stage_count == 3
        all_ids = [s.slice_id for p in seq.plans for s in p.slices]
        assert not any("_part1" in sid for sid in all_ids)
