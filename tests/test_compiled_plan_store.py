from __future__ import annotations

import json
from pathlib import Path

from app.compiled_plan_store import CompiledPlanStore
from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.raw_plan_models import CompileReport, CompiledPlanSequence, SemanticRawPlan, SemanticStage


def _sequence() -> CompiledPlanSequence:
    semantic = SemanticRawPlan(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_1",
        source_title="Plan v1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        stages=[
            SemanticStage(
                stage_id="stage_1",
                title="Stage 1",
                objective="Objective 1",
                actions=["Action 1"],
                success_criteria=["Done 1"],
            )
        ],
    )
    plan = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal=semantic.goal,
        baseline_ref=semantic.baseline_ref,
        global_constraints=semantic.global_constraints,
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Stage 1",
                hypothesis="Objective 1",
                objective="Objective 1",
                success_criteria=["Done 1"],
                allowed_tools=["events"],
                evidence_requirements=["Done 1"],
                policy_tags=["analysis"],
                max_turns=4,
                max_tool_calls=3,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    return CompiledPlanSequence(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash_1",
        sequence_id="compiled_plan_v1",
        semantic_plan=semantic,
        plans=[plan],
        report=CompileReport(
            source_file="raw_plans/plan_v1.md",
            source_hash="hash_1",
            sequence_id="compiled_plan_v1",
            compile_status="compiled",
            parser_confidence=0.8,
            semantic_method="llm",
            stage_count=1,
            compiled_plan_count=1,
        ),
    )


def test_compiled_plan_store_round_trip(tmp_path) -> None:
    store = CompiledPlanStore(tmp_path / "compiled")
    manifest_path = store.save_sequence(_sequence())
    manifests = store.load_manifests()

    assert manifest_path.exists()
    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.sequence_id == "compiled_plan_v1"
    assert manifest.semantic_path == "semantic.json"
    assert manifest.compile_report_path == "compile_report.json"
    loaded = store.load_plan(manifest, manifest.plan_files[0])
    assert loaded.plan_id == "compiled_plan_v1_batch_1"
    assert loaded.slices[0].allowed_tools == ["events"]
    assert loaded.plan_source_kind == "compiled_raw"
    assert loaded.source_sequence_id == "compiled_plan_v1"
    assert loaded.source_manifest_path.endswith("manifest.json")


def test_compiled_plan_store_load_manifests_uses_natural_raw_file_order(tmp_path) -> None:
    store = CompiledPlanStore(tmp_path / "compiled")
    for name in ("plan_v10.md", "plan_v2.md", "plan_v1.md"):
        sequence = _sequence()
        sequence.source_file = f"raw_plans/{name}"
        sequence.sequence_id = f"compiled_{name[:-3]}"
        sequence.report.source_file = sequence.source_file
        sequence.report.sequence_id = sequence.sequence_id
        sequence.plans[0].plan_id = f"{sequence.sequence_id}_batch_1"
        store.save_sequence(sequence)

    manifests = store.load_manifests()

    assert [Path(item.source_file).name for item in manifests] == ["plan_v1.md", "plan_v2.md", "plan_v10.md"]


def test_compiled_plan_store_backfills_slice_dependencies_from_semantic(tmp_path) -> None:
    store = CompiledPlanStore(tmp_path / "compiled")
    sequence = _sequence()
    sequence.semantic_plan.stages = [
        SemanticStage(
            stage_id="stage_1",
            title="Stage 1",
            objective="Objective 1",
            actions=["Action 1"],
            success_criteria=["Done 1"],
        ),
        SemanticStage(
            stage_id="stage_2",
            title="Stage 2",
            objective="Objective 2",
            actions=["Action 2"],
            success_criteria=["Done 2"],
            depends_on=["stage_1"],
        ),
    ]
    sequence.plans[0].slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_1",
            title="Stage 1",
            hypothesis="Objective 1",
            objective="Objective 1",
            success_criteria=["Done 1"],
            allowed_tools=["events"],
            evidence_requirements=["Done 1"],
            policy_tags=["analysis"],
            max_turns=4,
            max_tool_calls=3,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
        PlanSlice(
            slice_id="compiled_plan_v1_stage_2",
            title="Stage 2",
            hypothesis="Objective 2",
            objective="Objective 2",
            success_criteria=["Done 2"],
            allowed_tools=["events"],
            evidence_requirements=["Done 2"],
            policy_tags=["analysis"],
            max_turns=4,
            max_tool_calls=3,
            max_expensive_calls=0,
            parallel_slot=1,
        ),
    ]
    manifest_path = store.save_sequence(sequence)

    plan_file = manifest_path.parent / "plans" / f"{sequence.plans[0].plan_id}.json"
    payload = json.loads(plan_file.read_text(encoding="utf-8"))
    for item in payload.get("slices", []):
        item.pop("depends_on", None)
    plan_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = store.load_manifests()[0]
    loaded = store.load_plan(manifest, manifest.plan_files[0])

    assert loaded.slices[1].depends_on == ["compiled_plan_v1_stage_1"]
