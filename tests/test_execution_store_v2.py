from __future__ import annotations

import json
from pathlib import Path

from app.execution_models import BaselineRef, DirectAttemptMetadata, ExecutionPlan, ExecutionStateV2, ExecutionTurn, PlanSlice, WorkerAction
from app.execution_store import ExecutionStateStore


def test_execution_state_store_round_trip(tmp_path) -> None:
    store = ExecutionStateStore(tmp_path / "state" / "execution_state_v2.json")
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Test goal",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan="raw_plans/plan_v1.md",
        source_manifest_path="compiled_plans/plan_v1/manifest.json",
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Slice",
                hypothesis="H",
                objective="O",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["summary"],
                policy_tags=["cheap_first"],
                max_turns=3,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="checkpointed",
                facts={"rows": 42},
            )
        ],
        status="running",
    )
    state = ExecutionStateV2(goal="Test goal", status="running", plans=[plan], current_plan_id="plan_1")

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.current_plan_id == "plan_1"
    assert loaded.plans[0].slices[0].facts["rows"] == 42
    assert loaded.plans[0].source_sequence_id == "compiled_plan_v1"
    assert loaded.plans[0].source_manifest_path.endswith("manifest.json")


def test_execution_state_store_archives_legacy_runtime(tmp_path) -> None:
    state_dir = tmp_path / "state"
    plans_dir = tmp_path / "plans"
    state_dir.mkdir()
    plans_dir.mkdir()
    legacy_state = state_dir / "orchestrator_state.json"
    legacy_state.write_text(json.dumps({"legacy": True}), encoding="utf-8")
    (plans_dir / "plan_v1.md").write_text("# old plan", encoding="utf-8")

    store = ExecutionStateStore(state_dir / "execution_state_v2.json")
    archive_path = store.archive_legacy_runtime(
        legacy_state_path=legacy_state,
        legacy_plan_dir=plans_dir,
    )

    assert archive_path is not None
    assert (archive_path / "orchestrator_state.json").exists()
    assert (archive_path / "plans" / "plan_v1.md").exists()
    assert not legacy_state.exists()
    assert not (plans_dir / "plan_v1.md").exists()


def test_execution_state_store_load_ignores_additive_fields(tmp_path) -> None:
    state_path = tmp_path / "state" / "execution_state_v2.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "goal": "Test goal",
                "status": "running",
                "plans": [
                    {
                        "plan_id": "plan_1",
                        "goal": "Test goal",
                        "baseline_ref": {"snapshot_id": "active-signal-v1", "version": 1, "unexpected": "ignored"},
                        "global_constraints": [],
                        "slices": [
                            {
                                "slice_id": "slice_1",
                                "title": "Slice",
                                "hypothesis": "H",
                                "objective": "O",
                                "success_criteria": ["done"],
                                "allowed_tools": ["events"],
                                "evidence_requirements": ["summary"],
                                "policy_tags": ["cheap_first"],
                                "max_turns": 3,
                                "max_tool_calls": 1,
                                "max_expensive_calls": 0,
                                "parallel_slot": 1,
                                "unknown_field": "ignored",
                            }
                        ],
                    }
                ],
                "removed_legacy_field": {"status": "ignored"},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    store = ExecutionStateStore(state_path)

    loaded = store.load()

    assert loaded is not None
    assert loaded.plans[0].baseline_ref.snapshot_id == "active-signal-v1"
    assert loaded.plans[0].slices[0].slice_id == "slice_1"
    assert loaded.turn_history == []


def test_execution_state_store_trims_completed_turns_but_keeps_active_plan_turns(tmp_path) -> None:
    store = ExecutionStateStore(tmp_path / "state" / "execution_state_v2.json")
    state = ExecutionStateV2(goal="Test goal", status="running", current_plan_id="plan_active")
    state.turn_history = [
        ExecutionTurn(
            turn_id=f"turn_done_{index}",
            plan_id="plan_done",
            slice_id="slice_1",
            worker_id="worker-1",
            turn_index=index,
            action=WorkerAction(action_id=f"action_done_{index}", action_type="checkpoint", summary=f"done {index}"),
        )
        for index in range(60)
    ] + [
        ExecutionTurn(
            turn_id=f"turn_active_{index}",
            plan_id="plan_active",
            slice_id="slice_active",
            worker_id="worker-1",
            turn_index=index,
            action=WorkerAction(action_id=f"action_active_{index}", action_type="checkpoint", summary=f"active {index}"),
        )
        for index in range(3)
    ]

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert [turn.turn_id for turn in loaded.turn_history[:3]] == ["turn_done_10", "turn_done_11", "turn_done_12"]
    assert [turn.turn_id for turn in loaded.turn_history[-3:]] == ["turn_active_0", "turn_active_1", "turn_active_2"]
    assert len([turn for turn in loaded.turn_history if turn.plan_id == "plan_active"]) == 3
    assert len([turn for turn in loaded.turn_history if turn.plan_id == "plan_done"]) == 50


def test_execution_state_store_persists_direct_attempt_metadata(tmp_path) -> None:
    store = ExecutionStateStore(tmp_path / "state" / "execution_state_v2.json")
    state = ExecutionStateV2(goal="Test goal", status="running", current_plan_id="plan_active")
    state.turn_history = [
        ExecutionTurn(
            turn_id="turn_active_1",
            plan_id="plan_done",
            slice_id="slice_1",
            worker_id="worker-1",
            turn_index=1,
            action=WorkerAction(action_id="a1", action_type="checkpoint", status="partial", summary="ok"),
            direct_attempt=DirectAttemptMetadata(provider="test", tool_call_count=2, expensive_tool_call_count=1),
        )
    ]

    store.save(state)
    loaded = store.load()

    assert loaded is not None
    assert loaded.turn_history[0].direct_attempt.provider == "test"
    assert loaded.turn_history[0].direct_attempt.tool_call_count == 2
