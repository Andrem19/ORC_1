"""Tests for plan-store persistence around planner artifacts and future-compatible plan payloads."""

from __future__ import annotations

import json
from pathlib import Path

from app.plan_models import PlanStep, PlanTask, ResearchPlan, decision_gate_from_dict
from app.planner_runtime import PlannerRunSnapshot
from app.plan_store import PlanStore


def test_planner_run_snapshot_to_dict_persists_structured_fields() -> None:
    snapshot = PlannerRunSnapshot(
        request_type="create",
        request_version=1,
        attempt_number=1,
        prompt_length=9000,
        output_mode="stream-json",
        raw_stdout='{"type":"stream_event"}',
        raw_stderr="warn",
        rendered_output="Structured output provided successfully",
        structured_payload={"schema_version": 3, "plan_version": 1, "tasks": []},
        structured_payload_source="tool_use_input",
        structured_payload_bytes=123,
        structured_delta_bytes=45,
        transport_errors=["transport issue"],
        parse_status="parsed_from_tool_use_input",
    )

    payload = snapshot.to_dict()

    assert payload["raw_stream_transcript"] == '{"type":"stream_event"}'
    assert payload["rendered_output_clean"] == "Structured output provided successfully"
    assert payload["structured_payload"]["plan_version"] == 1
    assert payload["structured_payload_source"] == "tool_use_input"
    assert payload["transport_errors"] == ["transport issue"]
    assert payload["parse_status"] == "parsed_from_tool_use_input"
    assert "timing_summary" in payload


def test_save_rejected_plan_attempt_persists_failure_metadata(tmp_path: Path) -> None:
    store = PlanStore(str(tmp_path / "plans"))

    path = store.save_rejected_plan_attempt(
        plan_version=1,
        attempt_number=1,
        attempt_type="create",
        raw_output="Structured output provided successfully",
        parsed_data={"plan_action": "continue"},
        validation_errors=[],
        failure_class="transport_error",
        request_type="create",
        structured_payload={"schema_version": 3, "plan_version": 1, "tasks": []},
        planner_run_artifact="/tmp/create_v1_attempt_1.json",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["failure_class"] == "transport_error"
    assert payload["request_type"] == "create"
    assert payload["plan_version"] == 1
    assert payload["planner_run_artifact"] == "/tmp/create_v1_attempt_1.json"
    assert payload["structured_payload"]["plan_version"] == 1


def test_plan_store_roundtrip_preserves_steps_baseline_and_gate_reason(tmp_path: Path) -> None:
    store = PlanStore(str(tmp_path / "plans"))
    plan = ResearchPlan(
        version=1,
        goal="test",
        planner_run_artifact="/tmp/planner_run.json",
        baseline_run_id="baseline-run",
        baseline_snapshot_ref="active-signal-v1@1",
        baseline_metrics={"sharpe": 1.1, "trades": 10},
        tasks=[
            PlanTask(
                task_id="stage-0",
                plan_version=1,
                stage_number=0,
                stage_name="Baseline",
                steps=[
                    PlanStep(
                        step_id="baseline_run",
                        kind="tool_call",
                        instruction="Run baseline",
                        tool_name="backtests_runs",
                        args={"action": "start"},
                    )
                ],
                decision_gates=[],
            )
        ],
    )
    plan.tasks[0].decision_gates = []
    plan.tasks[0].decision_gates.append(
        decision_gate_from_dict(
            {
                "metric": "sharpe",
                "threshold": 1.0,
                "reason": "Baseline should reproduce expected Sharpe.",
                "future_field": "ignored",
            }
        )
    )

    store.save_plan(plan)
    loaded = store.load_plan(1)

    assert loaded is not None
    assert loaded.planner_run_artifact == "/tmp/planner_run.json"
    assert loaded.baseline_run_id == "baseline-run"
    assert loaded.baseline_snapshot_ref == "active-signal-v1@1"
    assert loaded.baseline_metrics["trades"] == 10
    assert loaded.tasks[0].steps[0].step_id == "baseline_run"
    assert loaded.tasks[0].decision_gates[0].reason == "Baseline should reproduce expected Sharpe."


def test_load_plan_ignores_unknown_decision_gate_fields(tmp_path: Path) -> None:
    store = PlanStore(str(tmp_path / "plans"))
    store.ensure_dirs()
    json_path = Path(tmp_path / "plans" / "plan_v1.json")
    md_path = Path(tmp_path / "plans" / "plan_v1.md")
    md_path.write_text("# plan", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "version": 1,
                "goal": "test",
                "tasks": [
                    {
                        "task_id": "stage-0",
                        "plan_version": 1,
                        "stage_number": 0,
                        "stage_name": "Baseline",
                        "steps": [],
                        "decision_gates": [
                            {
                                "metric": "sharpe",
                                "threshold": 1.0,
                                "reason": "keep",
                                "future_field": "ignored",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = store.load_plan(1)

    assert loaded is not None
    assert loaded.tasks[0].decision_gates[0].metric == "sharpe"
    assert loaded.tasks[0].decision_gates[0].reason == "keep"
