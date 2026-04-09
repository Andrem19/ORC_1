from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from app.adapters.fake_planner import FakePlanner
from app.compiled_plan_store import CompiledPlanStore
from app.config import OrchestratorConfig
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice, ToolResultEnvelope, WorkerAction
from app.raw_plan_models import CompileReport, CompiledPlanSequence, SemanticRawPlan, SemanticStage
from app.reporting.postrun_builder import PostRunReportBuilder
from app.runtime_incidents import LocalIncidentStore


def test_postrun_builder_skip_llm_does_not_log_degraded_warning(tmp_path, caplog) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="planner",
    )
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[],
        status="completed",
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[plan],
        current_plan_id=plan.plan_id,
        stop_reason="goal_reached",
    )
    builder = PostRunReportBuilder(
        config=cfg,
        planner_adapter=None,
        run_id="",
        skip_llm=True,
    )

    with caplog.at_level(logging.WARNING, logger="orchestrator.reporting"):
        result, _run_report = asyncio.run(builder.build(state=state))

    assert result["plan_reports"] == 1
    assert "Narrative generation degraded" not in caplog.text


def test_postrun_builder_emits_plan_sequence_and_run_reports(tmp_path) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="compiled_raw",
    )
    raw_dir = Path(cfg.raw_plan_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "plan_v1.md"
    raw_file.write_text("# plan_v1\n", encoding="utf-8")

    plan = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan=str(raw_file),
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Slice 1",
                hypothesis="h",
                objective="o",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="completed",
                facts={"funding_rows": 7147},
                artifacts=["artifact_1"],
                final_report_turn_id="turn_final",
                last_summary="Funding route confirmed.",
            )
        ],
        status="completed",
    )
    sequence = CompiledPlanSequence(
        source_file=str(raw_file),
        source_hash="hash_1",
        sequence_id="compiled_plan_v1",
        semantic_plan=SemanticRawPlan(
            source_file=str(raw_file),
            source_hash="hash_1",
            source_title="plan_v1",
            goal="Validate route",
            baseline_ref=plan.baseline_ref,
            global_constraints=["keep baseline fixed"],
            stages=[
                SemanticStage(
                    stage_id="stage_1",
                    title="Stage 1",
                    objective="Validate route",
                    actions=["Inspect route"],
                    success_criteria=["Route validated"],
                )
            ],
        ),
        plans=[plan],
        report=CompileReport(
            source_file=str(raw_file),
            source_hash="hash_1",
            sequence_id="compiled_plan_v1",
            compile_status="compiled",
            parser_confidence=0.8,
            semantic_method="llm",
            stage_count=1,
            compiled_plan_count=1,
        ),
    )
    CompiledPlanStore(cfg.compiled_plan_dir).save_sequence(sequence)

    artifact_store = ExecutionArtifactStore(cfg.plan_dir)
    artifact_store.save_plan(plan)
    artifact_store.save_report(
        plan_id=plan.plan_id,
        slice_id="slice_1",
        turn_id="turn_final",
        payload={
            **json.loads(json.dumps(WorkerAction(
                action_id="action_1",
                action_type="final_report",
                summary="Funding route confirmed.",
                verdict="WATCHLIST",
                key_metrics={"funding_rows": 7147},
                findings=["Funding rows confirmed"],
                next_actions=["Run orthogonality backtest"],
                confidence=0.8,
            ), default=lambda o: o.__dict__)),
            "type": "final_report",
        },
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[plan],
        current_plan_id=plan.plan_id,
        stop_reason="goal_reached",
    )
    state.tool_call_ledger.append(
        ToolResultEnvelope(
            call_id="call_1",
            tool="events",
            ok=True,
            retryable=False,
            duration_ms=15,
            summary="Funding rows loaded.",
            plan_id=plan.plan_id,
            slice_id="slice_1",
        )
    )
    LocalIncidentStore(cfg.state_dir).record(
        summary="test_incident",
        metadata={"plan_id": plan.plan_id, "slice_id": "slice_1", "affected_tool": "events"},
    )

    builder = PostRunReportBuilder(
        config=cfg,
        planner_adapter=FakePlanner(responses=[]),
        run_id="",
        skip_llm=False,
    )

    result, _run_report = asyncio.run(builder.build(state=state))

    assert result["plan_reports"] == 1
    assert result["sequence_reports"] == 1
    run_report_path = Path(result["run_report_json"])
    assert run_report_path.exists()
    payload = json.loads(run_report_path.read_text(encoding="utf-8"))
    assert payload["completed_sequences"] == 1
    assert payload["tool_usage_rollup"]["total_calls"] == 1
    assert (Path(cfg.plan_dir) / "plan_reports" / "compiled_plan_v1_batch_1.json").exists()
    assert (Path(cfg.plan_dir) / "sequence_reports" / "compiled_plan_v1.json.json").exists() is False
    assert (Path(cfg.plan_dir) / "sequence_reports" / "compiled_plan_v1.json").exists()


def test_postrun_builder_disables_narrative_automatically_in_compiled_raw_mode(tmp_path) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="compiled_raw",
    )
    plan = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan=str(Path(cfg.raw_plan_dir) / "plan_v1.md"),
        slices=[],
        status="completed",
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[plan],
        current_plan_id=plan.plan_id,
        stop_reason="goal_reached",
    )

    builder = PostRunReportBuilder(
        config=cfg,
        planner_adapter=FakePlanner(responses=[]),
        run_id="",
        skip_llm=False,
    )

    result, run_report = asyncio.run(builder.build(state=state))
    plan_payload = json.loads((Path(cfg.plan_dir) / "plan_reports" / f"{plan.plan_id}.json").read_text(encoding="utf-8"))

    assert builder.narrative.enabled is False
    assert run_report.narrative_status == "skipped"
    assert plan_payload["narrative_status"] == "skipped"
    assert Path(result["run_report_json"]).exists()


def test_postrun_builder_renders_russian_fallback_markdown(tmp_path) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="planner",
    )
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        slices=[],
        status="completed",
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[plan],
        current_plan_id=plan.plan_id,
        stop_reason="graceful_stop",
    )
    builder = PostRunReportBuilder(
        config=cfg,
        planner_adapter=None,
        run_id="",
        skip_llm=True,
    )

    result, _run_report = asyncio.run(builder.build(state=state))
    markdown = Path(result["run_report_md"]).read_text(encoding="utf-8")

    assert "Причина остановки" in markdown
    assert "Скомпилированных sequence" in markdown
    assert "Прогон `" in markdown
    assert "Completed sequences" not in markdown


def test_postrun_builder_emits_partial_reports_for_graceful_stop_mid_sequence(tmp_path) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="compiled_raw",
    )
    raw_dir = Path(cfg.raw_plan_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "plan_v1.md"
    raw_file.write_text("# plan_v1\n", encoding="utf-8")

    plan_1 = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan=str(raw_file),
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="Slice 1",
                hypothesis="h",
                objective="o",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="completed",
                final_report_turn_id="turn_final",
                last_summary="First batch completed.",
            )
        ],
        status="completed",
    )
    plan_2 = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_2",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan=str(raw_file),
        slices=[
            PlanSlice(
                slice_id="slice_2",
                title="Slice 2",
                hypothesis="h2",
                objective="o2",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    sequence = CompiledPlanSequence(
        source_file=str(raw_file),
        source_hash="hash_1",
        sequence_id="compiled_plan_v1",
            semantic_plan=SemanticRawPlan(
                source_file=str(raw_file),
                source_hash="hash_1",
                source_title="plan_v1",
                goal="Validate route",
                global_constraints=["keep baseline fixed"],
                baseline_ref=plan_1.baseline_ref,
                stages=[
                    SemanticStage(
                        stage_id="stage_1",
                        title="Stage 1",
                        objective="Validate route",
                        actions=["Run first batch"],
                        success_criteria=["First batch completed"],
                    ),
                    SemanticStage(
                        stage_id="stage_2",
                        title="Stage 2",
                        objective="Finish route",
                        actions=["Run second batch"],
                        success_criteria=["Second batch completed"],
                    ),
                ],
            ),
        plans=[plan_1, plan_2],
        report=CompileReport(
            source_file=str(raw_file),
            source_hash="hash_1",
            sequence_id="compiled_plan_v1",
            compile_status="compiled",
            parser_confidence=0.8,
            semantic_method="llm",
            stage_count=2,
            compiled_plan_count=2,
        ),
    )
    CompiledPlanStore(cfg.compiled_plan_dir).save_sequence(sequence)

    artifact_store = ExecutionArtifactStore(cfg.plan_dir)
    artifact_store.save_plan(plan_1)
    artifact_store.save_report(
        plan_id=plan_1.plan_id,
        slice_id="slice_1",
        turn_id="turn_final",
        payload={"type": "final_report", "summary": "First batch completed.", "verdict": "WATCHLIST"},
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[plan_1],
        current_plan_id=plan_1.plan_id,
        stop_reason="graceful_stop",
    )

    builder = PostRunReportBuilder(config=cfg, planner_adapter=None, run_id="", skip_llm=True)
    result, _run_report = asyncio.run(builder.build(state=state))

    sequence_payload = json.loads(
        (Path(cfg.plan_dir) / "sequence_reports" / "compiled_plan_v1.json").read_text(encoding="utf-8")
    )
    run_payload = json.loads(Path(result["run_report_json"]).read_text(encoding="utf-8"))

    assert sequence_payload["sequence_status"] == "partial"
    assert run_payload["partial_sequences"] == 1


def test_postrun_builder_reflects_failed_then_completed_sequence_when_queue_skips_failures(tmp_path) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        plan_source="compiled_raw",
        compiled_queue_skip_failures=True,
    )
    raw_dir = Path(cfg.raw_plan_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file_1 = raw_dir / "plan_v1.md"
    raw_file_2 = raw_dir / "plan_v2.md"
    raw_file_1.write_text("# plan_v1\n", encoding="utf-8")
    raw_file_2.write_text("# plan_v2\n", encoding="utf-8")

    failed_plan = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v1",
        source_raw_plan=str(raw_file_1),
        slices=[
            PlanSlice(
                slice_id="slice_fail",
                title="Slice fail",
                hypothesis="h",
                objective="o",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="failed",
                last_checkpoint_turn_id="turn_abort",
                last_error="failed early",
            )
        ],
        status="failed",
    )
    completed_plan = ExecutionPlan(
        plan_id="compiled_plan_v2_batch_1",
        goal="Validate route",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=["keep baseline fixed"],
        plan_source_kind="compiled_raw",
        source_sequence_id="compiled_plan_v2",
        source_raw_plan=str(raw_file_2),
        slices=[
            PlanSlice(
                slice_id="slice_ok",
                title="Slice ok",
                hypothesis="h2",
                objective="o2",
                success_criteria=["done"],
                allowed_tools=["events"],
                evidence_requirements=["done"],
                policy_tags=["analysis"],
                max_turns=2,
                max_tool_calls=1,
                max_expensive_calls=0,
                parallel_slot=1,
                status="completed",
                final_report_turn_id="turn_final",
                last_summary="completed",
            )
        ],
        status="completed",
    )
    CompiledPlanStore(cfg.compiled_plan_dir).save_sequence(
        CompiledPlanSequence(
            source_file=str(raw_file_1),
            source_hash="hash_1",
            sequence_id="compiled_plan_v1",
                semantic_plan=SemanticRawPlan(
                    source_file=str(raw_file_1),
                    source_hash="hash_1",
                    source_title="plan_v1",
                    goal="Validate route",
                    global_constraints=["keep baseline fixed"],
                    baseline_ref=failed_plan.baseline_ref,
                    stages=[
                        SemanticStage(
                            stage_id="stage_1",
                            title="Stage 1",
                            objective="Fail route",
                            actions=["Fail route"],
                            success_criteria=["Route failed"],
                        )
                    ],
                ),
            plans=[failed_plan],
            report=CompileReport(
                source_file=str(raw_file_1),
                source_hash="hash_1",
                sequence_id="compiled_plan_v1",
                compile_status="compiled",
                parser_confidence=0.8,
                semantic_method="llm",
                stage_count=1,
                compiled_plan_count=1,
            ),
        )
    )
    CompiledPlanStore(cfg.compiled_plan_dir).save_sequence(
        CompiledPlanSequence(
            source_file=str(raw_file_2),
            source_hash="hash_2",
            sequence_id="compiled_plan_v2",
                semantic_plan=SemanticRawPlan(
                    source_file=str(raw_file_2),
                    source_hash="hash_2",
                    source_title="plan_v2",
                    goal="Validate route",
                    global_constraints=["keep baseline fixed"],
                    baseline_ref=completed_plan.baseline_ref,
                    stages=[
                        SemanticStage(
                            stage_id="stage_1",
                            title="Stage 1",
                            objective="Complete route",
                            actions=["Complete route"],
                            success_criteria=["Route completed"],
                        )
                    ],
                ),
            plans=[completed_plan],
            report=CompileReport(
                source_file=str(raw_file_2),
                source_hash="hash_2",
                sequence_id="compiled_plan_v2",
                compile_status="compiled",
                parser_confidence=0.8,
                semantic_method="llm",
                stage_count=1,
                compiled_plan_count=1,
            ),
        )
    )

    artifact_store = ExecutionArtifactStore(cfg.plan_dir)
    artifact_store.save_plan(failed_plan)
    artifact_store.save_plan(completed_plan)
    artifact_store.save_report(
        plan_id=failed_plan.plan_id,
        slice_id="slice_fail",
        turn_id="turn_abort",
        payload={"type": "abort", "summary": "failed early", "reason_code": "test_failure"},
    )
    artifact_store.save_report(
        plan_id=completed_plan.plan_id,
        slice_id="slice_ok",
        turn_id="turn_final",
        payload={"type": "final_report", "summary": "completed", "verdict": "WATCHLIST"},
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[failed_plan, completed_plan],
        current_plan_id=completed_plan.plan_id,
        stop_reason="goal_reached",
    )

    builder = PostRunReportBuilder(config=cfg, planner_adapter=None, run_id="", skip_llm=True)
    result, _run_report = asyncio.run(builder.build(state=state))
    run_payload = json.loads(Path(result["run_report_json"]).read_text(encoding="utf-8"))
    seq_failed = json.loads((Path(cfg.plan_dir) / "sequence_reports" / "compiled_plan_v1.json").read_text(encoding="utf-8"))
    seq_completed = json.loads((Path(cfg.plan_dir) / "sequence_reports" / "compiled_plan_v2.json").read_text(encoding="utf-8"))

    assert run_payload["failed_sequences"] == 1
    assert run_payload["completed_sequences"] == 1
    assert seq_failed["sequence_status"] == "failed"
    assert seq_completed["sequence_status"] == "completed"
