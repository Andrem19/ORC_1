from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import reporter
from app.config import OrchestratorConfig
from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2
from app.execution_store import ExecutionStateStore


def test_reporter_run_rebuilds_reports_from_existing_artifacts(tmp_path, monkeypatch, capsys) -> None:
    cfg = OrchestratorConfig(
        goal="Test reporting",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        log_dir=str(tmp_path / "logs"),
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
        current_run_id="run_test",
        plan_source="planner",
    )
    state = ExecutionStateV2(
        goal="Test reporting",
        status="finished",
        plans=[
            ExecutionPlan(
                plan_id="plan_1",
                goal="Validate route",
                baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
                global_constraints=[],
                slices=[],
                status="completed",
            )
        ],
        current_plan_id="plan_1",
        stop_reason="goal_reached",
    )
    ExecutionStateStore(cfg.execution_state_path, run_id=cfg.current_run_id).save(state)
    monkeypatch.setattr(reporter, "_load_config", lambda: cfg)

    exit_code = asyncio.run(
        reporter._run(
            argparse.Namespace(
                run_id=cfg.current_run_id,
                skip_llm=True,
                json_only=False,
                rebuild_all=False,
            )
        )
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Reports rebuilt:" in captured.out
    assert (Path(cfg.plan_dir) / "runs" / cfg.current_run_id / "run_report.json").exists()
    assert (Path(cfg.plan_dir) / "runs" / cfg.current_run_id / "run_report.md").exists()
