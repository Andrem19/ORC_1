"""Tests for the reset manager and related store methods."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.plan_models import ResearchPlan, PlanTask, TaskReport
from app.plan_store import PlanStore
from app.reset_manager import ResetManager
from app.state_store import StateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_state_data(cycle: int = 5) -> dict:
    return {
        "goal": "test",
        "status": "running",
        "current_cycle": cycle,
        "empty_cycles": 0,
        "total_errors": 0,
        "tasks": [],
        "results": [],
        "processes": [],
        "memory": [],
        "last_planner_decision": None,
        "last_planner_call_at": None,
        "last_change_at": None,
        "stop_reason": None,
        "current_plan_version": 0,
        "plan_task_dispatch_map": {},
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


def _make_plan(version: int = 1) -> ResearchPlan:
    return ResearchPlan(
        plan_id="abc123",
        version=version,
        created_at="2026-01-01T00:00:00Z",
        goal="test",
        tasks=[
            PlanTask(
                task_id="t1",
                plan_version=version,
                stage_number=1,
                stage_name="step1",
                theory="",
                agent_instructions=[],
                results_table_columns=[],
                results_table_rows=[],
                decision_gates=[],
                created_at="2026-01-01T00:00:00Z",
            )
        ],
        execution_order=[1],
    )


def _make_report(task_id: str = "t1", plan_version: int = 1) -> TaskReport:
    return TaskReport(
        task_id=task_id,
        worker_id="w1",
        plan_version=plan_version,
        status="success",
        what_was_done="did stuff",
        timestamp="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_workspace(tmp_path: Path) -> dict:
    """Create a temp workspace with state, plans, and research_context."""
    state_dir = tmp_path / "state"
    plans_dir = tmp_path / "plans"

    # State file
    _write_json(state_dir / "orchestrator_state.json", _make_state_data(cycle=42))

    # Research context
    _write_json(state_dir / "research_context.json", {"snapshots": ["s1"]})

    # Plan files
    plan_store = PlanStore(str(plans_dir))
    plan_store.ensure_dirs()
    plan_store.save_plan(_make_plan(version=1))
    plan_store.save_plan(_make_plan(version=2))
    plan_store.save_report(_make_report("t1", 1))
    plan_store.save_report(_make_report("t2", 2))

    return {
        "state_dir": state_dir,
        "plans_dir": plans_dir,
        "state_store": StateStore(state_dir / "orchestrator_state.json"),
        "plan_store": plan_store,
    }


# ---------------------------------------------------------------------------
# PlanStore.clear_all / archive_to
# ---------------------------------------------------------------------------

class TestPlanStoreClearAll:
    def test_clears_plans_and_reports(self, tmp_workspace: dict) -> None:
        ps: PlanStore = tmp_workspace["plan_store"]
        plans_dir: Path = tmp_workspace["plans_dir"]

        assert len(list(plans_dir.glob("plan_v*"))) > 0
        assert len(list((plans_dir / "reports").glob("*.json"))) > 0

        ps.clear_all()

        assert list(plans_dir.glob("plan_v*")) == []
        assert list((plans_dir / "reports").glob("*.json")) == []

    def test_noop_on_empty_dir(self, tmp_path: Path) -> None:
        ps = PlanStore(str(tmp_path / "plans"))
        ps.clear_all()  # should not raise


class TestPlanStoreArchiveTo:
    def test_copies_files(self, tmp_workspace: dict, tmp_path: Path) -> None:
        ps: PlanStore = tmp_workspace["plan_store"]
        archive = tmp_path / "archive"

        count = ps.archive_to(archive)

        assert count > 0
        assert (archive / "plan_v1.json").exists()
        assert (archive / "plan_v2.json").exists()
        assert (archive / "reports" / "t1.json").exists()


# ---------------------------------------------------------------------------
# StateStore.archive_to
# ---------------------------------------------------------------------------

class TestStateStoreArchiveTo:
    def test_copies_state_file(self, tmp_workspace: dict, tmp_path: Path) -> None:
        ss: StateStore = tmp_workspace["state_store"]
        archive = tmp_path / "archive"

        result = ss.archive_to(archive)

        assert result is True
        assert (archive / "orchestrator_state.json").exists()

    def test_returns_false_when_no_file(self, tmp_path: Path) -> None:
        ss = StateStore(tmp_path / "nonexistent.json")
        result = ss.archive_to(tmp_path / "archive")
        assert result is False


# ---------------------------------------------------------------------------
# ResetManager
# ---------------------------------------------------------------------------

class TestResetManager:
    def test_reset_clears_state_and_plans_preserves_research(self, tmp_workspace: dict, tmp_path: Path) -> None:
        rm = ResetManager(
            state_dir=str(tmp_workspace["state_dir"]),
            state_store=tmp_workspace["state_store"],
            plan_store=tmp_workspace["plan_store"],
        )

        rm.perform_reset("reset")

        # State cleared
        assert not (tmp_workspace["state_dir"] / "orchestrator_state.json").exists()
        # Plans cleared
        plans_dir: Path = tmp_workspace["plans_dir"]
        assert list(plans_dir.glob("plan_v*")) == []
        assert list((plans_dir / "reports").glob("*.json")) == []
        # Research context preserved
        assert (tmp_workspace["state_dir"] / "research_context.json").exists()

    def test_reset_all_clears_everything(self, tmp_workspace: dict) -> None:
        rm = ResetManager(
            state_dir=str(tmp_workspace["state_dir"]),
            state_store=tmp_workspace["state_store"],
            plan_store=tmp_workspace["plan_store"],
        )

        rm.perform_reset("reset_all")

        assert not (tmp_workspace["state_dir"] / "orchestrator_state.json").exists()
        assert not (tmp_workspace["state_dir"] / "research_context.json").exists()
        plans_dir: Path = tmp_workspace["plans_dir"]
        assert list(plans_dir.glob("plan_v*")) == []

    def test_archive_created_before_clear(self, tmp_workspace: dict) -> None:
        state_dir: Path = tmp_workspace["state_dir"]
        rm = ResetManager(
            state_dir=str(state_dir),
            state_store=tmp_workspace["state_store"],
            plan_store=tmp_workspace["plan_store"],
        )

        rm.perform_reset("reset")

        # Archive should exist with files
        archive_base = state_dir / "archive"
        assert archive_base.exists()
        archives = list(archive_base.iterdir())
        assert len(archives) == 1
        archived = archives[0]
        assert (archived / "orchestrator_state.json").exists()
        assert (archived / "research_context.json").exists()

    def test_reset_with_no_existing_state(self, tmp_path: Path) -> None:
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        plans_dir = tmp_path / "plans"

        rm = ResetManager(
            state_dir=str(state_dir),
            state_store=StateStore(state_dir / "orchestrator_state.json"),
            plan_store=PlanStore(str(plans_dir)),
        )

        # Should not raise
        rm.perform_reset("reset")

    def test_invalid_mode_raises(self, tmp_workspace: dict) -> None:
        rm = ResetManager(
            state_dir=str(tmp_workspace["state_dir"]),
            state_store=tmp_workspace["state_store"],
            plan_store=tmp_workspace["plan_store"],
        )

        with pytest.raises(ValueError, match="Invalid reset mode"):
            rm.perform_reset("bad_mode")

    def test_reset_without_plan_store(self, tmp_workspace: dict) -> None:
        """Reset works even when plan_store is None (plan_mode disabled)."""
        rm = ResetManager(
            state_dir=str(tmp_workspace["state_dir"]),
            state_store=tmp_workspace["state_store"],
            plan_store=None,
        )

        rm.perform_reset("reset")
        assert not (tmp_workspace["state_dir"] / "orchestrator_state.json").exists()
        # Plans should NOT be cleared since plan_store is None
        plans_dir: Path = tmp_workspace["plans_dir"]
        assert len(list(plans_dir.glob("plan_v*"))) > 0
