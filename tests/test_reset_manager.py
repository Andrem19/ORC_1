"""Tests for markdown-plan reset and archival behavior."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.plan_models import Plan, PlanReport
from app.plan_store import PlanStore
from app.reset_manager import ResetManager
from app.state_store import StateStore


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
        "last_change_at": None,
        "stop_reason": None,
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> dict:
    state_dir = tmp_path / "state"
    plans_dir = tmp_path / "plans"

    _write_json(state_dir / "orchestrator_state.json", _make_state_data(cycle=42))
    _write_json(state_dir / "research_context.json", {"snapshots": ["s1"]})

    store = PlanStore(str(plans_dir))
    store.ensure_dirs()
    store.save_plan(Plan(version=1, markdown="# Plan v1\n\n## ETAP 1: Test"))
    store.save_report(PlanReport(plan_version=1, what_was_done="done"))

    return {
        "state_dir": state_dir,
        "plans_dir": plans_dir,
        "state_store": StateStore(state_dir / "orchestrator_state.json"),
        "plan_store": store,
    }


def test_reset_clears_state_and_plans_preserves_research(tmp_workspace: dict) -> None:
    rm = ResetManager(
        state_dir=str(tmp_workspace["state_dir"]),
        state_store=tmp_workspace["state_store"],
        plan_store=tmp_workspace["plan_store"],
    )

    rm.perform_reset("reset")

    assert not (tmp_workspace["state_dir"] / "orchestrator_state.json").exists()
    assert list(tmp_workspace["plans_dir"].glob("plan_v*")) == []
    assert (tmp_workspace["state_dir"] / "research_context.json").exists()


def test_reset_all_clears_everything(tmp_workspace: dict) -> None:
    rm = ResetManager(
        state_dir=str(tmp_workspace["state_dir"]),
        state_store=tmp_workspace["state_store"],
        plan_store=tmp_workspace["plan_store"],
    )

    rm.perform_reset("reset_all")

    assert not (tmp_workspace["state_dir"] / "orchestrator_state.json").exists()
    assert not (tmp_workspace["state_dir"] / "research_context.json").exists()
    assert list(tmp_workspace["plans_dir"].glob("plan_v*")) == []


def test_archive_created_before_clear(tmp_workspace: dict) -> None:
    state_dir: Path = tmp_workspace["state_dir"]
    rm = ResetManager(
        state_dir=str(state_dir),
        state_store=tmp_workspace["state_store"],
        plan_store=tmp_workspace["plan_store"],
    )

    rm.perform_reset("reset")

    archive_base = state_dir / "archive"
    assert archive_base.exists()
    archives = list(archive_base.iterdir())
    assert len(archives) == 1
    archive = archives[0]
    assert (archive / "orchestrator_state.json").exists()
    assert (archive / "plans" / "plan_v1.md").exists()
