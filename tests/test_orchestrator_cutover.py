from __future__ import annotations

import json

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.orchestrator import Orchestrator


def test_orchestrator_load_state_archives_legacy_runtime(tmp_path) -> None:
    state_dir = tmp_path / "state"
    plans_dir = tmp_path / "plans"
    state_dir.mkdir()
    plans_dir.mkdir()
    (state_dir / "orchestrator_state.json").write_text(json.dumps({"legacy": True}), encoding="utf-8")
    (plans_dir / "plan_v1.md").write_text("# old plan", encoding="utf-8")
    cfg = OrchestratorConfig(goal="test", state_dir=str(state_dir), plan_dir=str(plans_dir))
    orch = Orchestrator(
        config=cfg,
        planner_adapter=FakePlanner(responses=[]),
        worker_adapter=FakeWorker(responses=[]),
    )

    loaded = orch.load_state()

    assert loaded is False
    archive_root = state_dir / "archive" / "broker_cutover"
    archives = list(archive_root.iterdir())
    assert archives
    assert (archives[0] / "orchestrator_state.json").exists()
    assert (archives[0] / "plans" / "plan_v1.md").exists()
