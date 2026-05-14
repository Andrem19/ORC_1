from __future__ import annotations

from pathlib import Path

import debuger


class _FakeProc:
    def __init__(self) -> None:
        self.pid = 777
        self.returncode: int | None = None
        self._poll_calls = 0

    def poll(self) -> int | None:
        self._poll_calls += 1
        if self._poll_calls >= 2:
            self.returncode = 0
            return 0
        return None


class _FakeTailer:
    def __init__(self, _path: Path) -> None:
        self._poll_calls = 0

    def seek_to_current_eof(self) -> None:
        return None

    def poll(self):
        self._poll_calls += 1
        if self._poll_calls == 1:
            yield (
                "2026-04-17 01:25:39 [INFO] orchestrator.direct.fallback - "
                "Attempting fallback #1 with provider 'claude_cli' "
                "for slice compiled_plan_v2_stage_5"
            )
            return
        return
        yield

    def close(self) -> None:
        return None


def test_run_cycle_does_not_kill_on_fallback_attempt_only(monkeypatch, tmp_path) -> None:
    proc = _FakeProc()
    kill_called = {"value": False}
    investigate_called = {"value": False}

    monkeypatch.setattr(debuger, "start_orchestrator", lambda: proc)
    monkeypatch.setattr(debuger, "LogTailer", _FakeTailer)
    monkeypatch.setattr(debuger.time, "sleep", lambda _s: None)
    monkeypatch.setattr(debuger, "read_start_from", lambda _p: 2)
    monkeypatch.setattr(debuger, "_manifest_batch_ids", lambda _v: ["compiled_plan_v2_batch_2"])
    monkeypatch.setattr(debuger, "read_completed_batch_ids", lambda: {"compiled_plan_v2_batch_2"})
    monkeypatch.setattr(debuger, "bump_start_from", lambda _p: ("v2", "v3"))

    def _kill(_proc):
        kill_called["value"] = True

    monkeypatch.setattr(debuger, "kill_orchestrator", _kill)

    supervisor = debuger.Supervisor()

    def _investigate(_slice_id, _allowed, *, failure_type):
        investigate_called["value"] = True
        return failure_type

    monkeypatch.setattr(supervisor, "_investigate", _investigate)

    allowed_path = tmp_path / "allowed_fallback.json"
    allowed = debuger.AllowedFallbackStore(allowed_path)

    result = supervisor._run_cycle(allowed)

    assert result == "bumped"
    assert kill_called["value"] is False
    assert investigate_called["value"] is False
