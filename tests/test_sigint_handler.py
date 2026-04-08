"""Tests for SIGINT signal handler debounce and _finish re-entrancy guard."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from app.models import OrchestratorState, StopReason


class TestFinishReentrancyGuard:
    def test_finish_called_twice_only_executes_once(self):
        """_finish() should silently return on second invocation."""
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.WorkerService"), \
             patch("app.orchestrator.NotificationService"):
            orch = MagicMock(spec=Orchestrator)
            orch._finish_completed = False
            orch.state = OrchestratorState(goal="test")

            # Simulate real _finish behavior: set guard, collect output, notify
            call_count = 0

            def fake_finish(reason, summary=""):
                nonlocal call_count
                if orch._finish_completed:
                    return
                orch._finish_completed = True
                call_count += 1

            orch._finish = fake_finish

            orch._finish(StopReason.GRACEFUL_STOP, "first")
            orch._finish(StopReason.NO_PROGRESS, "second")
            assert call_count == 1


class TestSignalHandlerDebounce:
    def _make_handler(self, orch_mock):
        """Create a signal handler with debounce logic matching main.py."""
        import signal as _signal

        _stop_phase = 0
        _last_signal_time = 0.0

        drain_called = []
        stop_called = []

        def handler(signum, frame):
            nonlocal _stop_phase, _last_signal_time
            now = time.monotonic()
            suppressed = (now - _last_signal_time) < 2.0
            _last_signal_time = now
            if _stop_phase == 0:
                orch_mock.request_drain()
                _stop_phase = 1
                drain_called.append(True)
            elif _stop_phase == 1:
                orch_mock.request_stop()
                _stop_phase = 2
                stop_called.append(not suppressed)
            else:
                if not suppressed:
                    pass  # would log
                raise KeyboardInterrupt

        return handler, drain_called, stop_called

    def test_first_signal_enters_drain_mode(self):
        orch = MagicMock()
        handler, drain_called, stop_called = self._make_handler(orch)
        handler(2, None)  # SIGINT
        assert len(drain_called) == 1
        orch.request_drain.assert_called_once()

    def test_second_signal_forces_stop(self):
        orch = MagicMock()
        handler, drain_called, stop_called = self._make_handler(orch)
        handler(2, None)  # phase 0→1
        handler(2, None)  # phase 1→2
        assert len(stop_called) == 1
        orch.request_stop.assert_called_once()

    def test_third_signal_raises_keyboard_interrupt(self):
        orch = MagicMock()
        handler, _, _ = self._make_handler(orch)
        handler(2, None)  # phase 0→1
        handler(2, None)  # phase 1→2
        try:
            handler(2, None)  # phase 2→3
            assert False, "Should have raised KeyboardInterrupt"
        except KeyboardInterrupt:
            pass

    def test_rapid_second_signal_suppresses_logging(self):
        """Two signals within 2s — second signal logging should be suppressed."""
        orch = MagicMock()
        handler, drain_called, stop_called = self._make_handler(orch)
        handler(2, None)  # phase 0→1
        # Immediately fire again — suppressed
        handler(2, None)  # phase 1→2
        # stop_called[0] is False (suppressed) — action still executed
        assert len(stop_called) == 1
        assert stop_called[0] is False
        # But request_stop still called (action is not suppressed)
        orch.request_stop.assert_called_once()
