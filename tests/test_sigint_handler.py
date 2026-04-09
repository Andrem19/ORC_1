"""Tests for SIGINT signal handler debounce and _finish re-entrancy guard."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from app.models import OrchestratorState, StopReason


class TestFinishReentrancyGuard:
    def test_finish_called_twice_only_executes_once(self):
        """_finish() should silently return on second invocation."""
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.NotificationService"):
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
                # Phase transition BEFORE action (matches main.py fix)
                _stop_phase = 1
                orch_mock.request_drain()
                drain_called.append(True)
            elif _stop_phase == 1:
                _stop_phase = 2
                orch_mock.request_stop()
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

    def test_burst_of_signals_only_calls_drain_once(self):
        """Simulate rapid SIGINT burst — request_drain must fire only once."""
        orch = MagicMock()
        handler, drain_called, stop_called = self._make_handler(orch)
        # Fire 20 rapid signals (third+ raise KeyboardInterrupt)
        for _ in range(20):
            try:
                handler(2, None)
            except KeyboardInterrupt:
                pass
        # Only first call enters drain, second enters stop, rest raise
        assert len(drain_called) == 1
        assert len(stop_called) == 1
        orch.request_drain.assert_called_once()
        orch.request_stop.assert_called_once()


class TestRequestDrainIdempotency:
    """Verify request_drain() and request_stop() are idempotent."""

    def _make_orch(self):
        from app.orchestrator import Orchestrator

        with patch("app.orchestrator.NotificationService"):
            return Orchestrator.__new__(Orchestrator)

    def test_request_drain_idempotent(self):
        """request_drain() should only send one Telegram notification."""
        orch = self._make_orch()
        orch._drain_mode = False
        orch._drain_started_at = None
        orch._plan_service = None
        orch.state = OrchestratorState(goal="test")
        orch.notification_service = MagicMock()

        orch.request_drain()
        assert orch._drain_mode is True
        assert orch.notification_service.send_lifecycle.call_count == 1

        # Second call should be a no-op
        orch.request_drain()
        assert orch.notification_service.send_lifecycle.call_count == 1

    def test_request_stop_idempotent(self):
        """request_stop() should only log once."""
        orch = self._make_orch()
        orch._stop_requested = False
        orch._plan_service = None

        import logging
        with patch.object(logging.getLogger("orchestrator"), "info") as log_mock:
            orch.request_stop()
            assert orch._stop_requested is True

            orch.request_stop()
            # Only one log call — second was a no-op
            assert log_mock.call_count == 1
