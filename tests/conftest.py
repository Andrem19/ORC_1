"""Shared pytest fixtures and hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.pytest_timing import PytestTimingRecorder

_TIMING_RECORDER = PytestTimingRecorder(report_path=Path("artifacts") / "test_reports" / "pytest_durations.json")


def pytest_sessionstart(session) -> None:
    """Start wall-clock timing for the full test session."""

    del session
    _TIMING_RECORDER.start_session()


def pytest_runtest_logreport(report) -> None:
    """Accumulate setup/call/teardown timings for each test item."""

    if report.when in {"setup", "call", "teardown"}:
        _TIMING_RECORDER.record_phase(report.nodeid, report.when, report.duration)


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:
    """Print the overall test timing summary and the slowest tests."""

    del exitstatus, config
    report_path = _TIMING_RECORDER.write_report()
    total_seconds = _TIMING_RECORDER.session_total_seconds()
    top_records = _TIMING_RECORDER.top_records()

    terminalreporter.section("test timing summary", sep="=")
    terminalreporter.write_line(f"session_total_seconds: {total_seconds:.3f}")
    terminalreporter.write_line(f"full_report: {report_path}")
    if not top_records:
        terminalreporter.write_line("top_slowest: no collected test timings")
        return
    terminalreporter.write_line("top_slowest:")
    for index, record in enumerate(top_records, start=1):
        terminalreporter.write_line(
            f"{index:02d}. {record.total_seconds:8.3f}s total "
            f"(setup {record.setup_seconds:6.3f}s, call {record.call_seconds:6.3f}s, teardown {record.teardown_seconds:6.3f}s) "
            f"{record.nodeid}"
        )
