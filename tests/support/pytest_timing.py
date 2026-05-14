"""Pytest timing helpers and plugin state."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time


@dataclass(frozen=True, slots=True)
class TestTimingRecord:
    """One fully aggregated pytest test duration."""

    nodeid: str
    total_seconds: float
    setup_seconds: float
    call_seconds: float
    teardown_seconds: float


def aggregate_phase_timings(phase_timings: dict[str, dict[str, float]]) -> list[TestTimingRecord]:
    """Collapse per-phase durations into sorted per-test records."""

    records: list[TestTimingRecord] = []
    for nodeid, phases in phase_timings.items():
        setup_seconds = float(phases.get("setup", 0.0))
        call_seconds = float(phases.get("call", 0.0))
        teardown_seconds = float(phases.get("teardown", 0.0))
        records.append(
            TestTimingRecord(
                nodeid=nodeid,
                total_seconds=setup_seconds + call_seconds + teardown_seconds,
                setup_seconds=setup_seconds,
                call_seconds=call_seconds,
                teardown_seconds=teardown_seconds,
            )
        )
    return sorted(records, key=lambda item: (-item.total_seconds, item.nodeid))


class PytestTimingRecorder:
    """Collect per-phase test durations and persist sorted reports."""

    def __init__(self, *, report_path: Path, top_limit: int = 20) -> None:
        self._report_path = report_path
        self._top_limit = top_limit
        self._session_started_at: float | None = None
        self._phase_timings: dict[str, dict[str, float]] = {}

    def start_session(self) -> None:
        self._session_started_at = time.perf_counter()

    def record_phase(self, nodeid: str, phase: str, duration_seconds: float) -> None:
        phases = self._phase_timings.setdefault(nodeid, {})
        phases[phase] = phases.get(phase, 0.0) + float(duration_seconds)

    def build_records(self) -> list[TestTimingRecord]:
        return aggregate_phase_timings(self._phase_timings)

    def session_total_seconds(self) -> float:
        if self._session_started_at is None:
            return 0.0
        return max(time.perf_counter() - self._session_started_at, 0.0)

    def top_records(self) -> list[TestTimingRecord]:
        return self.build_records()[: self._top_limit]

    def write_report(self) -> Path:
        self._report_path.parent.mkdir(parents=True, exist_ok=True)
        records = self.build_records()
        payload = {
            "session_total_seconds": self.session_total_seconds(),
            "test_count": len(records),
            "top_limit": self._top_limit,
            "top_slowest": [_serialize_record(item) for item in records[: self._top_limit]],
            "all_tests": [_serialize_record(item) for item in records],
        }
        self._report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return self._report_path


def _serialize_record(record: TestTimingRecord) -> dict[str, float | str]:
    return {
        "nodeid": record.nodeid,
        "total_seconds": round(record.total_seconds, 6),
        "setup_seconds": round(record.setup_seconds, 6),
        "call_seconds": round(record.call_seconds, 6),
        "teardown_seconds": round(record.teardown_seconds, 6),
    }
