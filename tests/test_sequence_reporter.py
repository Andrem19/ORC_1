"""Tests for sequence_reporter module."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.config import OrchestratorConfig, SequenceReportConfig


def _make_state(plans=None):
    state = MagicMock()
    state.plans = plans or []
    return state


def _make_plan(plan_id="p1", seq_id="plan_v1", status="completed"):
    plan = MagicMock()
    plan.plan_id = plan_id
    plan.source_sequence_id = seq_id
    plan.status = status
    plan.is_terminal = status in {"completed", "failed", "stopped"}
    return plan


def _make_plan_source(is_complete=True):
    source = MagicMock()
    source.is_sequence_complete.return_value = is_complete
    return source


def test_build_and_send_skips_non_compiled_source():
    """Should return early if plan_source is not CompiledPlanSource."""
    from app.services.direct_execution.sequence_reporter import build_and_send_sequence_report
    cfg = OrchestratorConfig()
    not_compiled = MagicMock(spec=[])  # not CompiledPlanSource
    notification = MagicMock()
    plan = _make_plan()
    state = _make_state([plan])
    asyncio.run(build_and_send_sequence_report(
        plan=plan, state=state, config=cfg,
        plan_source=not_compiled, notification_service=notification,
    ))
    notification.send_sequence_complete.assert_not_called()


def test_build_and_send_skips_incomplete_sequence():
    """Should return early if sequence is not yet complete."""
    from app.plan_sources import CompiledPlanSource
    from app.services.direct_execution.sequence_reporter import build_and_send_sequence_report
    cfg = OrchestratorConfig()
    source = _make_plan_source(is_complete=False)
    notification = MagicMock()
    plan = _make_plan()
    state = _make_state([plan])
    # Need isinstance check to pass
    source_class = type("CompiledPlanSource", (), {})
    source.__class__ = type("CompiledPlanSource", (), {})
    asyncio.run(build_and_send_sequence_report(
        plan=plan, state=state, config=cfg,
        plan_source=source, notification_service=notification,
    ))
    notification.send_sequence_complete.assert_not_called()


def test_build_and_send_skips_empty_seq_id():
    """Should return early if sequence_id is empty."""
    from app.services.direct_execution.sequence_reporter import build_and_send_sequence_report
    cfg = OrchestratorConfig()
    source = _make_plan_source(is_complete=True)
    # Make isinstance pass
    from app.plan_sources import CompiledPlanSource
    source.__class__ = type("CompiledPlanSource", (), {})
    notification = MagicMock()
    plan = _make_plan(seq_id="")
    state = _make_state([plan])
    asyncio.run(build_and_send_sequence_report(
        plan=plan, state=state, config=cfg,
        plan_source=source, notification_service=notification,
    ))
    notification.send_sequence_complete.assert_not_called()


def test_build_and_send_handles_collector_exception():
    """Should handle exceptions from ReportCollector gracefully."""
    from app.services.direct_execution.sequence_reporter import build_and_send_sequence_report
    cfg = OrchestratorConfig()
    source = _make_plan_source(is_complete=True)
    from app.plan_sources import CompiledPlanSource
    source.__class__ = type("CompiledPlanSource", (), {})
    notification = MagicMock()
    plan = _make_plan()
    state = _make_state([plan])
    with patch("app.reporting.collector.ReportCollector") as mock_collector:
        mock_collector.return_value.collect_plan_reports.side_effect = RuntimeError("test error")
        asyncio.run(build_and_send_sequence_report(
            plan=plan, state=state, config=cfg,
            plan_source=source, notification_service=notification,
        ))
    notification.send_sequence_complete.assert_not_called()
