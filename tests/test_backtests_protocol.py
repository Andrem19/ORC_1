"""Tests for backtests protocol prompt generation."""

from __future__ import annotations

from app.services.direct_execution.backtests_protocol import (
    _is_standalone_backtest_slice,
    build_backtests_protocol_lines,
    format_backtests_plan_call,
)


def _baseline_bootstrap() -> dict:
    return {
        "baseline_snapshot_id": "active-signal-v1",
        "baseline_version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }


# ---------- _is_standalone_backtest_slice ----------


def test_detects_standalone_in_title() -> None:
    assert _is_standalone_backtest_slice({"title": "Standalone backtests new signals"})


def test_detects_standalone_in_objective() -> None:
    assert _is_standalone_backtest_slice({"objective": "Run standalone verification"})


def test_detects_standalone_in_success_criteria() -> None:
    assert _is_standalone_backtest_slice({"success_criteria": ["Standalone candidates pass verification"]})


def test_not_standalone_for_candidate_without_standalone() -> None:
    """'candidate' alone should NOT trigger standalone detection."""
    assert not _is_standalone_backtest_slice({"objective": "Test new signal candidates"})


def test_not_standalone_for_integration() -> None:
    assert not _is_standalone_backtest_slice({"title": "Integration analysis"})


def test_not_standalone_for_stability() -> None:
    assert not _is_standalone_backtest_slice({"title": "Stability analysis"})


def test_not_standalone_empty() -> None:
    assert not _is_standalone_backtest_slice({})


# ---------- build_backtests_protocol_lines ----------


def test_standalone_slice_does_not_hardcode_baseline_first_action() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={"title": "Standalone backtests of new signals", "objective": "test"},
        allowed_tools=["backtests_plan", "backtests_runs", "backtests_strategy", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Standalone slices should mention inspecting existing snapshots first
    assert "backtests_strategy(action='inspect'" in text
    # Should NOT say "First live action: call backtests_plan(snapshot_id='active-signal-v1'"
    assert "First live action: call backtests_plan" not in text


def test_non_standalone_slice_uses_baseline_first_action() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={"title": "Stability analysis", "objective": "test stability"},
        allowed_tools=["backtests_plan", "backtests_runs", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Non-standalone should have the hardcoded baseline first action
    assert "First live action: call backtests_plan(snapshot_id='active-signal-v1'" in text


def test_standalone_includes_research_memory_fallback_guidance() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={"title": "Standalone backtests of candidates", "objective": "test"},
        allowed_tools=["backtests_plan", "backtests_runs", "backtests_strategy", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should mention creating research_memory note when stuck
    assert "research_memory(action='create'" in text


def test_standalone_includes_evidence_ref_guidance() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={"title": "Standalone backtests", "objective": "test"},
        allowed_tools=["backtests_plan", "backtests_runs", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should mention not fabricating evidence_refs
    assert "NEVER fabricate refs" in text or "fabricate" in text.lower()
    # Should mention what to do when blocked
    assert "blocked" in text.lower()


def test_non_backtests_context_returns_empty() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={"title": "Setup project", "objective": "setup"},
        allowed_tools=["research_memory", "research_project"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    assert lines == []


def test_format_backtests_plan_call_uses_bootstrap() -> None:
    result = format_backtests_plan_call(baseline_bootstrap=_baseline_bootstrap())
    assert "snapshot_id='active-signal-v1'" in result
    assert "version=1" in result
    assert "symbol='BTCUSDT'" in result
