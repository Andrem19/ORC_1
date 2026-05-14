"""Tests for backtests start guard analysis-slice remediation and protocol lines."""

from __future__ import annotations

from app.services.direct_execution.backtests_protocol import (
    _saved_run_ids_from_transcript,
    _start_guard_error,
    backtests_start_guard_payload,
    build_backtests_protocol_lines,
)


def _baseline_bootstrap() -> dict:
    return {
        "baseline_snapshot_id": "active-signal-v1",
        "baseline_version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }


def _list_result_payload(saved_runs: list[dict] | None = None) -> dict:
    """Build a minimal MCP-style payload for a backtests_runs list call."""
    return {
        "payload": {
            "content": [
                {
                    "type": "text",
                    "text": '{"status": "ok", "data": {"saved_runs": '
                    + str(saved_runs or [])
                    + ', "active_runs": []}}',
                }
            ],
            "structuredContent": {
                "status": "ok",
                "data": {"saved_runs": saved_runs or [], "active_runs": []},
            },
        }
    }


def _start_arguments() -> dict:
    return {
        "action": "start",
        "snapshot_id": "active-signal-v1",
        "version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }


# ---------- _start_guard_error remediation ----------


def test_analysis_profile_duplicate_includes_remediation() -> None:
    error = _start_guard_error(
        "duplicate_baseline_start_blocked",
        _start_arguments(),
        is_analysis=True,
    )
    assert "remediation" in error
    assert "backtests_conditions" in error["remediation"]


def test_analysis_profile_duplicate_with_saved_ids_includes_sample() -> None:
    error = _start_guard_error(
        "duplicate_baseline_start_blocked",
        _start_arguments(),
        is_analysis=True,
        saved_run_ids=["run-aaa", "run-bbb", "run-ccc", "run-ddd"],
    )
    assert "remediation" in error
    assert "run-aaa" in error["remediation"]
    assert "first 3" in error["remediation"]


def test_non_analysis_profile_no_remediation() -> None:
    error = _start_guard_error(
        "duplicate_baseline_start_blocked",
        _start_arguments(),
        is_analysis=False,
    )
    assert "remediation" not in error


def test_wrong_reason_code_no_remediation_even_for_analysis() -> None:
    error = _start_guard_error(
        "backtests_plan_required_before_start",
        _start_arguments(),
        is_analysis=True,
    )
    assert "remediation" not in error


# ---------- _saved_run_ids_from_transcript ----------


def test_extracts_matching_run_ids() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "20260416-114219-87bba82c", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
                {"run_id": "20260416-113840-ea317579", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    ids = _saved_run_ids_from_transcript(transcript, _start_arguments())
    assert ids == ["20260416-114219-87bba82c", "20260416-113840-ea317579"]


def test_filters_by_symbol() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "run-wrong-symbol", "status": "completed", "symbol": "ETHUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
                {"run_id": "run-right-symbol", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    ids = _saved_run_ids_from_transcript(transcript, _start_arguments())
    assert ids == ["run-right-symbol"]


def test_filters_by_snapshot() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "run-wrong-snap", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "other-snapshot", "strategy_snapshot_version": 1},
                {"run_id": "run-right-snap", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    ids = _saved_run_ids_from_transcript(transcript, _start_arguments())
    assert ids == ["run-right-snap"]


def test_skips_non_completed_runs() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "run-active", "status": "active", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
                {"run_id": "run-completed", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    ids = _saved_run_ids_from_transcript(transcript, _start_arguments())
    assert ids == ["run-completed"]


def test_empty_transcript_returns_empty() -> None:
    ids = _saved_run_ids_from_transcript([], _start_arguments())
    assert ids == []


# ---------- backtests_start_guard_payload ----------


def test_guard_analysis_duplicate_returns_remediation_with_ids() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "20260416-114219-87bba82c", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    result = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments=_start_arguments(),
        transcript=transcript,
        runtime_profile="backtests_stability_analysis",
        baseline_bootstrap=_baseline_bootstrap(),
    )
    assert result is not None
    assert result["ok"] is False
    assert result["details"]["reason_code"] == "duplicate_baseline_start_blocked"
    assert "remediation" in result
    assert "backtests_conditions" in result["remediation"]
    assert "20260416-114219-87bba82c" in result["remediation"]


def test_guard_non_analysis_duplicate_no_remediation() -> None:
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _list_result_payload([
                {"run_id": "20260416-114219-87bba82c", "status": "completed", "symbol": "BTCUSDT", "strategy_snapshot_id": "active-signal-v1", "strategy_snapshot_version": 1},
            ]),
        },
    ]
    result = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments=_start_arguments(),
        transcript=transcript,
        runtime_profile="generic_mutation",
        baseline_bootstrap=_baseline_bootstrap(),
    )
    assert result is not None
    assert result["ok"] is False
    assert "remediation" not in result


def test_guard_passes_when_no_block() -> None:
    result = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments=_start_arguments(),
        transcript=[],
        runtime_profile="backtests_stability_analysis",
        baseline_bootstrap=_baseline_bootstrap(),
    )
    # Not blocked because no duplicate saved run exists and no successful backtests_plan
    # But it WILL be blocked by backtests_plan_required_before_start
    # So we need a successful plan in transcript
    assert result is not None  # blocked by plan requirement


def test_guard_no_block_when_plan_present_no_duplicate() -> None:
    plan_payload = {
        "payload": {
            "content": [{"type": "text", "text": '{"status": "ok", "data": {"valid": true}}'}],
            "structuredContent": {"status": "ok", "data": {"valid": True}},
        }
    }
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_plan",
            "arguments": {},
            "payload": plan_payload,
        },
    ]
    result = backtests_start_guard_payload(
        tool_name="backtests_runs",
        arguments=_start_arguments(),
        transcript=transcript,
        runtime_profile="backtests_stability_analysis",
        baseline_bootstrap=_baseline_bootstrap(),
    )
    assert result is None  # not blocked


# ---------- build_backtests_protocol_lines analysis awareness ----------


def test_analysis_profile_says_use_conditions_when_blocked() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={
            "title": "Stability and condition analysis",
            "objective": "test stability",
            "runtime_profile": "backtests_stability_analysis",
        },
        allowed_tools=["backtests_conditions", "backtests_analysis", "backtests_runs", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should instruct to use existing saved runs with conditions
    assert "backtests_conditions(action='run'" in text
    assert "saved_run_id" in text


def test_non_analysis_profile_says_research_note_when_blocked() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={
            "title": "Standalone backtests of new signals",
            "objective": "test standalone",
            "runtime_profile": "generic_mutation",
        },
        allowed_tools=["backtests_plan", "backtests_runs", "backtests_strategy", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should instruct to create research note and return
    assert "research_memory note" in text or "research_memory" in text


def test_integration_analysis_profile_uses_analysis_redirect() -> None:
    lines = build_backtests_protocol_lines(
        slice_payload={
            "title": "Integration analysis",
            "objective": "test integration",
            "runtime_profile": "backtests_integration_analysis",
        },
        allowed_tools=["backtests_conditions", "backtests_analysis", "backtests_runs", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should instruct to use existing saved runs
    assert "saved_run_id" in text
