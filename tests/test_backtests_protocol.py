"""Tests for backtests protocol prompt generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.services.direct_execution.backtests_protocol import (
    _is_standalone_backtest_slice,
    augment_allowed_tools_for_backtests,
    build_backtests_protocol_lines,
    coerce_analysis_start_to_existing_run,
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


# ---------- augment_allowed_tools_for_backtests ----------


def _catalog(has_plan: bool = True, has_runs: bool = True) -> MagicMock:
    catalog = MagicMock()
    catalog.has_tool_name.side_effect = lambda name: {
        "backtests_plan": has_plan,
        "backtests_runs": has_runs,
    }.get(name, False)
    return catalog


def test_augment_adds_plan_when_runs_present() -> None:
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_runs", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="generic_mutation",
        title="Forensic review of A1 backtests",
        objective="review backtest results",
    )
    assert "backtests_plan" in tools


def test_augment_no_change_when_plan_already_present() -> None:
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_runs", "backtests_plan"},
        catalog_snapshot=_catalog(),
        runtime_profile="generic_mutation",
        title="Forensic review of A1 backtests",
        objective="review backtest results",
    )
    # Should be unchanged
    assert tools == {"backtests_runs", "backtests_plan"}


def test_augment_no_change_when_no_runs_and_no_analysis_profile() -> None:
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="research_setup",
        title="Setup project",
        objective="setup",
    )
    assert "backtests_runs" not in tools
    assert "backtests_plan" not in tools


def test_augment_adds_runs_for_stability_analysis_profile() -> None:
    """Regression test: stability analysis slices need backtests_runs to discover saved runs."""
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_analysis", "backtests_conditions", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="backtests_stability_analysis",
        title="Stability и concentration для A1",
        objective="Determine whether A1 survives rigorous window-stability and concentration checks",
        policy_tags=["stability", "gate"],
    )
    assert "backtests_runs" in tools


def test_augment_adds_runs_for_integration_analysis_profile() -> None:
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_analysis", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="backtests_integration_analysis",
        title="Integration analysis for A1",
        objective="Check additive integration of A1",
    )
    assert "backtests_runs" in tools


def test_augment_adds_runs_for_cannibalization_analysis_profile() -> None:
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_analysis", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="backtests_cannibalization_analysis",
        title="Cannibalization analysis",
        objective="Check cannibalization risk",
    )
    assert "backtests_runs" in tools


def test_augment_no_runs_for_non_analysis_without_existing_runs() -> None:
    """Non-analysis profile should not get backtests_runs auto-added."""
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_analysis", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="generic_mutation",
        title="Run a new backtest",
        objective="Start a new standalone backtest",
    )
    assert "backtests_runs" not in tools


def test_augment_no_runs_when_catalog_lacks_tool() -> None:
    """If catalog doesn't have backtests_runs, don't add it."""
    catalog = _catalog(has_runs=False)
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_analysis", "backtests_conditions", "research_memory"},
        catalog_snapshot=catalog,
        runtime_profile="backtests_stability_analysis",
        title="Stability analysis for A1",
        objective="Check stability",
    )
    assert "backtests_runs" not in tools


def test_augment_analysis_profile_with_studies_tool() -> None:
    """backtests_studies should also trigger runs addition."""
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_studies", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="backtests_stability_analysis",
        title="Stability study for A1",
        objective="Run stability studies",
    )
    assert "backtests_runs" in tools


def test_augment_analysis_profile_no_change_when_runs_already_present() -> None:
    """If runs already present, should not duplicate."""
    tools = augment_allowed_tools_for_backtests(
        allowed_tools={"backtests_runs", "backtests_analysis", "research_memory"},
        catalog_snapshot=_catalog(),
        runtime_profile="backtests_stability_analysis",
        title="Stability analysis",
        objective="Check stability",
    )
    assert "backtests_runs" in tools
    assert len([t for t in tools if t == "backtests_runs"]) == 1


def test_stability_analysis_protocol_includes_list_runs_guidance() -> None:
    """When backtests_runs is available but backtests_plan is not, protocol should guide to list runs."""
    lines = build_backtests_protocol_lines(
        slice_payload={
            "title": "Stability analysis for A1",
            "objective": "Check stability",
            "runtime_profile": "backtests_stability_analysis",
        },
        allowed_tools=["backtests_analysis", "backtests_conditions", "backtests_runs", "research_memory"],
        baseline_bootstrap=_baseline_bootstrap(),
    )
    text = "\n".join(lines)
    # Should have guidance about listing saved runs
    assert "list saved runs" in text.lower() or "backtests_runs(action='inspect'" in text
    # Should NOT have the "do not call unavailable readiness tools" since backtests_runs IS available
    assert "do not call unavailable readiness tools" not in text.lower()


# ---------- coerce_analysis_start_to_existing_run ----------


def _saved_run_list_payload(saved_runs: list[dict] | None = None) -> dict:
    return {
        "payload": {
            "structuredContent": {
                "status": "ok",
                "data": {"saved_runs": saved_runs or [], "active_runs": []},
            },
        }
    }


def _start_args(**overrides: object) -> dict:
    args = {
        "action": "start",
        "snapshot_id": "active-signal-v1",
        "version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m",
    }
    args.update(overrides)
    return args


def _transcript_with_saved_run(run_id: str = "run-saved-001") -> list[dict]:
    return [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"view": "list"},
            "payload": _saved_run_list_payload([
                {
                    "run_id": run_id,
                    "status": "completed",
                    "symbol": "BTCUSDT",
                    "strategy_snapshot_id": "active-signal-v1",
                    "strategy_snapshot_version": 1,
                },
            ]),
        },
    ]


def test_coerce_analysis_start_to_conditions() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-001"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_conditions", "backtests_analysis", "backtests_runs"},
    )
    assert result is not None
    new_tool, new_args, note = result
    assert new_tool == "backtests_conditions"
    assert new_args["action"] == "run"
    assert new_args["run_id"] == "run-saved-001"
    assert "Coerced" in note
    assert "run-saved-001" in note


def test_coerce_analysis_start_to_analysis_when_no_conditions() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-002"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_analysis", "backtests_runs"},
    )
    assert result is not None
    new_tool, new_args, note = result
    assert new_tool == "backtests_analysis"
    assert new_args["action"] == "start"
    assert new_args["run_id"] == "run-saved-002"


def test_coerce_returns_none_for_non_analysis_profile() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-001"),
        runtime_profile="generic_mutation",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is None


def test_coerce_returns_none_for_non_start_action() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments={"action": "inspect", "view": "list"},
        transcript=_transcript_with_saved_run("run-saved-001"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is None


def test_coerce_returns_none_for_non_backtests_runs() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_analysis",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-001"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_analysis", "backtests_runs"},
    )
    assert result is None


def test_coerce_returns_none_when_no_saved_runs() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=[],
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is None


def test_coerce_returns_none_when_no_allowed_analysis_tools() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-001"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_runs", "research_memory"},
    )
    assert result is None


def test_coerce_preserves_extra_args_in_conditions() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(project_id="proj-123"),
        transcript=_transcript_with_saved_run("run-saved-003"),
        runtime_profile="backtests_stability_analysis",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is not None
    _, new_args, _ = result
    assert new_args["project_id"] == "proj-123"
    assert new_args["symbol"] == "BTCUSDT"
    assert new_args["anchor_timeframe"] == "1h"


def test_coerce_integration_analysis_profile() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-int"),
        runtime_profile="backtests_integration_analysis",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is not None
    assert result[0] == "backtests_conditions"


def test_coerce_cannibalization_analysis_profile() -> None:
    result = coerce_analysis_start_to_existing_run(
        tool_name="backtests_runs",
        arguments=_start_args(),
        transcript=_transcript_with_saved_run("run-saved-cann"),
        runtime_profile="backtests_cannibalization_analysis",
        allowed_tools={"backtests_conditions", "backtests_runs"},
    )
    assert result is not None
    assert result[0] == "backtests_conditions"
