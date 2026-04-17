"""Unit tests for null_action_repair: repeated null-action loop detection and repair."""

from __future__ import annotations

import pytest

from app.services.direct_execution.null_action_repair import (
    _is_null_action,
    _count_consecutive_null_action_results,
    _extract_completed_job_ids,
    _extract_saved_run_ids,
    _extract_snapshot_ids,
    repair_null_action_loop,
)


# ---------------------------------------------------------------------------
# _is_null_action
# ---------------------------------------------------------------------------

class TestIsNullAction:
    def test_none_action(self) -> None:
        assert _is_null_action({"action": None}) is True

    def test_missing_action(self) -> None:
        assert _is_null_action({}) is True

    def test_empty_string_action(self) -> None:
        assert _is_null_action({"action": ""}) is True

    def test_whitespace_action(self) -> None:
        assert _is_null_action({"action": "  "}) is True

    def test_concrete_action_is_not_null(self) -> None:
        assert _is_null_action({"action": "list"}) is False

    def test_concrete_result_action(self) -> None:
        assert _is_null_action({"action": "result"}) is False


# ---------------------------------------------------------------------------
# _count_consecutive_null_action_results
# ---------------------------------------------------------------------------

class TestCountConsecutiveNullActionResults:
    def test_empty_transcript(self) -> None:
        assert _count_consecutive_null_action_results([], "backtests_conditions") == 0

    def test_single_null_action(self) -> None:
        transcript = [
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
        ]
        assert _count_consecutive_null_action_results(transcript, "backtests_conditions") == 1

    def test_multiple_consecutive_null_actions(self) -> None:
        transcript = [
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
        ]
        assert _count_consecutive_null_action_results(transcript, "backtests_conditions") == 3

    def test_stops_at_real_action(self) -> None:
        transcript = [
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": "list"}},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
        ]
        assert _count_consecutive_null_action_results(transcript, "backtests_conditions") == 2

    def test_stops_at_different_tool(self) -> None:
        transcript = [
            {"kind": "tool_result", "tool": "research_memory", "original_arguments": {"action": None}},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
        ]
        assert _count_consecutive_null_action_results(transcript, "backtests_conditions") == 1

    def test_skips_non_result_entries(self) -> None:
        transcript = [
            {"kind": "assistant_response", "content": "..."},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
            {"kind": "assistant_response", "content": "..."},
            {"kind": "tool_result", "tool": "backtests_conditions", "original_arguments": {"action": None}},
        ]
        assert _count_consecutive_null_action_results(transcript, "backtests_conditions") == 2


# ---------------------------------------------------------------------------
# _extract_completed_job_ids
# ---------------------------------------------------------------------------

class TestExtractCompletedJobIds:
    def test_extracts_completed_job_ids(self) -> None:
        payload = {
            "data": {
                "jobs": [
                    {"job_id": "cond-abc123", "status": "completed"},
                    {"job_id": "cond-def456", "status": "completed"},
                    {"job_id": "cond-xyz789", "status": "failed"},
                ]
            }
        }
        assert _extract_completed_job_ids(payload) == ["cond-abc123", "cond-def456"]

    def test_no_completed_jobs(self) -> None:
        payload = {
            "data": {
                "jobs": [
                    {"job_id": "cond-abc123", "status": "failed"},
                ]
            }
        }
        assert _extract_completed_job_ids(payload) == []

    def test_empty_jobs_list(self) -> None:
        payload = {"data": {"jobs": []}}
        assert _extract_completed_job_ids(payload) == []

    def test_non_dict_payload(self) -> None:
        assert _extract_completed_job_ids("not a dict") == []

    def test_no_jobs_key(self) -> None:
        assert _extract_completed_job_ids({"data": {}}) == []

    def test_nested_data_key(self) -> None:
        """The MCP server sometimes nests data under data.data."""
        payload = {
            "data": {
                "data": {
                    "jobs": [
                        {"job_id": "cond-nested", "status": "completed"},
                    ]
                }
            }
        }
        assert _extract_completed_job_ids(payload) == ["cond-nested"]


# ---------------------------------------------------------------------------
# _extract_saved_run_ids
# ---------------------------------------------------------------------------

class TestExtractSavedRunIds:
    def test_extracts_run_ids(self) -> None:
        payload = {
            "data": {
                "saved_runs": [
                    {"run_id": "run-abc123"},
                    {"run_id": "run-def456"},
                ]
            }
        }
        assert _extract_saved_run_ids(payload) == ["run-abc123", "run-def456"]

    def test_empty(self) -> None:
        assert _extract_saved_run_ids({"data": {"saved_runs": []}}) == []


# ---------------------------------------------------------------------------
# repair_null_action_loop — main entry point
# ---------------------------------------------------------------------------

class TestRepairNullActionLoop:
    def test_no_repair_when_action_is_set(self) -> None:
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": "list"},
            transcript=[],
        )
        assert result is None
        assert note is None

    def test_no_repair_when_transcript_empty(self) -> None:
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=[],
        )
        assert result is None
        assert note is None

    def test_no_repair_below_threshold(self) -> None:
        """Only 1 prior null-action result — below threshold of 2."""
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {
                    "data": {
                        "jobs": [{"job_id": "cond-abc", "status": "completed"}]
                    }
                },
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is None

    def test_repairs_after_threshold_with_completed_jobs(self) -> None:
        """After 2 consecutive null-action results, repair to action='result'."""
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {
                    "data": {
                        "jobs": [
                            {"job_id": "cond-first", "status": "completed"},
                            {"job_id": "cond-second", "status": "completed"},
                        ]
                    }
                },
            },
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {
                    "data": {
                        "jobs": [
                            {"job_id": "cond-third", "status": "completed"},
                        ]
                    }
                },
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "result"
        assert result["job_id"] == "cond-third"
        assert "null-action loop repair" in note
        assert "consecutive_null_actions=2" in note

    def test_no_repair_when_no_completed_jobs_in_responses(self) -> None:
        """If prior results have no completed jobs, don't repair."""
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": []}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": []}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is None

    def test_unknown_tool_no_repair(self) -> None:
        """Tools without strategies are not repaired."""
        transcript = [
            {"kind": "tool_result", "tool": "unknown_tool", "original_arguments": {"action": None}, "payload": {}},
            {"kind": "tool_result", "tool": "unknown_tool", "original_arguments": {"action": None}, "payload": {}},
        ]
        result, note = repair_null_action_loop(
            tool_name="unknown_tool",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is None

    def test_repairs_backtests_runs_with_saved_runs(self) -> None:
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_runs",
                "original_arguments": {"action": None},
                "payload": {"data": {"saved_runs": [{"run_id": "run-111"}]}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_runs",
                "original_arguments": {"action": None},
                "payload": {"data": {"saved_runs": [{"run_id": "run-222"}]}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_runs",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "detail"
        assert result["run_id"] == "run-222"

    def test_preserves_existing_non_action_args(self) -> None:
        """Non-action arguments should be preserved in the repair."""
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "cond-aaa", "status": "completed"}]}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "cond-bbb", "status": "completed"}]}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None, "project_id": "proj-123", "snapshot_id": "snap-1"},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "result"
        assert result["job_id"] == "cond-bbb"
        assert result["project_id"] == "proj-123"
        assert result["snapshot_id"] == "snap-1"

    def test_backtests_analysis_repair(self) -> None:
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_analysis",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "analysis-111", "status": "completed"}]}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_analysis",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "analysis-222", "status": "completed"}]}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_analysis",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "status"
        assert result["job_id"] == "analysis-222"

    def test_backtests_studies_repair(self) -> None:
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_studies",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "study-111", "status": "succeeded"}]}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_studies",
                "original_arguments": {"action": None},
                "payload": {"data": {"jobs": [{"job_id": "study-222", "status": "completed"}]}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_studies",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "result"
        assert result["job_id"] == "study-222"

    def test_realistic_mcp_response_format(self) -> None:
        """Test with the actual MCP response format from the incident."""
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {
                    "ok": True,
                    "payload": {
                        "content": [{"type": "text", "text": "{...}"}],
                        "structuredContent": {
                            "status": "ok",
                            "data": {
                                "jobs": [
                                    {
                                        "job_id": "cond-38ad1cf60068",
                                        "status": "completed",
                                        "total_conditions": 5,
                                        "completed_conditions": 5,
                                    },
                                    {
                                        "job_id": "cond-6d5b72212700",
                                        "status": "completed",
                                        "total_conditions": 5,
                                        "completed_conditions": 5,
                                    },
                                ]
                            },
                        },
                    },
                },
            },
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": {
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "data": {
                                "jobs": [
                                    {"job_id": "cond-38ad1cf60068", "status": "completed"},
                                ]
                            }
                        }
                    },
                },
            },
        ]
        # The extractor needs to handle nested payload.payload.data or
        # payload.data structures.  Let's verify it works with both formats.
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=transcript,
        )
        # The extractor should find the job_id in the structuredContent
        assert result is not None
        assert result["action"] == "result"
        assert "cond-" in result["job_id"]


# ---------------------------------------------------------------------------
# Integration-style test: simulating the exact scenario from the incident
# ---------------------------------------------------------------------------

class TestIncidentScenario:
    """Simulate the exact compiled_plan_v3_stage_3 failure scenario."""

    def _make_condition_list_response(self, job_ids: list[str]) -> dict:
        """Build a response mimicking the real MCP backtests_conditions list."""
        jobs = [
            {"job_id": jid, "status": "completed", "total_conditions": 5, "completed_conditions": 5}
            for jid in job_ids
        ]
        return {
            "ok": True,
            "payload": {
                "structuredContent": {
                    "status": "ok",
                    "data": {"jobs": jobs, "action": "list"},
                }
            },
        }

    def test_minimax_loop_repair_scenario(self) -> None:
        """Replicate the exact scenario: minimax calls backtests_conditions
        with action=null 3 times, getting rich responses each time."""
        transcript = [
            # First call - null action, server returns list
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": self._make_condition_list_response([
                    "cond-38ad1cf60068",
                    "cond-6d5b72212700",
                    "cond-1b46192b1bd5",
                ]),
            },
            # Second call - null action again
            {
                "kind": "tool_result",
                "tool": "backtests_conditions",
                "original_arguments": {"action": None},
                "payload": self._make_condition_list_response([
                    "cond-38ad1cf60068",
                ]),
            },
        ]

        # The third call with action=null should be auto-repaired
        result, note = repair_null_action_loop(
            tool_name="backtests_conditions",
            arguments={"action": None},
            transcript=transcript,
        )

        assert result is not None, "Should have repaired the null-action loop"
        assert result["action"] == "result", "Should auto-fill action='result'"
        assert result["job_id"] == "cond-38ad1cf60068", (
            "Should pick the first completed job_id from the most recent response"
        )
        assert "consecutive_null_actions=2" in note


# ---------------------------------------------------------------------------
# New: backtests_strategy repair
# ---------------------------------------------------------------------------

class TestBacktestsStrategyRepair:
    def test_repairs_backtests_strategy_with_snapshot_id(self) -> None:
        transcript = [
            {
                "kind": "tool_result",
                "tool": "backtests_strategy",
                "original_arguments": {"action": None},
                "payload": {"data": {"snapshots": [{"snapshot_id": "active-signal-v1"}]}},
            },
            {
                "kind": "tool_result",
                "tool": "backtests_strategy",
                "original_arguments": {"action": None},
                "payload": {"data": {"snapshots": [{"snapshot_id": "active-signal-v1"}]}},
            },
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_strategy",
            arguments={"action": None, "snapshot_id": "active-signal-v1"},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "inspect"
        assert result["snapshot_id"] == "active-signal-v1"
        assert "null-action loop repair" in note

    def test_extract_snapshot_ids_from_list(self) -> None:
        payload = {"data": {"snapshots": [
            {"snapshot_id": "snap-1"},
            {"snapshot_id": "snap-2"},
        ]}}
        assert _extract_snapshot_ids(payload) == ["snap-1", "snap-2"]

    def test_extract_snapshot_ids_top_level(self) -> None:
        payload = {"snapshot_id": "snap-top"}
        assert _extract_snapshot_ids(payload) == ["snap-top"]

    def test_extract_snapshot_ids_empty(self) -> None:
        assert _extract_snapshot_ids({}) == []
        assert _extract_snapshot_ids({"data": {}}) == []


# ---------------------------------------------------------------------------
# New: generic fallback for tools without specific strategies
# ---------------------------------------------------------------------------

class TestGenericFallback:
    def test_generic_fallback_with_project_id(self) -> None:
        """research_memory with project_id should get action='inspect'."""
        transcript = [
            {"kind": "tool_result", "tool": "research_memory", "original_arguments": {"action": None}, "payload": {}},
            {"kind": "tool_result", "tool": "research_memory", "original_arguments": {"action": None}, "payload": {}},
        ]
        result, note = repair_null_action_loop(
            tool_name="research_memory",
            arguments={"action": None, "project_id": "proj-123"},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "inspect"
        assert result["project_id"] == "proj-123"
        assert "generic" in note

    def test_generic_fallback_with_snapshot_id(self) -> None:
        """backtests_plan with snapshot_id should get action='inspect'."""
        transcript = [
            {"kind": "tool_result", "tool": "backtests_plan", "original_arguments": {"action": None}, "payload": {}},
            {"kind": "tool_result", "tool": "backtests_plan", "original_arguments": {"action": None}, "payload": {}},
        ]
        result, note = repair_null_action_loop(
            tool_name="backtests_plan",
            arguments={"action": None, "snapshot_id": "snap-1"},
            transcript=transcript,
        )
        assert result is not None
        assert result["action"] == "inspect"

    def test_no_generic_fallback_without_handle_fields(self) -> None:
        """If no handle fields in arguments, generic fallback can't repair."""
        transcript = [
            {"kind": "tool_result", "tool": "some_tool", "original_arguments": {"action": None}, "payload": {}},
            {"kind": "tool_result", "tool": "some_tool", "original_arguments": {"action": None}, "payload": {}},
        ]
        result, note = repair_null_action_loop(
            tool_name="some_tool",
            arguments={"action": None},
            transcript=transcript,
        )
        assert result is None
