from __future__ import annotations

import pytest

from app.execution_parsing import StructuredOutputError, parse_execution_plan_output, parse_worker_action_output


def test_parse_execution_plan_output_accepts_valid_json() -> None:
    raw = """
    {
      "plan_id": "plan_1",
      "goal": "Validate funding signal",
      "baseline_ref": {
        "snapshot_id": "active-signal-v1",
        "version": 1,
        "symbol": "BTCUSDT",
        "anchor_timeframe": "1h",
        "execution_timeframe": "5m"
      },
      "global_constraints": ["keep baseline fixed"],
      "slices": [
        {
          "slice_id": "slice_1",
          "title": "Funding data check",
          "hypothesis": "funding can add orthogonal information",
          "objective": "verify cheap evidence first",
          "success_criteria": ["catalog exists"],
          "allowed_tools": ["events", "events_sync"],
          "evidence_requirements": ["fresh rows"],
          "policy_tags": ["cheap_first"],
          "max_turns": 4,
          "max_tool_calls": 2,
          "max_expensive_calls": 1,
          "parallel_slot": 1
        }
      ]
    }
    """

    plan = parse_execution_plan_output(raw)

    assert plan.plan_id == "plan_1"
    assert plan.baseline_ref.snapshot_id == "active-signal-v1"
    assert plan.slices[0].allowed_tools == ["events", "events_sync"]


def test_parse_execution_plan_output_rejects_missing_slices() -> None:
    raw = """
    {
      "plan_id": "plan_1",
      "goal": "Bad plan",
      "baseline_ref": {"snapshot_id": "active-signal-v1", "version": 1},
      "global_constraints": []
    }
    """

    with pytest.raises(StructuredOutputError, match="execution_plan_missing_fields"):
        parse_execution_plan_output(raw)


def test_parse_worker_action_output_rejects_tool_outside_allowlist() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "system_reset_space",
      "arguments": {"action": "preview"},
      "reason": "reset everything"
    }
    """

    with pytest.raises(StructuredOutputError, match="tool_not_in_allowlist"):
        parse_worker_action_output(raw, allowlist={"events", "events_sync"})


def test_parse_worker_action_output_rejects_prefixed_mcp_tool_name() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "mcp__dev_space1__features_catalog",
      "arguments": {"scope": "available"},
      "reason": "inspect catalog"
    }
    """

    with pytest.raises(StructuredOutputError, match="tool_prefixed_namespace_forbidden"):
        parse_worker_action_output(raw, allowlist={"features_catalog"})


def test_parse_worker_action_output_accepts_final_report() -> None:
    raw = """
    {
      "type": "final_report",
      "summary": "Funding data is ready for further feature work.",
      "verdict": "WATCHLIST",
      "facts": {"funding_rows": 7147},
      "artifacts": ["job_123"],
      "key_metrics": {"funding_rows": 7147},
      "confidence": 0.7
    }
    """

    action = parse_worker_action_output(raw, allowlist={"events"})

    assert action.action_type == "final_report"
    assert action.verdict == "WATCHLIST"
    assert action.facts["funding_rows"] == 7147


def test_parse_worker_action_output_accepts_optional_reporting_fields() -> None:
    raw = """
    {
      "type": "final_report",
      "summary": "Funding route looks promising.",
      "verdict": "WATCHLIST",
      "findings": ["Funding rows confirmed"],
      "rejected_findings": ["No liquidation edge found"],
      "next_actions": ["Run orthogonality backtest"],
      "risks": ["Data freshness could drift"],
      "evidence_refs": ["artifact_1"],
      "key_metrics": {"funding_rows": 7147},
      "confidence": 0.75
    }
    """

    action = parse_worker_action_output(raw, allowlist={"events"})

    assert action.findings == ["Funding rows confirmed"]
    assert action.rejected_findings == ["No liquidation edge found"]
    assert action.next_actions == ["Run orthogonality backtest"]
    assert action.risks == ["Data freshness could drift"]
    assert action.evidence_refs == ["artifact_1"]


def test_parse_worker_action_output_uses_reason_as_summary_for_final_report_when_summary_missing() -> None:
    raw = """
    {
      "type": "final_report",
      "reason": "Slice objective satisfied after baseline setup and atlas initialization.",
      "verdict": "pass",
      "facts": {"project_id": "proj_123"}
    }
    """

    action = parse_worker_action_output(raw, allowlist={"research_record"})

    assert action.action_type == "final_report"
    assert action.summary == "Slice objective satisfied after baseline setup and atlas initialization."
    assert action.verdict == "pass"


def test_parse_execution_plan_output_accepts_single_item_array_wrapper() -> None:
    raw = """
    [
      {
        "plan_id": "plan_1",
        "goal": "Validate funding signal",
        "baseline_ref": {"snapshot_id": "active-signal-v1", "version": 1},
        "global_constraints": ["keep baseline fixed"],
        "slices": [
          {
            "slice_id": "slice_1",
            "title": "Funding data check",
            "hypothesis": "funding can add orthogonal information",
            "objective": "verify cheap evidence first",
            "success_criteria": ["catalog exists"],
            "allowed_tools": ["events"],
            "evidence_requirements": ["fresh rows"],
            "policy_tags": ["cheap_first"],
            "max_turns": 4,
            "max_tool_calls": 2,
            "max_expensive_calls": 1,
            "parallel_slot": 1
          }
        ]
      }
    ]
    """

    plan = parse_execution_plan_output(raw)

    assert plan.plan_id == "plan_1"


def test_parse_worker_action_output_flattens_nested_tool_arguments_wrapper() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "events_sync",
      "arguments": {
        "tool": "events_sync",
        "arguments": {
          "family": "expiry",
          "scope": "incremental",
          "wait": "started"
        }
      },
      "reason": "sync expiry events"
    }
    """

    action = parse_worker_action_output(raw, allowlist={"events_sync"})

    assert action.arguments == {"family": "expiry", "scope": "incremental", "wait": "started"}


def test_parse_worker_action_output_recovers_inline_tool_call_from_free_form_output() -> None:
    raw = """
    I need to inspect the catalog first.

    **Tool call**: `features_catalog(scope="all")`

    Expected evidence: which feature families exist.
    """

    action = parse_worker_action_output(raw, allowlist={"features_catalog"})

    assert action.action_type == "tool_call"
    assert action.tool == "features_catalog"
    assert action.arguments == {"scope": "all"}


def test_parse_worker_action_output_flattens_single_tool_named_argument_wrapper() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "datasets_preview",
      "arguments": {
        "datasets_preview": {
          "dataset_id": "BTCUSDT_4h",
          "view": "rows"
        }
      },
      "reason": "preview the dataset rows"
    }
    """

    action = parse_worker_action_output(raw, allowlist={"datasets_preview"})

    assert action.tool == "datasets_preview"
    assert action.arguments == {"dataset_id": "BTCUSDT_4h", "view": "rows"}


def test_parse_worker_action_output_flattens_tool_object_argument_wrapper() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "research_project",
      "arguments": {
        "tool": {
          "action": "create",
          "project": {
            "name": "active-signal-v1-cycle",
            "goal": "Lock baseline"
          }
        }
      },
      "reason": "create the project"
    }
    """

    action = parse_worker_action_output(raw, allowlist={"research_project"})

    assert action.tool == "research_project"
    assert action.arguments == {
        "action": "create",
        "project": {
            "name": "active-signal-v1-cycle",
            "goal": "Lock baseline",
        },
    }


def test_parse_worker_action_output_ignores_direct_json_after_valid_action() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "events",
      "arguments": {"view": "catalog"},
      "reason": "inspect events"
    }
    {
      "status": "ok",
      "tool_name": "events",
      "data": {"summary": "Completed events action catalog."}
    }
    """

    action = parse_worker_action_output(raw, allowlist={"events"})

    assert action.action_type == "tool_call"
    assert action.tool == "events"
    assert action.arguments == {"view": "catalog"}


def test_parse_worker_action_output_rejects_conflicting_nested_tool_wrapper() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "research_project",
      "arguments": {
        "tool": {
          "tool": "research_search",
          "arguments": {"query": "baseline"}
        }
      },
      "reason": "run a tool"
    }
    """

    with pytest.raises(StructuredOutputError, match="tool_argument_wrapper_conflicts_with_tool_name"):
        parse_worker_action_output(raw, allowlist={"research_project"})


def test_parse_worker_action_output_raises_structured_error_for_invalid_control_character_json() -> None:
    raw = '{"type":"tool_call","tool":"research_record","arguments":{"record":"bad \x01 text"},"reason":"record"}'

    with pytest.raises(StructuredOutputError, match="json_decode_error:Invalid control character"):
        parse_worker_action_output(raw, allowlist={"research_record"})


def test_parse_worker_action_output_recovers_reason_after_redundant_closer_before_top_level_fields() -> None:
    raw = """
    {
      "type": "tool_call",
      "tool": "research_record",
      "arguments": {
        "project_id": "proj_123",
        "action": "create",
        "record": {"title": "baseline"},
        "atlas": {"statement": "s", "coordinates": {"phase": "discovery"}}}},
      "reason": "record the cycle invariant",
      "expected_evidence": ["hypothesis created"]
    }
    """

    action = parse_worker_action_output(raw, allowlist={"research_record"})

    assert action.tool == "research_record"
    assert action.reason == "record the cycle invariant"
    assert action.expected_evidence == ["hypothesis created"]
