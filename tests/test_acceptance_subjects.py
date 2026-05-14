"""Tests for acceptance subject extraction helpers."""

from __future__ import annotations

import json

from app.execution_models import WorkerAction
from app.services.direct_execution.acceptance.subjects import (
    _extract_cf_names_from_payload,
    _is_plausible_node_id,
    feature_names_from_action_and_transcript,
    node_ids_from_action_and_transcript,
    run_ids_from_action_and_transcript,
)


def _make_feature_list_payload(features: list[dict]) -> dict:
    """Build a realistic features_custom MCP response payload."""
    text = json.dumps({
        "status": "ok",
        "data": {"features": features},
    })
    return {
        "ok": True,
        "payload": {
            "content": [{"type": "text", "text": text}],
        },
    }


# ---------- feature_names_from_action_and_transcript ----------


def test_extracts_feature_name_from_facts() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"feature_name": "cf_test_signal"},
    )
    result = feature_names_from_action_and_transcript(action, [])
    assert result == ["cf_test_signal"]


def test_extracts_cf_names_from_facts_list() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"feature_names": ["cf_alpha", "cf_beta", "not_cf"]},
    )
    result = feature_names_from_action_and_transcript(action, [])
    assert result == ["cf_alpha", "cf_beta"]


def test_extracts_name_from_tool_call_arguments() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    transcript = [
        {
            "tool": "features_custom",
            "arguments": {"action": "validate", "name": "cf_momentum"},
        },
    ]
    result = feature_names_from_action_and_transcript(action, transcript)
    assert result == ["cf_momentum"]


def test_extracts_names_from_response_payload_as_fallback() -> None:
    """When model calls features_custom(action='list') without a name argument,
    feature names should be extracted from the response payload."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    payload = _make_feature_list_payload([
        {"name": "cf_btc_wave_b_signal", "status": "published"},
        {"name": "cf_cross_tf_divergence", "status": "published"},
        {"name": "cf_eth_lead", "status": "published"},
    ])
    transcript = [
        {
            "tool": "features_custom",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": payload,
        },
    ]
    result = feature_names_from_action_and_transcript(action, transcript)
    assert "cf_btc_wave_b_signal" in result
    assert "cf_cross_tf_divergence" in result
    assert "cf_eth_lead" in result


def test_prefers_name_arg_over_payload() -> None:
    """When the tool call has a name argument, payload extraction is skipped."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    payload = _make_feature_list_payload([
        {"name": "cf_other", "status": "published"},
    ])
    transcript = [
        {
            "tool": "features_custom",
            "arguments": {"action": "validate", "name": "cf_target"},
            "payload": payload,
        },
    ]
    result = feature_names_from_action_and_transcript(action, transcript)
    assert result == ["cf_target"]


def test_does_not_extract_from_other_tools() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    transcript = [
        {
            "tool": "features_dataset",
            "arguments": {"action": "inspect", "view": "columns"},
            "payload": _make_feature_list_payload([
                {"name": "cf_should_not_appear"},
            ]),
        },
    ]
    result = feature_names_from_action_and_transcript(action, transcript)
    assert result == []


def test_empty_when_no_sources() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    result = feature_names_from_action_and_transcript(action, [])
    assert result == []


def test_deduplicates_names() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"feature_name": "cf_alpha", "features": ["cf_alpha", "cf_beta"]},
    )
    result = feature_names_from_action_and_transcript(action, [])
    assert result == ["cf_alpha", "cf_beta"]


def test_exact_incident_pattern() -> None:
    """Reproduce the exact scenario from run 20260415T235735Z-923dec6e."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"runtime_state": "empty"},  # MiniMax reported empty despite 23 features
    )
    features_data = [
        {"name": f"cf_feature_{i}"} for i in range(23)
    ]
    payload = _make_feature_list_payload(features_data)
    transcript = [
        {
            "tool": "features_custom",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": payload,
        },
    ]
    result = feature_names_from_action_and_transcript(action, transcript)
    assert len(result) == 23
    assert result[0] == "cf_feature_0"
    assert result[-1] == "cf_feature_22"


# ---------- _extract_cf_names_from_payload ----------


def test_extract_cf_names_handles_valid_payload() -> None:
    payload = _make_feature_list_payload([
        {"name": "cf_alpha", "status": "published"},
        {"name": "cf_beta", "status": "draft"},
    ])
    result = _extract_cf_names_from_payload(payload)
    assert result == ["cf_alpha", "cf_beta"]


def test_extract_cf_names_ignores_non_cf_names() -> None:
    payload = _make_feature_list_payload([
        {"name": "rsi_14", "status": "published"},
        {"name": "cf_target", "status": "published"},
    ])
    result = _extract_cf_names_from_payload(payload)
    assert result == ["cf_target"]


def test_extract_cf_names_returns_empty_for_none() -> None:
    assert _extract_cf_names_from_payload(None) == []


def test_extract_cf_names_returns_empty_for_non_dict() -> None:
    assert _extract_cf_names_from_payload("not a dict") == []


def test_extract_cf_names_returns_empty_for_missing_content() -> None:
    assert _extract_cf_names_from_payload({"ok": True, "payload": {}}) == []


def test_extract_cf_names_returns_empty_for_non_json_text() -> None:
    assert _extract_cf_names_from_payload({
        "ok": True,
        "payload": {"content": [{"type": "text", "text": "not json"}]},
    }) == []


def test_extract_cf_names_returns_empty_for_no_features_key() -> None:
    text = json.dumps({"status": "ok", "data": {"items": []}})
    assert _extract_cf_names_from_payload({
        "ok": True,
        "payload": {"content": [{"type": "text", "text": text}]},
    }) == []


# ---------- run_ids_from_action_and_transcript ----------


def test_run_ids_extracts_from_facts() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"run_id": "20260416-021356-efed0832"},
    )
    result = run_ids_from_action_and_transcript(action, [])
    assert result == ["20260416-021356-efed0832"]


def test_run_ids_extracts_from_arguments() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "run_id": "20260416-021356-efed0832"},
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    assert result == ["20260416-021356-efed0832"]


def test_run_ids_does_not_pollute_when_facts_present() -> None:
    """When run_ids are available in facts, response payloads must NOT inject
    additional historical run_ids into the acceptance verification set."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"run_id": "20260416-021356-efed0832"},
    )
    # Simulate a list response with many old run_ids
    old_runs_text = json.dumps({
        "status": "ok",
        "data": {
            "saved_runs": [
                {"run_id": f"20260401-120000-run{i:08d}", "status": "completed"}
                for i in range(100)
            ],
            "active_runs": [],
        },
    })
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": old_runs_text}],
                },
            },
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    # Should ONLY have the run_id from facts, not the 100 old runs from payload
    assert result == ["20260416-021356-efed0832"]


def test_run_ids_extracts_base_and_candidate_run_ids() -> None:
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={
            "base_run_id": "20260416-021356-aaaa0000",
            "candidate_run_id": "20260416-021356-bbbb1111",
        },
    )
    result = run_ids_from_action_and_transcript(action, [])
    assert "20260416-021356-aaaa0000" in result
    assert "20260416-021356-bbbb1111" in result


def test_run_ids_extracts_from_string_values_in_facts() -> None:
    """Run IDs embedded in arbitrary string fact values should be found."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"description": "Used run 20260416-021356-abc12345 for analysis"},
    )
    result = run_ids_from_action_and_transcript(action, [])
    assert "20260416-021356-abc12345" in result


def test_run_ids_fallback_to_payload_when_facts_empty() -> None:
    """When no run_ids in facts or arguments, extract a limited number from
    tool response payloads.  This helps weaker models (e.g. minimax) that
    successfully call backtests_runs(view=list) but do not echo run_ids in
    their output facts."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    runs_text = json.dumps({
        "status": "ok",
        "data": {
            "saved_runs": [
                {"run_id": "20260416-021356-efed0832", "status": "completed"},
                {"run_id": "20260416-021400-aabbccdd", "status": "completed"},
                {"run_id": "20260416-021500-11223344", "status": "completed"},
                {"run_id": "20260416-021600-55667788", "status": "completed"},
            ],
            "active_runs": [],
        },
    })
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": runs_text}],
                },
            },
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    # Should extract up to 3 run_ids from the payload as fallback
    assert len(result) <= 3
    assert len(result) >= 1
    # All extracted IDs must match the expected format
    for rid in result:
        assert rid.startswith("2026")


def test_run_ids_fallback_capped_at_max() -> None:
    """Payload fallback must not extract more than 3 run_ids total."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    runs_text = json.dumps({
        "status": "ok",
        "data": {
            "saved_runs": [
                {"run_id": f"20260401-120000-{i:08x}", "status": "completed"}
                for i in range(100)
            ],
            "active_runs": [],
        },
    })
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": runs_text}],
                },
            },
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    assert len(result) <= 3
    assert len(result) >= 1


def test_run_ids_fallback_not_used_when_facts_present() -> None:
    """Payload extraction is a fallback only; facts/arguments take priority."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={"run_id": "20260416-021356-efed0832"},
    )
    runs_text = json.dumps({
        "status": "ok",
        "data": {
            "saved_runs": [
                {"run_id": "20260415-120000-oldrun01", "status": "completed"},
            ],
            "active_runs": [],
        },
    })
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "view": "list"},
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": runs_text}],
                },
            },
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    # Should ONLY have the run_id from facts, not the one from payload
    assert result == ["20260416-021356-efed0832"]


def test_run_ids_fallback_from_direct_payload_run_id() -> None:
    """Payload fallback also works for direct payload with run_id key."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={},
    )
    transcript = [
        {
            "kind": "tool_result",
            "tool": "backtests_runs",
            "arguments": {"action": "inspect", "view": "detail"},
            "payload": {"run_id": "20260416-183638-1c9ca540"},
        },
    ]
    result = run_ids_from_action_and_transcript(action, transcript)
    assert "20260416-183638-1c9ca540" in result


# ---------- node_ids_from_action_and_transcript: node_types false positive ----------


def test_node_types_not_extracted_as_node_id() -> None:
    """Regression: 'node_types' from research_memory search arguments or
    assistant reasoning payloads must NOT appear in extracted node IDs.

    Root cause of slice compiled_plan_v1_stage_2 fallback failure:
    the regex ``\\b(?:node|note|incident)_[A-Za-z0-9_-]+\\b`` matched
    ``node_types`` from the stringified transcript payload.  The MCP
    prove endpoint then failed for ``node_id="node_types"`` because no
    such node exists, causing ``research_node_proof_pass`` to report
    FAIL and triggering an unnecessary fallback.
    """
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={
            "research.memory_node_id": "node-c89611cb87e243afbd507831afb06f39",
            "research.hypothesis_refs": ["node-c89611cb87e243afbd507831afb06f39"],
            "project_id": "cycle-invariants-v1-31651827",
        },
        evidence_refs=["node-c89611cb87e243afbd507831afb06f39"],
    )
    # Simulate a realistic transcript with node_types in search arguments
    # and assistant reasoning.
    transcript = [
        {
            "kind": "tool_result",
            "tool": "research_memory",
            "arguments": {
                "action": "search",
                "node_types": ["note", "hypothesis", "result"],
                "project_id": "cycle-invariants-v1-31651827",
            },
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": '{"status":"ok","data":{}}'}],
                },
            },
        },
        {
            "kind": "assistant_response",
            "payload": {
                "choices": [
                    {
                        "message": {
                            "content": "I will search with node_types to find relevant nodes.",
                        },
                    },
                ],
            },
        },
    ]
    ids = node_ids_from_action_and_transcript(action, transcript)
    assert "node_types" not in ids, f"node_types should not be in extracted IDs: {ids}"
    assert "node-c89611cb87e243afbd507831afb06f39" in ids


def test_node_types_in_created_ids_transcript_payload() -> None:
    """Even when node_types appears inside a deeply nested MCP response
    payload (from research_memory create), it must not be treated as a
    node ID."""
    action = WorkerAction(
        action_id="a1",
        action_type="final_report",
        summary="done",
        facts={
            "research.memory_node_id": "node-abc123",
        },
    )
    # MCP response with node_types in the response data
    response_data = json.dumps({
        "status": "ok",
        "data": {
            "action": "create",
            "node_id": "node-abc123",
            "node_types": ["milestone"],
            "record": {"node_type": "milestone"},
        },
    })
    transcript = [
        {
            "kind": "tool_result",
            "tool": "research_memory",
            "arguments": {"action": "create", "node_id": "node-abc123"},
            "payload": {
                "ok": True,
                "payload": {
                    "content": [{"type": "text", "text": response_data}],
                },
            },
        },
    ]
    ids = node_ids_from_action_and_transcript(action, transcript)
    assert "node_types" not in ids
    assert "node-abc123" in ids


# ---------- _is_plausible_node_id ----------


def test_plausible_node_id_accepts_dash_format() -> None:
    assert _is_plausible_node_id("node-c89611cb87e243afbd507831afb06f39") is True


def test_plausible_node_id_rejects_node_types() -> None:
    assert _is_plausible_node_id("node_types") is False


def test_plausible_node_id_rejects_node_id() -> None:
    assert _is_plausible_node_id("node_id") is False


def test_plausible_node_id_accepts_short_underscore_format() -> None:
    """Short underscore-format IDs from structured extraction are accepted."""
    assert _is_plausible_node_id("node_abc123") is True
    assert _is_plausible_node_id("node_correct") is True


def test_plausible_node_id_rejects_note_types() -> None:
    assert _is_plausible_node_id("note_types") is False
    assert _is_plausible_node_id("note_type") is False


def test_plausible_node_id_rejects_incident_variants() -> None:
    assert _is_plausible_node_id("incident_types") is False
    assert _is_plausible_node_id("incident_severity") is False


def test_plausible_node_id_rejects_node_created() -> None:
    """Regression: minimax models report facts with key 'node_created' which
    matches the node ID regex but is a dict key, not a real node ID.
    The MCP prove endpoint fails for node_id='node_created' causing
    research_node_proof_pass to report FAIL."""
    assert _is_plausible_node_id("node_created") is False
    assert _is_plausible_node_id("node_updated") is False
    assert _is_plausible_node_id("node_deleted") is False


def test_plausible_node_id_rejects_empty_and_non_prefixed() -> None:
    assert _is_plausible_node_id("") is False
    assert _is_plausible_node_id("project-abc") is False
    assert _is_plausible_node_id("branch-xyz") is False
