"""Tests for acceptance subject extraction helpers."""

from __future__ import annotations

import json

from app.execution_models import WorkerAction
from app.services.direct_execution.acceptance.subjects import (
    _extract_cf_names_from_payload,
    feature_names_from_action_and_transcript,
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


def test_run_ids_does_not_extract_from_response_payloads() -> None:
    """Response payloads (e.g., list views with 100 saved runs) must NOT
    inject historical run_ids into the acceptance verification set.
    Only facts and arguments should be sources."""
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
