"""Tests for feature profitability filter auto-finalizer evidence checks."""

from __future__ import annotations

import json

from app.services.direct_execution.feature_contract_runtime import (
    _catalog_cf_feature_names,
    _has_sufficient_profitability_filter_evidence,
    build_feature_profitability_filter_final_report,
)


def _make_catalog_response_payload(columns: list[dict]) -> dict:
    """Build a realistic features_catalog MCP response payload."""
    text = json.dumps({
        "status": "ok",
        "data": {"columns": columns},
    })
    return {
        "ok": True,
        "payload": {
            "content": [{"type": "text", "text": text}],
        },
    }


def _make_tool_entry(tool: str, arguments: dict, payload: dict | None = None) -> dict:
    entry: dict = {"kind": "tool_result", "tool": tool, "arguments": arguments}
    if payload is not None:
        entry["payload"] = payload
    else:
        entry["payload"] = {"ok": True}
    return entry


def _make_error_tool_entry(tool: str, arguments: dict, message: str = "error") -> dict:
    """Build a tool entry where the MCP response returned an error status."""
    return {
        "kind": "tool_result",
        "tool": tool,
        "arguments": arguments,
        "payload": {
            "ok": True,
            "payload": {
                "content": [{"type": "text", "text": json.dumps({"status": "error", "message": message})}],
                "structuredContent": {"status": "error", "message": message},
            },
        },
    }


# ---------- _catalog_cf_feature_names ----------


def test_extracts_cf_names_from_catalog_response() -> None:
    payload = _make_catalog_response_payload([
        {"name": "cf_funding_dislocation", "type": "custom"},
        {"name": "cf_expiry_proximity", "type": "custom"},
        {"name": "rsi_1", "type": "managed"},
    ])
    transcript = [_make_tool_entry("features_catalog", {"scope": "available"}, payload)]
    result = _catalog_cf_feature_names(transcript)
    assert result == ["cf_funding_dislocation", "cf_expiry_proximity"]


def test_returns_empty_for_no_catalog_calls() -> None:
    transcript = [_make_tool_entry("features_analytics", {"action": "status"})]
    assert _catalog_cf_feature_names(transcript) == []


def test_returns_empty_when_no_cf_names() -> None:
    payload = _make_catalog_response_payload([
        {"name": "rsi_1", "type": "managed"},
        {"name": "atr_1", "type": "managed"},
    ])
    transcript = [_make_tool_entry("features_catalog", {"scope": "available"}, payload)]
    assert _catalog_cf_feature_names(transcript) == []


def test_deduplicates_cf_names() -> None:
    payload = _make_catalog_response_payload([
        {"name": "cf_alpha"},
        {"name": "cf_alpha"},
        {"name": "cf_beta"},
    ])
    transcript = [_make_tool_entry("features_catalog", {"scope": "available"}, payload)]
    assert _catalog_cf_feature_names(transcript) == ["cf_alpha", "cf_beta"]


def test_handles_malformed_payload_gracefully() -> None:
    transcript = [
        {"tool": "features_catalog", "arguments": {}, "payload": "not a dict"},
        {"tool": "features_catalog", "arguments": {}, "payload": {"ok": True}},
        {"tool": "features_catalog", "arguments": {}, "payload": {"ok": True, "payload": {}}},
    ]
    assert _catalog_cf_feature_names(transcript) == []


# ---------- _has_sufficient_profitability_filter_evidence ----------


def test_primary_path_two_analytics_with_feature_names() -> None:
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}),
        _make_tool_entry("features_analytics", {"action": "analytics", "feature_name": "cf_alpha"}),
        _make_tool_entry("features_analytics", {"action": "heatmap", "feature_name": "cf_alpha"}),
    ]
    assert _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_relaxed_path_one_analytics_plus_catalog_cf_names() -> None:
    """The exact scenario from run 20260416T002035Z-a90bfcdb:
    Model calls features_analytics(action=status) once and features_catalog returns cf_ names."""
    catalog_payload = _make_catalog_response_payload([
        {"name": "cf_funding_dislocation"},
        {"name": "cf_expiry_proximity"},
        {"name": "cf_cross_tf_divergence"},
    ])
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_tool_entry("features_analytics", {"action": "status"}),
    ]
    assert _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_insufficient_one_analytics_no_catalog_features() -> None:
    """One analytics action but no cf_ names from catalog — insufficient."""
    payload = _make_catalog_response_payload([{"name": "rsi_1"}])
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, payload),
        _make_tool_entry("features_analytics", {"action": "status"}),
    ]
    assert not _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_insufficient_no_analytics_at_all() -> None:
    catalog_payload = _make_catalog_response_payload([{"name": "cf_alpha"}])
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
    ]
    assert not _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_insufficient_missing_research_memory() -> None:
    catalog_payload = _make_catalog_response_payload([{"name": "cf_alpha"}])
    transcript = [
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_tool_entry("features_analytics", {"action": "status"}),
    ]
    assert not _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_insufficient_missing_required_catalog() -> None:
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_analytics", {"action": "analytics", "feature_name": "cf_alpha"}),
        _make_tool_entry("features_analytics", {"action": "heatmap", "feature_name": "cf_alpha"}),
    ]
    # features_catalog is in allowed_tools but was never called
    assert not _has_sufficient_profitability_filter_evidence(
        transcript, allowed_tools={"research_memory", "features_catalog", "features_analytics"}
    )


def test_exact_incident_pattern() -> None:
    """Exact scenario from incident: features_analytics(status) + features_catalog(available) with 22 cf_ features."""
    features = [{"name": f"cf_feature_{i}"} for i in range(22)]
    catalog_payload = _make_catalog_response_payload(features)
    transcript = [
        _make_tool_entry("research_memory", {"action": "search", "project_id": "proj-1"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_tool_entry("features_analytics", {"action": "status"}),
        _make_tool_entry("research_map", {"action": "inspect", "view": "detail"}),
        _make_tool_entry("research_memory", {"action": "create", "kind": "note"}),
    ]
    assert _has_sufficient_profitability_filter_evidence(
        transcript,
        allowed_tools={"research_memory", "features_catalog", "features_analytics", "research_map"},
    )


def test_catalog_only_path_analytics_failed_with_wrong_names() -> None:
    """Exact scenario from run 20260416T005102Z-614d9fd7:
    Model calls features_analytics with wrong feature names (volume_imbalance, cycle_phase),
    both fail with error status. But features_catalog returned 23 cf_ features."""
    catalog_payload = _make_catalog_response_payload(
        [{"name": f"cf_feature_{i}"} for i in range(23)]
    )
    transcript = [
        _make_tool_entry("research_memory", {"action": "search", "project_id": "proj-1"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_error_tool_entry(
            "features_analytics",
            {"action": "analytics", "feature_name": "volume_imbalance"},
            "Stored analytics are not ready for feature 'volume_imbalance'",
        ),
        _make_error_tool_entry(
            "features_analytics",
            {"action": "heatmap", "feature_name": "cycle_phase"},
            "Feature 'cycle_phase' is not active",
        ),
    ]
    assert _has_sufficient_profitability_filter_evidence(
        transcript,
        allowed_tools={"research_memory", "features_catalog", "features_analytics"},
    )


def test_catalog_only_path_requires_analytics_attempt() -> None:
    """Catalog-only path does not trigger if features_analytics was never attempted."""
    catalog_payload = _make_catalog_response_payload([{"name": "cf_alpha"}])
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
    ]
    assert not _has_sufficient_profitability_filter_evidence(
        transcript,
        allowed_tools={"research_memory", "features_catalog", "features_analytics"},
    )


def test_catalog_only_path_requires_cf_names() -> None:
    """Catalog-only path does not trigger if catalog returned no cf_ features."""
    catalog_payload = _make_catalog_response_payload([{"name": "rsi_1"}])
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_error_tool_entry(
            "features_analytics",
            {"action": "analytics", "feature_name": "volume_imbalance"},
            "not ready",
        ),
    ]
    assert not _has_sufficient_profitability_filter_evidence(
        transcript,
        allowed_tools={"research_memory", "features_catalog", "features_analytics"},
    )


# ---------- build_feature_profitability_filter_final_report ----------


def test_auto_finalizer_includes_shortlist_families_for_acceptance() -> None:
    """Auto-finalizer must emit research.shortlist_families so candidate_decision_set_present passes."""
    features = [{"name": f"cf_feature_{i}"} for i in range(5)]
    catalog_payload = _make_catalog_response_payload(features)
    transcript = [
        _make_tool_entry("research_memory", {"action": "search", "project_id": "proj-1"}),
        _make_tool_entry("features_catalog", {"scope": "available"}, catalog_payload),
        _make_tool_entry("features_analytics", {"action": "status"}),
    ]
    result = build_feature_profitability_filter_final_report(
        transcript=transcript,
        tool_name="research_memory",
        result_payload={"ok": True},
        allowed_tools={"research_memory", "features_catalog", "features_analytics"},
        slice_title="Fast filter via signal plausibility and feature profitability",
        slice_objective="Screen hypotheses before backtests.",
        success_criteria=["Only plausible hypotheses remain."],
        policy_tags=["filter", "cheap_first"],
        required_output_facts=[],
    )
    assert result is not None
    assert "research.shortlist_families" in result
    # Verify the feature names are present
    for name in ["cf_feature_0", "cf_feature_1", "cf_feature_2", "cf_feature_3"]:
        assert name in result


def test_auto_finalizer_uses_analytics_features_as_shortlist() -> None:
    """When analytics features are available, they become the shortlist."""
    transcript = [
        _make_tool_entry("research_memory", {"action": "search"}),
        _make_tool_entry("features_catalog", {"scope": "available"}),
        _make_tool_entry(
            "features_analytics",
            {"action": "analytics", "feature_name": "cf_alpha", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
        ),
        _make_tool_entry(
            "features_analytics",
            {"action": "heatmap", "feature_name": "cf_alpha", "symbol": "BTCUSDT", "anchor_timeframe": "1h"},
        ),
    ]
    result = build_feature_profitability_filter_final_report(
        transcript=transcript,
        tool_name="features_analytics",
        result_payload={"ok": True},
        allowed_tools={"research_memory", "features_catalog", "features_analytics"},
        slice_title="Fast filter via signal plausibility and feature profitability",
        slice_objective="Screen hypotheses before backtests.",
        success_criteria=["Only plausible hypotheses remain."],
        policy_tags=["filter"],
        required_output_facts=[],
    )
    assert result is not None
    assert "research.shortlist_families" in result
    assert "cf_alpha" in result
