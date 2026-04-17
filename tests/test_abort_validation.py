"""
Unit tests for abort_validation module.

Tests the detection of hallucinated abort claims and transcript-validated
correction prompts that are injected into retry attempts.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.services.direct_execution.abort_validation import (
    abort_claims_empty_results,
    build_transcript_correction_prompt,
    extract_transcript_evidence_summary,
    transcript_has_successful_tool_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool_result(
    tool: str,
    *,
    ok: bool = True,
    error: str = "",
    arguments: dict[str, Any] | None = None,
    structured_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a tool_result transcript entry."""
    payload: dict[str, Any] = {}
    if ok:
        payload["ok"] = True
        inner_payload: dict[str, Any] = {}
        if structured_data:
            inner_payload["structuredContent"] = {
                "status": "ok",
                "data": structured_data,
            }
        else:
            inner_payload["structuredContent"] = {"status": "ok", "data": {}}
        payload["payload"] = inner_payload
    else:
        payload["ok"] = False
        payload["error"] = error
    return {
        "kind": "tool_result",
        "tool": tool,
        "arguments": arguments or {},
        "payload": payload,
    }


def _features_dataset_result(columns: list[str]) -> dict[str, Any]:
    return _tool_result(
        "features_dataset",
        arguments={"action": "inspect", "view": "columns", "symbol": "BTCUSDT", "timeframe": "1h"},
        structured_data={"view": "columns", "columns": columns},
    )


def _features_custom_contract_result() -> dict[str, Any]:
    return _tool_result(
        "features_custom",
        arguments={"action": "inspect", "view": "contract"},
        structured_data={
            "view": "contract",
            "contract": {
                "entrypoint": "def compute_series(ctx): ...",
                "preferred_entrypoint_name": "compute_series",
                "required_name_prefix": "cf_",
            },
        },
    )


def _research_memory_result(total: int = 5) -> dict[str, Any]:
    return _tool_result(
        "research_memory",
        arguments={"action": "search", "project_id": "test-project"},
        structured_data={
            "memory_matches": {
                "total": total,
                "results": [{"node_id": f"node-{i}"} for i in range(min(total, 3))],
            },
        },
    )


def _assistant_response() -> dict[str, Any]:
    return {"kind": "assistant_response", "payload": {"choices": []}}


# ---------------------------------------------------------------------------
# abort_claims_empty_results
# ---------------------------------------------------------------------------

class TestAbortClaimsEmptyResults:
    def test_infrastructure_data_unavailable(self):
        assert abort_claims_empty_results("INFRASTRUCTURE_DATA_UNAVAILABLE", "no data", "") is True

    def test_empty_results_claim(self):
        assert abort_claims_empty_results("", "tools returned empty results", "") is True

    def test_no_columns_available(self):
        assert abort_claims_empty_results("", "", "no columns available for BTCUSDT") is True

    def test_no_features_found(self):
        assert abort_claims_empty_results("", "no custom features found", "") is True

    def test_no_datasets_found(self):
        assert abort_claims_empty_results("", "", "no datasets found") is True

    def test_legitimate_abort(self):
        assert abort_claims_empty_results("TIMEOUT", "execution timed out", "") is False

    def test_empty_strings(self):
        assert abort_claims_empty_results("", "", "") is False

    def test_unrelated_reason(self):
        assert abort_claims_empty_results("CONTRACT_VIOLATION", "schema mismatch", "bad args") is False


# ---------------------------------------------------------------------------
# transcript_has_successful_tool_data
# ---------------------------------------------------------------------------

class TestTranscriptHasSuccessfulToolData:
    def test_empty_transcript(self):
        assert transcript_has_successful_tool_data([]) is False

    def test_only_assistant_responses(self):
        assert transcript_has_successful_tool_data([_assistant_response()]) is False

    def test_successful_tool_result(self):
        assert transcript_has_successful_tool_data([_tool_result("research_memory")]) is True

    def test_failed_tool_result(self):
        assert transcript_has_successful_tool_data([
            _tool_result("research_memory", ok=False, error="connection failed"),
        ]) is False

    def test_mixed_results(self):
        assert transcript_has_successful_tool_data([
            _assistant_response(),
            _tool_result("research_memory"),
        ]) is True


# ---------------------------------------------------------------------------
# extract_transcript_evidence_summary
# ---------------------------------------------------------------------------

class TestExtractTranscriptEvidenceSummary:
    def test_features_dataset_columns(self):
        transcript = [
            _features_dataset_result(["atr_1", "rsi_1", "cf_funding_dislocation", "cf_expiry_proximity"]),
        ]
        summaries = extract_transcript_evidence_summary(transcript)
        assert len(summaries) == 1
        assert "4 columns" in summaries[0]
        assert "cf_funding_dislocation" in summaries[0]

    def test_features_custom_contract(self):
        transcript = [_features_custom_contract_result()]
        summaries = extract_transcript_evidence_summary(transcript)
        assert len(summaries) == 1
        assert "contract available" in summaries[0]
        assert "compute_series" in summaries[0]

    def test_research_memory(self):
        transcript = [_research_memory_result(total=10)]
        summaries = extract_transcript_evidence_summary(transcript)
        assert len(summaries) == 1
        assert "10 nodes" in summaries[0]

    def test_max_items_limit(self):
        transcript = [_tool_result(f"tool_{i}") for i in range(20)]
        summaries = extract_transcript_evidence_summary(transcript, max_items=3)
        assert len(summaries) == 3

    def test_skips_failed_results(self):
        transcript = [
            _tool_result("features_dataset", ok=False, error="timeout"),
            _features_dataset_result(["atr_1"]),
        ]
        summaries = extract_transcript_evidence_summary(transcript)
        assert len(summaries) == 1

    def test_empty_transcript(self):
        assert extract_transcript_evidence_summary([]) == []


# ---------------------------------------------------------------------------
# build_transcript_correction_prompt
# ---------------------------------------------------------------------------

class TestBuildTranscriptCorrectionPrompt:
    def test_empty_transcript_returns_empty(self):
        assert build_transcript_correction_prompt([]) == ""

    def test_with_evidence_produces_correction(self):
        transcript = [
            _features_dataset_result(["cf_funding_dislocation", "cf_expiry_proximity", "rsi_1"]),
        ]
        prompt = build_transcript_correction_prompt(transcript)
        assert "CRITICAL" in prompt
        assert "incorrectly" in prompt
        assert "cf_funding_dislocation" in prompt
        assert "Do NOT abort" in prompt

    def test_with_failed_tools_returns_empty(self):
        transcript = [
            _tool_result("features_dataset", ok=False, error="timeout"),
        ]
        prompt = build_transcript_correction_prompt(transcript)
        assert prompt == ""

    def test_multiple_tools(self):
        transcript = [
            _features_dataset_result(["atr_1", "rsi_1"]),
            _features_custom_contract_result(),
            _research_memory_result(total=5),
        ]
        prompt = build_transcript_correction_prompt(transcript)
        assert "features_dataset" in prompt
        assert "features_custom" in prompt
        assert "research_memory" in prompt


# ---------------------------------------------------------------------------
# Integration: realistic transcript from the actual failure
# ---------------------------------------------------------------------------

class TestRealisticMiniMaxTranscript:
    """Replicate the exact scenario from the real failure."""

    def _realistic_transcript(self) -> list[dict[str, Any]]:
        """Build a transcript matching the real MiniMax failure pattern."""
        return [
            _assistant_response(),
            _research_memory_result(total=9),
            _assistant_response(),
            _features_custom_contract_result(),
            _assistant_response(),
            _tool_result(
                "features_custom",
                arguments={"action": "inspect", "view": "list"},
                structured_data={"view": "list", "features": [
                    {"name": "cf_funding_dislocation"},
                    {"name": "cf_expiry_proximity"},
                    {"name": "cf_eth_lead"},
                ]},
            ),
            _assistant_response(),
            _features_dataset_result([
                "atr_1", "cl_15m", "cl_1d", "cl_1h", "cl_4h", "dow", "hour",
                "iv_est_1", "rsi_1", "iv_est", "rsi", "is_funding_settlement",
                "is_expiry", "bars_until_funding_settlement", "bars_until_expiry",
                "cf_vol_term_spread", "cf_liquidity_imbalance", "cf_eth_lead",
                "cf_funding_dislocation", "cf_expiry_proximity", "cf_funding_expiry_cross",
            ]),
            _assistant_response(),
            _tool_result(
                "features_dataset",
                arguments={"action": "inspect", "view": "list"},
                structured_data={"view": "list", "datasets": [
                    {"dataset_id": "binance/um/BTCUSDT/1h"},
                ]},
            ),
            _assistant_response(),
            _tool_result(
                "models_dataset",
                arguments={"action": "inspect", "view": "list"},
                payload={
                    "ok": True,
                    "payload": {
                        "structuredContent": {
                            "status": "ok",
                            "data": {"datasets": []},
                        },
                    },
                },
            ),
        ]

    def test_abort_detected_as_hallucinated(self):
        """The abort reason 'INFRASTRUCTURE_DATA_UNAVAILABLE' claims empty results."""
        assert abort_claims_empty_results(
            "INFRASTRUCTURE_DATA_UNAVAILABLE",
            "Cannot establish data/feature contracts: all domain tools return empty results",
            "",
        ) is True

    def test_transcript_shows_successful_data(self):
        """The transcript has many successful tool results."""
        assert transcript_has_successful_tool_data(self._realistic_transcript()) is True

    def test_correction_prompt_includes_actual_columns(self):
        """The correction prompt lists actual columns contradicting the abort."""
        transcript = self._realistic_transcript()
        prompt = build_transcript_correction_prompt(transcript)
        assert prompt != ""
        assert "cf_funding_dislocation" in prompt
        assert "cf_expiry_proximity" in prompt

    def test_evidence_summary_captures_multiple_tools(self):
        """The evidence summary captures data from multiple tools."""
        transcript = self._realistic_transcript()
        summaries = extract_transcript_evidence_summary(transcript)
        tool_names = [s.split(" ")[0] for s in summaries]
        assert "research_memory" in tool_names
        assert "features_custom" in tool_names
        assert "features_dataset" in tool_names
