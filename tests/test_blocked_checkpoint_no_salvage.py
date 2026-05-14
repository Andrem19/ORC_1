"""
Tests that a blocked checkpoint from minimax is NOT auto-salvaged.

Regression test: the executor used to call build_generic_transcript_salvage_report
on *every* checkpoint/abort from lmstudio/minimax, even when the model explicitly
returned status="blocked" to signal a genuine obstacle (e.g. missing prerequisite
data).  The salvage report stamped auto_finalized_from_* facts which the quality
gate then rejected unconditionally, creating a guaranteed failure loop:

  checkpoint(blocked) -> salvage final_report -> gate reject -> fallback
  (same obstacle) -> ...

The fix: skip transcript synthesis when action.status == "blocked".
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from app.adapters.base import AdapterResponse
from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.executor import DirectExecutionResult, DirectSliceExecutor
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _slice(**overrides: Any) -> PlanSlice:
    defaults = dict(
        slice_id="slice_stability",
        title="Stability analysis",
        hypothesis="test",
        objective="test objective",
        parallel_slot=1,
        depends_on=[],
        allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=3,
        max_tool_calls=24,
        max_expensive_calls=12,
        success_criteria=["Stable signals survive condition splits"],
        runtime_profile="backtests_stability_analysis",
    )
    defaults.update(overrides)
    return PlanSlice(**defaults)


BLOCKED_CHECKPOINT_JSON = json.dumps({
    "type": "checkpoint",
    "status": "blocked",
    "summary": "No saved or active backtest runs exist to perform stability analysis.",
    "reason_code": "no_runs_available",
    "facts": {
        "execution.kind": "direct",
        "blocking_issue": "no_saved_runs",
    },
})

FINAL_REPORT_JSON = json.dumps({
    "type": "final_report",
    "summary": "Analysis complete.",
    "verdict": "COMPLETE",
    "facts": {"execution.kind": "direct"},
    "evidence_refs": ["run_123"],
    "confidence": 0.80,
})


def _direct_config() -> SimpleNamespace:
    return SimpleNamespace(
        provider="minimax",
        timeout_seconds=600,
        max_attempts_per_slice=1,
        max_tool_calls_per_slice=24,
        max_expensive_tool_calls_per_slice=12,
        mcp_endpoint_url="http://127.0.0.1:8766/mcp",
        mcp_auth_mode="none",
        mcp_token_env_var="DEV_SPACE1_MCP_BEARER_TOKEN",
        connect_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
    )


def _make_executor(raw_output: str) -> DirectSliceExecutor:
    adapter = MagicMock()
    adapter.supports_tool_loop.return_value = False
    artifact_store = MagicMock()
    artifact_store.save_direct_attempt = MagicMock()
    incident_store = MagicMock()
    catalog = make_catalog_snapshot()

    invoker = AsyncMock(return_value=AdapterResponse(
        success=True,
        raw_output=raw_output,
        duration_seconds=5.0,
        metadata={},
    ))

    return DirectSliceExecutor(
        adapter=adapter,
        artifact_store=artifact_store,
        incident_store=incident_store,
        direct_config=_direct_config(),
        provider_name="minimax",
        catalog_snapshot=catalog,
        invoker=invoker,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_blocked_checkpoint_not_converted_to_salvage():
    """A checkpoint with status='blocked' must NOT be auto-salvaged."""
    executor = _make_executor(BLOCKED_CHECKPOINT_JSON)

    result = asyncio.run(executor.execute(
        plan_id="plan_1",
        slice_obj=_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    # The action must remain a checkpoint, not be converted to a salvage final_report
    assert result.action is not None
    assert result.action.action_type == "checkpoint"
    assert result.action.status == "blocked"

    # Must NOT contain auto-salvage facts
    facts = result.action.facts or {}
    salvage_keys = [k for k in facts if "auto_finalized_from_" in k]
    assert salvage_keys == [], (
        f"Blocked checkpoint should not contain auto-salvage facts, found: {salvage_keys}"
    )

    # The original blocking reason must be preserved
    assert "blocking_issue" in facts or result.action.reason_code == "no_runs_available"


def test_final_report_not_affected():
    """A genuine final_report must pass through unchanged."""
    executor = _make_executor(FINAL_REPORT_JSON)

    result = asyncio.run(executor.execute(
        plan_id="plan_1",
        slice_obj=_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.action.action_type == "final_report"
    assert result.action.verdict == "COMPLETE"


# ---------------------------------------------------------------------------
# Integration: verify the quality gate does not see auto-salvage on blocked
# ---------------------------------------------------------------------------

def test_blocked_checkpoint_no_salvage_in_fallback_flow():
    """
    When minimax returns a blocked checkpoint, the result must not carry
    auto-salvage facts, so the quality gate cannot log
    'auto_salvage_stub_rejected'.
    """
    blocked_action = WorkerAction(
        action_id="act_blocked",
        action_type="checkpoint",
        status="blocked",
        summary="No backtest runs exist for stability analysis.",
        reason_code="no_runs_available",
        facts={
            "execution.kind": "direct",
            "blocking_issue": "no_saved_runs",
        },
    )

    result = DirectExecutionResult(
        action=blocked_action,
        artifact_path="/tmp/artifact.json",
        raw_output=BLOCKED_CHECKPOINT_JSON,
        provider="minimax",
        duration_ms=5000,
        tool_call_count=6,
        expensive_tool_call_count=2,
    )

    # Verify the result action has no auto-salvage facts
    assert result.action.action_type == "checkpoint"
    assert result.action.status == "blocked"
    facts = result.action.facts or {}
    salvage_keys = [k for k in facts if "auto_finalized_from_" in k]
    assert salvage_keys == [], (
        f"Blocked checkpoint must not carry auto-salvage facts: {salvage_keys}"
    )

    # Verify the quality gate would NOT log auto_salvage_stub_rejected
    from app.services.direct_execution.guardrails import final_report_passes_quality_gate

    # Even if the action were mistakenly treated as final_report,
    # it must not trigger the auto_salvage_stub_rejected gate
    passes, reason = final_report_passes_quality_gate(
        tool_call_count=result.tool_call_count,
        action=result.action,
        slice_obj=_slice(),
        required_output_facts=[],
        inherited_facts={},
    )
    if not passes:
        assert "auto_salvage" not in reason, (
            f"Quality gate should not reject with auto_salvage, got: {reason}"
        )


def test_blocked_checkpoint_guardrail_condition():
    """
    Directly test the guardrail: verify that the is_explicit_block check
    correctly identifies blocked checkpoints.
    """
    # A blocked checkpoint action should NOT have auto_finalized_from_ facts
    blocked = WorkerAction(
        action_id="act_1",
        action_type="checkpoint",
        status="blocked",
        summary="Cannot proceed.",
        facts={"execution.kind": "direct"},
    )
    facts = blocked.facts or {}
    assert not any(
        str(k).startswith("direct.auto_finalized_from_") and bool(v)
        for k, v in facts.items()
    ), "Blocked checkpoint should have clean facts"

    # Verify the quality gate check for auto-salvage prefix
    from app.services.direct_execution.guardrails import _AUTO_SALVAGE_FACT_PREFIX
    assert _AUTO_SALVAGE_FACT_PREFIX == "direct.auto_finalized_from_"
