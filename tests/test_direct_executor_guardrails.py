from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.adapters.base import AdapterResponse, BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.executor import DirectSliceExecutor
from app.services.direct_execution.guardrails import final_report_passes_quality_gate


class _FakeAdapter(BaseAdapter):
    def __init__(self, *, name: str, raw_output: str) -> None:
        self._name = name
        self._raw_output = raw_output

    def invoke(self, prompt: str, timeout: int = 120, **kwargs):  # type: ignore[no-untyped-def]
        del prompt, timeout, kwargs
        return AdapterResponse(success=True, raw_output=self._raw_output, duration_seconds=0.01)

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return self._name


def _slice() -> PlanSlice:
    return PlanSlice(
        slice_id="slice_1",
        title="slice",
        hypothesis="h",
        objective="o",
        success_criteria=["done"],
        allowed_tools=["research_record"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=2,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=1,
    )


def test_direct_attempt_ids_do_not_overwrite_between_primary_and_fallback(tmp_path) -> None:
    adapter = _FakeAdapter(
        name="qwen_worker_cli",
        raw_output='```json\n{"type":"final_report","summary":"ok","verdict":"COMPLETE","findings":[],"facts":{},"evidence_refs":["artifact_1"],"confidence":0.8}\n```',
    )
    artifact_store = ExecutionArtifactStore(tmp_path)
    cfg = SimpleNamespace(
        provider="lmstudio",
        timeout_seconds=30,
        max_attempts_per_slice=1,
        max_tool_calls_per_slice=4,
        max_expensive_tool_calls_per_slice=0,
        mcp_endpoint_url="http://127.0.0.1:8766/mcp",
        mcp_auth_mode="none",
        mcp_token_env_var="DEV_SPACE1_MCP_BEARER_TOKEN",
        connect_timeout_seconds=1.0,
        read_timeout_seconds=1.0,
        retry_budget=0,
        safe_exclude_tools=[],
        first_action_timeout_seconds=1,
        stalled_action_timeout_seconds=1,
    )
    executor = DirectSliceExecutor(
        adapter=adapter,
        artifact_store=artifact_store,
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=cfg,
        worker_system_prompt="",
        provider_name="qwen_cli",
    )

    primary = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=_slice(),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
            attempt_label="direct",
        )
    )
    fallback = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=_slice(),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
            attempt_label="fallback_1",
        )
    )

    assert primary.artifact_path != fallback.artifact_path
    assert primary.provider == "qwen_cli"
    assert fallback.provider == "qwen_cli"
    assert primary.artifact_path.endswith("direct_slice_1_1.json")
    assert fallback.artifact_path.endswith("fallback_1_slice_1_1.json")


# ---------------------------------------------------------------------------
# Unit tests for final_report_passes_quality_gate
# ---------------------------------------------------------------------------


def _gate_action(**overrides: Any) -> WorkerAction:
    defaults = dict(
        action_id="gate_test",
        action_type="final_report",
        summary="Test result",
        verdict="SUCCESS",
        confidence=0.8,
        evidence_refs=["node_test_1"],
        facts={"research.project_id": "proj_test"},
    )
    defaults.update(overrides)
    return WorkerAction(**defaults)


def _gate_slice() -> PlanSlice:
    return PlanSlice(
        slice_id="gate_slice",
        title="gate",
        hypothesis="h",
        objective="o",
        success_criteria=[],
        allowed_tools=["research_record"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=1,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=1,
    )


def _check(**action_overrides: Any) -> tuple[bool, str]:
    action = _gate_action(**action_overrides)
    return final_report_passes_quality_gate(
        tool_call_count=2,
        action=action,
        slice_obj=_gate_slice(),
        required_output_facts=[],
        inherited_facts={},
    )


def test_quality_gate_passes_for_valid_result() -> None:
    ok, reason = _check()
    assert ok is True
    assert reason == ""


def test_quality_gate_rejects_incomplete_verdict() -> None:
    ok, reason = _check(verdict="INCOMPLETE")
    assert ok is False
    assert "incomplete" in reason


def test_quality_gate_rejects_partial_verdict() -> None:
    ok, reason = _check(verdict="PARTIAL")
    assert ok is False
    assert "partial" in reason


def test_quality_gate_rejects_failed_verdict() -> None:
    ok, reason = _check(verdict="FAILED")
    assert ok is False
    assert "failed" in reason


def test_quality_gate_rejects_low_confidence() -> None:
    ok, reason = _check(confidence=0.3)
    assert ok is False
    assert "confidence" in reason


def test_quality_gate_rejects_zero_tool_calls() -> None:
    action = _gate_action()
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=0,
        action=action,
        slice_obj=_gate_slice(),
        required_output_facts=[],
    )
    assert ok is False
    assert "zero_tool_calls" in reason


def test_quality_gate_rejects_empty_evidence_refs() -> None:
    ok, reason = _check(evidence_refs=[])
    assert ok is False
    assert "evidence" in reason


def test_quality_gate_rejects_missing_required_facts() -> None:
    action = _gate_action(facts={})
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=2,
        action=action,
        slice_obj=_gate_slice(),
        required_output_facts=["research.project_id"],
    )
    assert ok is False
    assert "missing_required_facts" in reason


def test_quality_gate_passes_with_required_facts_present() -> None:
    action = _gate_action(facts={"research.project_id": "proj_1"})
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=2,
        action=action,
        slice_obj=_gate_slice(),
        required_output_facts=["research.project_id"],
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_rejects_empty_verdict_with_zero_confidence() -> None:
    """Empty verdict should NOT be in _FAIL_VERDICTS, but 0 confidence fails."""
    ok, reason = _check(verdict="", confidence=0.0)
    assert ok is False
    assert "confidence" in reason


def test_quality_gate_at_exact_confidence_threshold() -> None:
    """Confidence exactly at the threshold (0.5) should pass."""
    ok, reason = _check(confidence=0.5)
    assert ok is True


def test_quality_gate_case_insensitive_verdict() -> None:
    """Verdict comparison is case-insensitive."""
    ok, reason = _check(verdict="incomplete")
    assert ok is False
    assert "incomplete" in reason
