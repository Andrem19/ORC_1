from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.adapters.base import AdapterResponse, BaseAdapter
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.executor import DirectSliceExecutor
from app.services.direct_execution.guardrails import final_report_passes_quality_gate, synthesize_transcript_evidence_refs


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


def _gate_slice(**overrides: Any) -> PlanSlice:
    defaults = dict(
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
    defaults.update(overrides)
    return PlanSlice(
        **defaults,
    )


def _check(*, slice_overrides: dict[str, Any] | None = None, tool_call_count: int = 2, **action_overrides: Any) -> tuple[bool, str]:
    action = _gate_action(**action_overrides)
    return final_report_passes_quality_gate(
        tool_call_count=tool_call_count,
        action=action,
        slice_obj=_gate_slice(**(slice_overrides or {})),
        required_output_facts=[],
        inherited_facts={},
    )


def test_quality_gate_passes_for_valid_result() -> None:
    ok, reason = _check(
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["node_test_1"],
        }
    )
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
    action = _gate_action(
        facts={
            "research.project_id": "proj_1",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["node_test_1"],
        }
    )
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=2,
        action=action,
        slice_obj=_gate_slice(),
        required_output_facts=["research.project_id"],
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_accepts_safe_backtests_integration_aliases_after_canonicalization() -> None:
    action = _gate_action(
        evidence_refs=["analysis-18dfbc658a2a-rm-feature-long"],
        facts={
            "integration_handles": {"feature_long": "20260413-143910-3db43a14"},
            "integration_refs": ["analysis-18dfbc658a2a-rm-feature-long"],
            "direct.provider": "qwen_cli",
            "direct.successful_tool_count": 10,
            "direct.successful_tool_names": ["backtests_runs", "backtests_analysis", "research_memory"],
        },
    )
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=10,
        action=action,
        slice_obj=_gate_slice(
            title="Integration",
            objective="Integrate surviving candidates over the base.",
            allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
            policy_tags=["integration"],
            runtime_profile="backtests_integration_analysis",
        ),
        required_output_facts=["backtests.integration_handles", "backtests.integration_refs"],
    )

    assert ok is True
    assert reason == ""


def test_quality_gate_still_rejects_missing_backtests_integration_facts_without_aliases() -> None:
    action = _gate_action(
        evidence_refs=["analysis-18dfbc658a2a-rm-feature-long"],
        facts={
            "direct.provider": "qwen_cli",
            "direct.successful_tool_count": 10,
            "direct.successful_tool_names": ["backtests_runs", "backtests_analysis"],
        },
    )
    ok, reason = final_report_passes_quality_gate(
        tool_call_count=10,
        action=action,
        slice_obj=_gate_slice(
            title="Integration",
            objective="Integrate surviving candidates over the base.",
            allowed_tools=["backtests_runs", "backtests_analysis", "research_memory"],
            policy_tags=["integration"],
            runtime_profile="backtests_integration_analysis",
        ),
        required_output_facts=["backtests.integration_handles", "backtests.integration_refs"],
    )

    assert ok is False
    assert "missing_required_facts" in reason


def test_quality_gate_rejects_transport_only_evidence_refs_after_sanitation() -> None:
    ok, reason = _check(
        evidence_refs=["1", "0f116f24435a4d9ebe19eb11105e183f"],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["1", "0f116f24435a4d9ebe19eb11105e183f"],
        },
    )

    assert ok is False
    assert reason == "empty_evidence_refs"


def test_quality_gate_rejects_empty_verdict_with_zero_confidence() -> None:
    """Empty verdict should NOT be in _FAIL_VERDICTS, but 0 confidence fails."""
    ok, reason = _check(verdict="", confidence=0.0)
    assert ok is False
    assert "confidence" in reason


def test_quality_gate_at_exact_confidence_threshold() -> None:
    """Confidence exactly at the threshold (0.5) should pass."""
    ok, reason = _check(
        confidence=0.5,
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["node_test_1"],
        },
    )
    assert ok is True


def test_quality_gate_case_insensitive_verdict() -> None:
    """Verdict comparison is case-insensitive."""
    ok, reason = _check(verdict="incomplete")
    assert ok is False
    assert "incomplete" in reason


def test_claude_cli_in_repairable_providers() -> None:
    """claude_cli must be in REPAIRABLE_PROVIDER_NAMES for contract repair."""
    from app.services.direct_execution.guardrails import REPAIRABLE_PROVIDER_NAMES

    assert "claude_cli" in REPAIRABLE_PROVIDER_NAMES
    assert "lmstudio" in REPAIRABLE_PROVIDER_NAMES
    assert "qwen_cli" in REPAIRABLE_PROVIDER_NAMES


def test_quality_gate_rejects_unsupported_evidence_refs() -> None:
    ok, reason = _check(
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["transcript:1:research_memory", "node_real_1"],
        }
    )
    assert ok is False
    assert reason == "unsupported_evidence_refs"


def test_quality_gate_accepts_legacy_lmstudio_local_transcript_refs() -> None:
    ok, reason = _check(
        evidence_refs=[
            "transcript:1:research_project_list",
            "transcript:2:research_project_create",
            "transcript:3:research_map_define",
        ],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 3,
            "direct.successful_mutating_tool_count": 2,
            "direct.supported_evidence_refs": [
                "transcript:2:research_project",
                "transcript:4:research_project",
                "transcript:6:research_map",
                "node_real_1",
            ],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_accepts_ordered_tool_sequence_when_transcript_indices_drift() -> None:
    ok, reason = _check(
        evidence_refs=[
            "transcript:5:research_project",
            "transcript:6:research_map",
            "transcript:7:research_memory",
        ],
        facts={
            "research.project_id": "proj_test",
            "research.branch_id": "branch_test",
            "research.baseline_configured": True,
            "research.baseline_snapshot_id": "active-signal-v1",
            "research.baseline_version": 1,
            "research.atlas_defined": True,
            "research.memory_node_id": "node_real_1",
            "research.invariants_recorded": True,
            "research.naming_recorded": True,
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 5,
            "direct.successful_mutating_tool_count": 3,
            "direct.supported_evidence_refs": [
                "transcript:2:research_project",
                "transcript:4:research_project",
                "transcript:6:research_project",
                "transcript:8:research_map",
                "transcript:10:research_memory",
                "node_real_1",
            ],
        },
        slice_overrides={
            "runtime_profile": "research_setup",
            "allowed_tools": ["research_project", "research_map", "research_memory"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_leaves_write_result_semantics_to_acceptance_verifier() -> None:
    ok, reason = _check(
        slice_overrides={
            "runtime_profile": "write_result",
            "success_criteria": ["one", "two"],
            "evidence_requirements": ["one", "two"],
        },
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 0,
            "direct.supported_evidence_refs": ["node_test_1"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_leaves_research_setup_semantics_to_acceptance_verifier() -> None:
    ok, reason = _check(
        slice_overrides={
            "allowed_tools": ["research_project", "research_map", "research_memory"],
        },
        facts={
            "research.project_id": "proj_test",
            "research.branch_id": "branch_1",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.supported_evidence_refs": ["node_test_1"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_leaves_feature_contract_semantics_to_acceptance_verifier() -> None:
    ok, reason = _check(
        slice_overrides={
            "title": "Feature contract stage",
            "objective": "Validate custom feature publication and leakage checks.",
            "allowed_tools": ["research_memory", "features_custom", "features_dataset"],
            "success_criteria": ["Custom features validated and published."],
            "evidence_requirements": ["Leakage checked for each contract."],
            "policy_tags": ["feature_contract"],
        },
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.successful_tool_names": ["research_memory"],
            "direct.supported_evidence_refs": ["node_test_1"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_leaves_backtests_live_validation_to_acceptance_verifier() -> None:
    ok, reason = _check(
        slice_overrides={
            "allowed_tools": ["backtests_plan", "backtests_runs"],
            "requires_persisted_artifact": True,
            "requires_live_handle_validation": True,
        },
        evidence_refs=["20260413-143910-3db43a14"],
        facts={
            "research.project_id": "proj_test",
            "run_id": "20260413-143910-3db43a14",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_tool_names": ["backtests_plan", "backtests_runs"],
            "direct.supported_evidence_refs": [
                "transcript:1:backtests_plan",
                "transcript:2:backtests_runs",
                "20260413-143910-3db43a14",
            ],
            "direct.statuses": ["started"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_leaves_backtests_resource_probe_to_acceptance_verifier() -> None:
    ok, reason = _check(
        slice_overrides={
            "allowed_tools": ["backtests_plan", "backtests_runs"],
            "requires_persisted_artifact": True,
            "requires_live_handle_validation": True,
        },
        evidence_refs=["20260413-143910-3db43a14"],
        facts={
            "research.project_id": "proj_test",
            "run_id": "20260413-143910-3db43a14",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 3,
            "direct.successful_tool_names": ["backtests_plan", "backtests_runs"],
            "direct.supported_evidence_refs": [
                "transcript:1:backtests_plan",
                "transcript:2:backtests_runs",
                "transcript:3:backtests_runs",
                "20260413-143910-3db43a14",
            ],
            "direct.statuses": ["running", "resource_not_found"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_accepts_research_only_domain_evidence() -> None:
    ok, reason = _check(
        slice_overrides={
            "title": "Closure audit stage",
            "objective": "Validate research coverage for the closure audit.",
            "allowed_tools": ["research_project", "research_memory"],
            "success_criteria": ["Coverage verified for the selected project."],
            "evidence_requirements": ["Coverage evidence recorded."],
            "policy_tags": ["closure_audit"],
        },
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "lmstudio",
            "direct.successful_tool_count": 2,
            "direct.successful_mutating_tool_count": 1,
            "direct.successful_tool_names": ["research_memory"],
            "direct.supported_evidence_refs": ["node_test_1"],
        },
    )
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Unit tests for is_prerequisite_block_terminal
# ---------------------------------------------------------------------------


def _prereq_action(**overrides: Any) -> WorkerAction:
    defaults = dict(
        action_id="prereq_test",
        action_type="final_report",
        summary="Test result",
        verdict="REJECT",
        confidence=0.8,
        evidence_refs=[],
        facts={},
    )
    defaults.update(overrides)
    return WorkerAction(**defaults)


def test_is_prerequisite_block_terminal_returns_true_with_all_rejected_marker() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"stage_6.all_rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_returns_true_with_blocks_downstream_marker() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"stage_5.blocks_downstream": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_returns_true_with_no_surviving_candidates_marker() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"backtests.no_surviving_candidates": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_returns_true_with_skipped_by_prerequisite_marker() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="SKIP",
        facts={},
    )
    prereq_facts = {"direct.skipped_by_prerequisite": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_returns_false_without_marker() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"some_other_fact": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is False


def test_is_prerequisite_block_terminal_returns_false_with_empty_facts() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(verdict="REJECT", facts={})
    assert is_prerequisite_block_terminal(action, {}) is False


def test_is_prerequisite_block_terminal_returns_false_with_success_verdict() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="SUCCESS",
        facts={},
    )
    prereq_facts = {"stage_6.all_rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is False


def test_is_prerequisite_block_terminal_returns_false_with_watchlist_verdict() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="WATCHLIST",
        facts={},
    )
    prereq_facts = {"stage_6.all_rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is False


def test_is_prerequisite_block_terminal_returns_false_with_false_marker_value() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"stage_6.all_rejected": False}
    assert is_prerequisite_block_terminal(action, prereq_facts) is False


def test_is_prerequisite_block_terminal_case_insensitive_verdict() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="reject",
        facts={},
    )
    prereq_facts = {"stage_6.all_rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_marker_case_insensitive() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts = {"stage_6.All_Rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True


def test_is_prerequisite_block_terminal_requires_truthy_value() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action1 = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts1 = {"stage_6.all_rejected": "true"}
    assert is_prerequisite_block_terminal(action1, prereq_facts1) is True

    action2 = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts2 = {"stage_6.all_rejected": ""}
    assert is_prerequisite_block_terminal(action2, prereq_facts2) is False

    action3 = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts3 = {"stage_6.all_rejected": None}
    assert is_prerequisite_block_terminal(action3, prereq_facts3) is False


def test_is_prerequisite_block_terminal_uses_prerequisite_facts_param() -> None:
    from app.services.direct_execution.guardrails import is_prerequisite_block_terminal

    action = _prereq_action(verdict="REJECT", facts={})
    prereq_facts = {"stage_6.all_rejected": True}
    assert is_prerequisite_block_terminal(action, prereq_facts) is True

    action2 = _prereq_action(
        verdict="REJECT",
        facts={},
    )
    prereq_facts2 = {"stage_6.blocks_downstream": True}
    assert is_prerequisite_block_terminal(action2, prereq_facts2) is True


# ---------- Fix 2: WATCHLIST confidence boost with tool calls ----------


def test_watchlist_confidence_boosts_with_sufficient_tool_calls() -> None:
    """WATCHLIST + confidence=0.30 + 3 tool calls should pass (boosted to 0.70)."""
    ok, reason = _check(
        verdict="WATCHLIST",
        confidence=0.30,
        tool_call_count=3,
        facts={
            "research.project_id": "proj_test",
            "direct.successful_tool_count": 3,
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is True
    assert reason == ""


def test_watchlist_confidence_rejects_without_tool_calls() -> None:
    """WATCHLIST + confidence=0.30 + 0 tool calls should still reject."""
    ok, reason = _check(
        verdict="WATCHLIST",
        confidence=0.30,
        tool_call_count=0,
        facts={
            "research.project_id": "proj_test",
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is False
    assert "confidence" in reason


def test_watchlist_confidence_boosts_at_exact_tool_call_threshold() -> None:
    """WATCHLIST + confidence=0.30 + exactly 3 tool calls should pass."""
    ok, reason = _check(
        verdict="WATCHLIST",
        confidence=0.30,
        tool_call_count=3,
        facts={
            "research.project_id": "proj_test",
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is True


def test_watchlist_confidence_rejects_with_2_tool_calls() -> None:
    """WATCHLIST + confidence=0.30 + 2 tool calls (below threshold) should reject."""
    ok, reason = _check(
        verdict="WATCHLIST",
        confidence=0.30,
        tool_call_count=2,
        facts={
            "research.project_id": "proj_test",
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is False
    assert "confidence" in reason


def test_watchlist_salvage_still_rejected_despite_tool_calls() -> None:
    """Salvage WATCHLIST with many tool calls must still be rejected."""
    ok, reason = _check(
        verdict="WATCHLIST",
        confidence=0.30,
        tool_call_count=14,
        facts={
            "research.project_id": "proj_test",
            "direct.auto_finalized_from_budget_salvage": True,
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is False
    assert "auto_salvage" in reason


def test_complete_verdict_not_affected_by_confidence_boost() -> None:
    """COMPLETE + confidence=0.30 + 3 tool calls should pass via regular floor boost."""
    ok, reason = _check(
        verdict="COMPLETE",
        confidence=0.30,
        tool_call_count=3,
        facts={
            "research.project_id": "proj_test",
            "direct.supported_evidence_refs": ["transcript:1:research_memory"],
        },
        evidence_refs=["transcript:1:research_memory"],
    )
    assert ok is True


# ---------- Fix 1: synthesize_transcript_evidence_refs ----------


def test_synthesize_refs_from_successful_tool_calls() -> None:
    transcript = [
        {"kind": "tool_result", "tool": "research_project", "payload": {}},
        {"kind": "tool_result", "tool": "research_memory", "payload": {}},
        {"kind": "tool_result", "tool": "research_map", "payload": {}},
    ]
    refs = synthesize_transcript_evidence_refs(transcript)
    assert len(refs) == 3
    assert refs[0] == "transcript:1:research_project"
    assert refs[1] == "transcript:2:research_memory"
    assert refs[2] == "transcript:3:research_map"


def test_synthesize_refs_skips_error_entries() -> None:
    transcript = [
        {"kind": "tool_result", "tool": "research_project", "payload": {}},
        {"kind": "tool_result", "tool": "research_memory", "payload": {"error": "fail"}},
        {"kind": "tool_result", "tool": "research_map", "payload": {}},
    ]
    refs = synthesize_transcript_evidence_refs(transcript)
    assert len(refs) == 2
    assert "transcript:2:research_memory" not in refs


def test_synthesize_refs_skips_non_tool_result_entries() -> None:
    transcript = [
        {"kind": "assistant_message", "content": "hello"},
        {"kind": "tool_result", "tool": "research_project", "payload": {}},
        {"kind": "tool_call", "tool": "research_memory"},
    ]
    refs = synthesize_transcript_evidence_refs(transcript)
    assert len(refs) == 1
    assert refs[0] == "transcript:2:research_project"


def test_synthesize_refs_returns_empty_for_empty_transcript() -> None:
    refs = synthesize_transcript_evidence_refs([])
    assert refs == []


def test_synthesize_refs_respects_limit() -> None:
    transcript = [
        {"kind": "tool_result", "tool": f"tool_{i}", "payload": {}}
        for i in range(10)
    ]
    refs = synthesize_transcript_evidence_refs(transcript, limit=3)
    assert len(refs) == 3


# ---------- Evidence ref rescue for fabricated refs ----------


def test_quality_gate_rescues_fabricated_refs_with_sufficient_tool_calls() -> None:
    """When model made >=3 tool calls but fabricated evidence_refs, rescue with real refs."""
    ok, reason = _check(
        tool_call_count=5,
        evidence_refs=["backtests_plan-response-empty", "backtests_runs-start-blocked"],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "minimax",
            "direct.successful_tool_count": 5,
            "direct.successful_mutating_tool_count": 0,
            "direct.supported_evidence_refs": [
                "transcript:1:backtests_plan",
                "transcript:3:backtests_runs",
                "active-signal-v1@1",
            ],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_rescue_requires_minimum_tool_calls() -> None:
    """Rescue does NOT apply when tool_call_count < 3 (below floor)."""
    ok, reason = _check(
        tool_call_count=2,
        evidence_refs=["backtests_plan-response-empty"],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "minimax",
            "direct.successful_tool_count": 2,
            "direct.supported_evidence_refs": ["transcript:1:backtests_plan"],
        },
    )
    assert ok is False
    assert reason == "unsupported_evidence_refs"


def test_quality_gate_rescue_at_exact_floor() -> None:
    """Rescue applies at exactly 3 tool calls (the floor)."""
    ok, reason = _check(
        tool_call_count=3,
        evidence_refs=["fabricated-ref-1"],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "minimax",
            "direct.successful_tool_count": 3,
            "direct.supported_evidence_refs": [
                "transcript:1:backtests_plan",
                "transcript:2:backtests_runs",
            ],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_no_rescue_needed_when_refs_are_real() -> None:
    """When model provides real evidence_refs, rescue is not needed (existing behavior)."""
    ok, reason = _check(
        tool_call_count=5,
        evidence_refs=["active-signal-v1@1"],
        facts={
            "research.project_id": "proj_test",
            "direct.provider": "minimax",
            "direct.successful_tool_count": 5,
            "direct.supported_evidence_refs": ["transcript:1:backtests_plan", "active-signal-v1@1"],
        },
    )
    assert ok is True
    assert reason == ""


def test_quality_gate_rescue_exactly_matches_incident_pattern() -> None:
    """Reproduce the exact stage_5 incident: 5 tool calls, fabricated refs, supported refs available."""
    ok, reason = _check(
        tool_call_count=5,
        evidence_refs=[
            "compiled_plan_v1_stage_4.shortlist_families",
            "backtests_plan-response-empty",
            "backtests_runs-start-response-empty",
        ],
        facts={
            "shortlist_families_from_previous_stage": ["rsi_1"],
            "baseline_snapshot": "active-signal-v1@1",
            "direct.provider": "minimax",
            "direct.successful_tool_count": 5,
            "direct.successful_mutating_tool_count": 0,
            "direct.supported_evidence_refs": [
                "transcript:1:backtests_plan",
                "transcript:3:backtests_runs",
                "transcript:5:backtests_strategy",
                "transcript:9:research_memory",
                "active-signal-v1@1",
            ],
        },
    )
    assert ok is True
    assert reason == ""

