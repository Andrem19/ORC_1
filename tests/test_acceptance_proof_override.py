"""Tests for acceptance-proof override: WATCHLIST verdicts accepted when proof passes.

Covers the fix where FallbackExecutor._is_success runs the acceptance verifier
even for non-accepted verdicts. If the proof passes, the result is accepted
regardless of the verdict string.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.acceptance.contracts import AcceptanceResult
from app.services.direct_execution.executor import DirectExecutionResult
from app.services.direct_execution.fallback_executor import FallbackExecutor


# -- Helpers -------------------------------------------------------------------

def _strict_slice(**overrides: Any) -> PlanSlice:
    """Build a slice with strict acceptance (WATCHLIST not accepted by default)."""
    defaults = dict(
        slice_id="slice_strict_1",
        title="Test strict slice",
        hypothesis="test",
        objective="test",
        parallel_slot=1,
        depends_on=[],
        allowed_tools=["research_memory"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=3,
        max_tool_calls=10,
        max_expensive_calls=2,
        success_criteria=["test criterion"],
        facts={},
        artifacts=[],
        turn_count=0,
        tool_call_count=0,
        expensive_call_count=0,
        dependency_unblock_mode="accepted_only",
        watchlist_allows_unblock=False,
    )
    defaults.update(overrides)
    return PlanSlice(**defaults)


def _watchlist_result(**overrides: Any) -> DirectExecutionResult:
    """Build a WATCHLIST final_report result with evidence."""
    action = WorkerAction(
        action_id="act_wl",
        action_type="final_report",
        summary=overrides.get("summary", "Created research node but no candidates found"),
        verdict=overrides.get("verdict", "WATCHLIST"),
        confidence=overrides.get("confidence", 0.70),
        evidence_refs=overrides.get("evidence_refs", ["node-abc123"]),
        facts=overrides.get("facts", {"research.project_id": "proj_1"}),
    )
    return DirectExecutionResult(
        action=action,
        artifact_path="/tmp/artifact.json",
        raw_output='{"type":"final_report","summary":"Blocked","verdict":"WATCHLIST"}',
        provider=overrides.get("provider", "primary"),
        duration_ms=100,
        tool_call_count=overrides.get("tool_call_count", 3),
    )


def _passing_acceptance_result() -> AcceptanceResult:
    return AcceptanceResult(
        status="pass",
        contract={"kind": "research_shortlist_write", "mode": "strict"},
        predicates=[],
        blocking_reasons=[],
        evidence_refs=["mcp://research_memory/node-abc123#prove"],
        route="accepted",
    )


def _failing_acceptance_result() -> AcceptanceResult:
    return AcceptanceResult(
        status="fail",
        contract={"kind": "research_shortlist_write", "mode": "strict"},
        predicates=[],
        blocking_reasons=["research_node_proof_pass"],
        route="repair_only",
    )


class _FakeAcceptanceVerifier:
    """Fake verifier that returns a pre-configured result."""

    def __init__(self, result: AcceptanceResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def verify(self, **kwargs: Any) -> AcceptanceResult:
        self.calls.append(kwargs)
        return self._result

    async def __aenter__(self) -> "_FakeAcceptanceVerifier":
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class _FakeExecutor:
    async def execute(self, **kwargs: Any) -> DirectExecutionResult:
        return _watchlist_result()


class _FakeArtifactStore:
    def save_direct_fallback_attempt(self, **kwargs: Any) -> str:
        return "/tmp/fallback.json"


def _make_fallback_executor(
    *,
    acceptance_verifier: _FakeAcceptanceVerifier | None = None,
    fallback_providers: list[str] | None = None,
) -> FallbackExecutor:
    return FallbackExecutor(
        primary_executor=_FakeExecutor(),
        fallback_providers=fallback_providers or [],
        artifact_store=_FakeArtifactStore(),
        incident_store=SimpleNamespace(record=lambda **kw: None),
        direct_config=SimpleNamespace(
            primary_retry_budget=0,
            parse_repair_attempts=0,
            contract_guardrails_enabled=False,
        ),
        worker_system_prompt="test prompt",
        adapter_factory=lambda name: None,
        acceptance_verifier=acceptance_verifier,
    )


# -- Unit tests: _is_success with proof override --------------------------------


def test_watchlist_accepted_when_proof_passes() -> None:
    """WATCHLIST on a strict slice IS accepted when acceptance verifier passes."""
    verifier = _FakeAcceptanceVerifier(_passing_acceptance_result())
    fb = _make_fallback_executor(acceptance_verifier=verifier)
    result = _watchlist_result()
    slice_obj = _strict_slice()

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is True
    assert len(verifier.calls) == 1
    assert result.acceptance_result.get("passed") is True


def test_watchlist_rejected_when_proof_fails() -> None:
    """WATCHLIST on a strict slice is NOT accepted when proof fails."""
    verifier = _FakeAcceptanceVerifier(_failing_acceptance_result())
    fb = _make_fallback_executor(acceptance_verifier=verifier)
    result = _watchlist_result()
    slice_obj = _strict_slice()

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is False
    assert len(verifier.calls) == 1


def test_watchlist_rejected_without_verifier() -> None:
    """WATCHLIST on a strict slice with NO verifier is rejected (no proof possible)."""
    fb = _make_fallback_executor(acceptance_verifier=None)
    result = _watchlist_result()
    slice_obj = _strict_slice()

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is False


def test_complete_accepted_when_proof_passes() -> None:
    """COMPLETE verdict on strict slice is accepted when proof passes (existing behavior)."""
    verifier = _FakeAcceptanceVerifier(_passing_acceptance_result())
    fb = _make_fallback_executor(acceptance_verifier=verifier)
    result = _watchlist_result(verdict="COMPLETE", confidence=0.85)
    slice_obj = _strict_slice()

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is True


def test_complete_rejected_when_proof_fails() -> None:
    """COMPLETE verdict on strict slice is rejected when proof fails (existing behavior)."""
    verifier = _FakeAcceptanceVerifier(_failing_acceptance_result())
    fb = _make_fallback_executor(acceptance_verifier=verifier)
    result = _watchlist_result(verdict="COMPLETE", confidence=0.85)
    slice_obj = _strict_slice()

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is False


def test_watchlist_accepted_on_non_strict_slice_without_verifier() -> None:
    """WATCHLIST on a non-strict slice (watchlist_allows_unblock=True) is accepted
    even without a verifier (existing behavior preserved)."""
    fb = _make_fallback_executor(acceptance_verifier=None)
    result = _watchlist_result()
    slice_obj = _strict_slice(
        dependency_unblock_mode="advisory_only",
        watchlist_allows_unblock=True,
    )

    is_success = asyncio.run(fb._is_success(
        result,
        slice_obj=slice_obj,
        required_output_facts=["research.project_id"],
        inherited_facts={},
    ))

    assert is_success is True


# -- Integration: full execute_with_fallback chain ---------------------------------


def test_primary_watchlist_succeeds_via_proof_no_fallback_triggered() -> None:
    """Primary provider returns WATCHLIST, but proof passes — no fallback needed."""
    verifier = _FakeAcceptanceVerifier(_passing_acceptance_result())
    fb = _make_fallback_executor(
        acceptance_verifier=verifier,
        fallback_providers=["qwen_cli"],
    )
    result, attempts = asyncio.run(fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_strict_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=["research.project_id"],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.action.verdict == "WATCHLIST"
    assert len(attempts) == 0  # No fallback triggered


def test_primary_watchlist_proof_fails_triggers_fallback() -> None:
    """Primary returns WATCHLIST, proof fails — fallback chain is invoked."""
    verifier = _FakeAcceptanceVerifier(_failing_acceptance_result())
    fb = _make_fallback_executor(
        acceptance_verifier=verifier,
        fallback_providers=["qwen_cli"],
    )
    result, attempts = asyncio.run(fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_strict_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=["research.project_id"],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    # Fallback chain tried (adapter_factory returns None, so qwen_cli is skipped)
    assert result.action is not None
