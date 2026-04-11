"""Tests for FallbackExecutor fallback chain logic."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from app.execution_models import PlanSlice, WorkerAction
from app.services.direct_execution.executor import DirectExecutionResult
from app.services.direct_execution.fallback_executor import FallbackAttempt, FallbackExecutor


def _make_slice(**overrides: Any) -> PlanSlice:
    defaults = dict(
        slice_id="slice_1",
        title="Test slice",
        hypothesis="test hypothesis",
        objective="test objective",
        parallel_slot=1,
        depends_on=[],
        allowed_tools=["features_catalog"],
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
    )
    defaults.update(overrides)
    return PlanSlice(**defaults)


def _success_result(provider: str = "primary") -> DirectExecutionResult:
    return DirectExecutionResult(
        action=WorkerAction(
            action_id="act_1",
            action_type="final_report",
            summary="Done",
            verdict="SUCCESS",
            evidence_refs=["node_1"],
            facts={"research.project_id": "proj_1"},
            confidence=0.8,
        ),
        artifact_path="/tmp/artifact.json",
        raw_output='{"type":"final_report","summary":"Done"}',
        provider=provider,
        duration_ms=100,
        tool_call_count=2,
    )


def _fail_result(provider: str = "primary", error: str = "timeout") -> DirectExecutionResult:
    return DirectExecutionResult(
        action=None,
        artifact_path="/tmp/artifact.json",
        raw_output="",
        error=error,
        provider=provider,
        duration_ms=50,
    )


def _blocked_checkpoint_result(provider: str = "primary", reason: str = "direct_tool_budget_exhausted") -> DirectExecutionResult:
    """Result with a valid but blocked checkpoint action — soft failure."""
    return DirectExecutionResult(
        action=WorkerAction(
            action_id="act_blocked",
            action_type="checkpoint",
            status="blocked",
            summary=f"Slice blocked: {reason}",
            reason_code=reason,
        ),
        artifact_path="/tmp/artifact.json",
        raw_output=f'{{"type":"checkpoint","status":"blocked","reason_code":"{reason}"}}',
        provider=provider,
        duration_ms=50,
        error=reason,
    )


def _partial_checkpoint_result(provider: str = "primary", summary: str = "need one more step") -> DirectExecutionResult:
    return DirectExecutionResult(
        action=WorkerAction(
            action_id="act_partial",
            action_type="checkpoint",
            status="partial",
            summary=summary,
        ),
        artifact_path="/tmp/artifact.json",
        raw_output=f'{{"type":"checkpoint","status":"partial","summary":"{summary}"}}',
        provider=provider,
        duration_ms=50,
    )


def _complete_checkpoint_result(provider: str = "primary") -> DirectExecutionResult:
    return DirectExecutionResult(
        action=WorkerAction(
            action_id="act_complete",
            action_type="checkpoint",
            status="complete",
            summary="Evidence is complete",
            facts={"research.project_id": "proj_1"},
            evidence_refs=["node_1"],
        ),
        artifact_path="/tmp/artifact.json",
        raw_output='{"type":"checkpoint","status":"complete","summary":"Evidence is complete"}',
        provider=provider,
        duration_ms=50,
    )


class _FakeExecutor:
    """Fake executor that returns preconfigured results in sequence."""

    def __init__(self, results: list[DirectExecutionResult]) -> None:
        self.results = list(results)
        self.calls: list[dict[str, Any]] = []
        self._idx = 0

    async def execute(self, *, extra_prompt_section: str = "", **kwargs: Any) -> DirectExecutionResult:
        self.calls.append({"kwargs": kwargs, "extra_prompt_section": extra_prompt_section})
        if self._idx < len(self.results):
            result = self.results[self._idx]
            self._idx += 1
            return result
        return _fail_result()


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.saved: list[dict[str, Any]] = []

    def save_direct_fallback_attempt(self, *, plan_id: str, slice_id: str, payload: dict[str, Any]) -> str:
        self.saved.append(payload)
        return f"/tmp/fallback_{len(self.saved)}.json"


class _FakeAdapter:
    def __init__(self, name: str = "fake") -> None:
        self._name = name

    def name(self) -> str:
        return self._name


class _ChainTest:
    """Helper that sets up FallbackExecutor with fully-faked inner executors."""

    def __init__(
        self,
        *,
        fallback_providers: list[str],
        primary_results: list[DirectExecutionResult],
        fallback_results_map: dict[str, list[DirectExecutionResult]] | None = None,
        adapter_map: dict[str, _FakeAdapter | None] | None = None,
    ) -> None:
        self.primary_executor = _FakeExecutor(primary_results)
        self.artifact_store = _FakeArtifactStore()
        self.fallback_executors: dict[str, _FakeExecutor] = {}
        self._fallback_results_map = fallback_results_map or {}
        self._adapter_map = adapter_map or {}

        def factory(provider_name: str):
            return self._adapter_map.get(provider_name)

        self.fb = FallbackExecutor(
            primary_executor=self.primary_executor,
            fallback_providers=fallback_providers,
            artifact_store=self.artifact_store,
            incident_store=SimpleNamespace(record=lambda **kw: None),
            direct_config=SimpleNamespace(),
            worker_system_prompt="test prompt",
            adapter_factory=factory,
        )
        # Monkey-patch _make_fallback_executor to return our fakes
        self.fb._make_fallback_executor = self._make_fake

    def _make_fake(self, adapter: _FakeAdapter) -> _FakeExecutor:
        results = self._fallback_results_map.get(adapter.name(), [_fail_result(provider=adapter.name())])
        ex = _FakeExecutor(results)
        self.fallback_executors[adapter.name()] = ex
        return ex


def test_primary_succeeds_no_fallback() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_success_result()],
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.action.action_type == "final_report"
    assert len(attempts) == 0
    assert len(ct.primary_executor.calls) == 1


def test_fallback_1_succeeds_after_primary_fails() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_fail_result(error="timeout")],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.provider == "qwen_cli"
    assert result.fallback_provider_index == 1
    assert len(attempts) == 1
    assert attempts[0].provider == "qwen_cli"
    assert attempts[0].attempt_index == 1
    assert len(ct.artifact_store.saved) == 1
    assert ct.artifact_store.saved[0]["raw_output_excerpt"].startswith("{")
    assert ct.artifact_store.saved[0]["adapter_name"] == "qwen_worker_cli"
    assert ct.artifact_store.saved[0]["terminal_action_type"] == "final_report"


def test_fallback_2_succeeds_after_both_fail() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli", "claude_cli"],
        primary_results=[_fail_result(error="timeout")],
        fallback_results_map={
            "qwen_worker_cli": [_fail_result(provider="qwen_cli", error="parse_error")],
            "claude_worker_cli": [_success_result(provider="claude_cli")],
        },
        adapter_map={
            "qwen_cli": _FakeAdapter("qwen_worker_cli"),
            "claude_cli": _FakeAdapter("claude_worker_cli"),
        },
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.provider == "claude_cli"
    assert result.fallback_provider_index == 2
    assert len(attempts) == 2
    assert attempts[0].provider == "qwen_cli"
    assert attempts[1].provider == "claude_cli"
    assert len(ct.artifact_store.saved) == 2


def test_all_providers_fail() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli", "claude_cli"],
        primary_results=[_fail_result(error="timeout")],
        fallback_results_map={
            "qwen_worker_cli": [_fail_result(provider="qwen_cli", error="parse_error")],
            "claude_worker_cli": [_fail_result(provider="claude_cli", error="connection_error")],
        },
        adapter_map={
            "qwen_cli": _FakeAdapter("qwen_worker_cli"),
            "claude_cli": _FakeAdapter("claude_worker_cli"),
        },
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is None
    assert result.provider == "claude_cli"
    assert result.fallback_provider_index == 2
    assert len(attempts) == 2
    assert result.error == "connection_error"


def test_no_fallback_providers_configured() -> None:
    ct = _ChainTest(
        fallback_providers=[],
        primary_results=[_fail_result()],
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is None
    assert len(attempts) == 0


def test_fallback_receives_error_context_in_prompt() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_fail_result(error="model_stalled")],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    # The fallback executor call should have extra_prompt_section with error context
    qwen_ex = ct.fallback_executors["qwen_worker_cli"]
    assert len(qwen_ex.calls) == 1
    extra = qwen_ex.calls[0]["extra_prompt_section"]
    assert "model_stalled" in extra
    assert "Fallback Context" in extra


def test_adapter_factory_returns_none_skips_fallback() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_fail_result()],
        adapter_map={"qwen_cli": None},  # adapter not available
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is None
    assert len(attempts) == 0  # skipped because adapter factory returned None


def test_empty_fallback_provider_strings_are_filtered() -> None:
    ct = _ChainTest(
        fallback_providers=["", "  ", "qwen_cli"],
        primary_results=[_fail_result()],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.provider == "qwen_cli"
    assert len(attempts) == 1


def test_fallback_triggers_on_blocked_checkpoint() -> None:
    """Budget exhausted returns a valid checkpoint action (not None) — fallback must still trigger."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_blocked_checkpoint_result(reason="direct_tool_budget_exhausted")],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    # Fallback triggered despite primary having a non-None action
    assert len(attempts) == 1
    assert result.action is not None
    assert result.action.action_type == "final_report"
    assert result.provider == "qwen_cli"
    assert result.fallback_provider_index == 1


def test_fallback_triggers_on_stalled_watchdog_checkpoint() -> None:
    """Model stalled returns a valid watchdog checkpoint — fallback must trigger."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli", "claude_cli"],
        primary_results=[_blocked_checkpoint_result(reason="direct_model_stalled_before_first_action")],
        fallback_results_map={
            "qwen_worker_cli": [_blocked_checkpoint_result(provider="qwen_cli", reason="direct_model_stalled_between_actions")],
            "claude_worker_cli": [_success_result(provider="claude_cli")],
        },
        adapter_map={
            "qwen_cli": _FakeAdapter("qwen_worker_cli"),
            "claude_cli": _FakeAdapter("claude_worker_cli"),
        },
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    # Both primary and fallback_1 stalled — fallback_2 succeeded
    assert len(attempts) == 2
    assert result.action is not None
    assert result.action.action_type == "final_report"
    assert result.provider == "claude_cli"


def test_checkpoint_complete_counts_as_success_without_claude() -> None:
    ct = _ChainTest(
        fallback_providers=["claude_cli"],
        primary_results=[_complete_checkpoint_result()],
        adapter_map={"claude_cli": _FakeAdapter("claude_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=["research.project_id"],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.action.action_type == "checkpoint"
    assert result.action.status == "complete"
    assert attempts == []


def test_checkpoint_partial_still_falls_back() -> None:
    ct = _ChainTest(
        fallback_providers=["claude_cli"],
        primary_results=[_partial_checkpoint_result()],
        fallback_results_map={"claude_worker_cli": [_success_result(provider="claude_cli")]},
        adapter_map={"claude_cli": _FakeAdapter("claude_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=["research.project_id"],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.provider == "claude_cli"
    assert len(attempts) == 1


def test_tool_not_in_allowlist_gets_one_repair_before_next_provider() -> None:
    ct = _ChainTest(
        fallback_providers=["qwen_cli", "claude_cli"],
        primary_results=[_fail_result(error="timeout")],
        fallback_results_map={
            "qwen_worker_cli": [
                _fail_result(
                    provider="qwen_cli",
                    error="direct_output_parse_failed:tool_not_in_allowlist:backtests_strategy",
                ),
                _success_result(provider="qwen_cli"),
            ],
            "claude_worker_cli": [_success_result(provider="claude_cli")],
        },
        adapter_map={
            "qwen_cli": _FakeAdapter("qwen_worker_cli"),
            "claude_cli": _FakeAdapter("claude_worker_cli"),
        },
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(allowed_tools=["backtests_runs"]),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    qwen_ex = ct.fallback_executors["qwen_worker_cli"]
    assert result.provider == "qwen_cli"
    assert len(attempts) == 1
    assert len(qwen_ex.calls) == 2
    assert "Contract Repair" in qwen_ex.calls[1]["extra_prompt_section"]
    assert "backtests_runs" in qwen_ex.calls[1]["extra_prompt_section"]
    assert "backtests_strategy" not in qwen_ex.calls[1]["extra_prompt_section"]


# ---------------------------------------------------------------------------
# Quality gate tests – ensure low-quality final_reports trigger fallback
# ---------------------------------------------------------------------------


def _low_quality_result(
    *,
    provider: str = "primary",
    verdict: str = "SUCCESS",
    confidence: float = 0.8,
    tool_call_count: int = 2,
    evidence_refs: list[str] | None = None,
    facts: dict[str, Any] | None = None,
) -> DirectExecutionResult:
    """Build a final_report result with individually controllable quality knobs."""
    return DirectExecutionResult(
        action=WorkerAction(
            action_id="act_lq",
            action_type="final_report",
            summary="Low quality result",
            verdict=verdict,
            confidence=confidence,
            evidence_refs=evidence_refs if evidence_refs is not None else ["node_1"],
            facts=facts if facts is not None else {"research.project_id": "proj_1"},
        ),
        artifact_path="/tmp/artifact.json",
        raw_output='{"type":"final_report","summary":"Low quality"}',
        provider=provider,
        duration_ms=100,
        tool_call_count=tool_call_count,
    )


def test_zero_tool_calls_triggers_fallback() -> None:
    """final_report with 0 tool calls is a hallucinated result — must fall back."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(tool_call_count=0)],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1, "Zero tool calls should trigger fallback"
    assert result.provider == "qwen_cli"
    assert result.tool_call_count == 2  # fallback result


def test_incomplete_verdict_triggers_fallback() -> None:
    """final_report with verdict=INCOMPLETE should fall back."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(verdict="INCOMPLETE")],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1
    assert result.provider == "qwen_cli"


def test_low_confidence_triggers_fallback() -> None:
    """final_report with confidence 0.4 (< 0.5 threshold) should fall back."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(confidence=0.4)],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1
    assert result.provider == "qwen_cli"


def test_empty_evidence_refs_triggers_fallback() -> None:
    """final_report with no evidence_refs should fall back."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(evidence_refs=[])],
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1
    assert result.provider == "qwen_cli"


def test_missing_required_facts_triggers_fallback() -> None:
    """final_report missing a required_output_fact should fall back."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(facts={})],  # missing research.project_id
        fallback_results_map={"qwen_worker_cli": [_success_result(provider="qwen_cli")]},
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=["research.project_id"],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1
    assert result.provider == "qwen_cli"


def test_high_quality_final_report_passes_gate() -> None:
    """A final_report with good verdict, confidence, tools, evidence, and facts should pass."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_success_result()],
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert result.action is not None
    assert result.action.action_type == "final_report"
    assert len(attempts) == 0, "High quality result should NOT trigger fallback"


def test_all_quality_failures_exhaust_fallbacks() -> None:
    """When every provider returns low-quality final_reports, the last result is returned."""
    ct = _ChainTest(
        fallback_providers=["qwen_cli"],
        primary_results=[_low_quality_result(tool_call_count=0)],
        fallback_results_map={
            "qwen_worker_cli": [_low_quality_result(provider="qwen_cli", confidence=0.3)],
        },
        adapter_map={"qwen_cli": _FakeAdapter("qwen_worker_cli")},
    )

    result, attempts = asyncio.run(ct.fb.execute_with_fallback(
        plan_id="plan_1",
        slice_obj=_make_slice(),
        baseline_bootstrap={},
        known_facts={},
        required_output_facts=[],
        recent_turn_summaries=[],
        checkpoint_summary="",
    ))

    assert len(attempts) == 1
    # Last fallback result is returned even though it failed the quality gate
    assert result.action is not None
    assert result.action.confidence == 0.3
