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
        ),
        artifact_path="/tmp/artifact.json",
        raw_output='{"type":"final_report","summary":"Done"}',
        provider=provider,
        duration_ms=100,
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
