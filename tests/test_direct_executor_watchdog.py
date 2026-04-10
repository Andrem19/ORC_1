from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice
from app.services.direct_execution.executor import DirectSliceExecutor


class _NeverFinishingLoop:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs

    async def invoke(self, **kwargs):  # type: ignore[no-untyped-def]
        on_progress = kwargs.get("on_progress")
        if callable(on_progress):
            on_progress("tool_result", {"tool_name": "features_catalog"})
        await asyncio.Event().wait()


def _slice() -> PlanSlice:
    return PlanSlice(
        slice_id="slice_1",
        title="slice",
        hypothesis="h",
        objective="o",
        success_criteria=["c1"],
        allowed_tools=["features_catalog"],
        evidence_requirements=["e1"],
        policy_tags=[],
        max_turns=2,
        max_tool_calls=24,
        max_expensive_calls=6,
        parallel_slot=1,
    )


def test_watchdog_uses_remaining_time_budget(monkeypatch, tmp_path) -> None:
    adapter = LmStudioWorkerApi(base_url="http://localhost:1234", model="qwen/qwen3.5-9b")
    artifact_store = ExecutionArtifactStore(tmp_path)
    direct_cfg = SimpleNamespace(
        provider="lmstudio",
        timeout_seconds=600,
        max_attempts_per_slice=1,
        max_tool_calls_per_slice=24,
        max_expensive_tool_calls_per_slice=6,
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
        direct_config=direct_cfg,
        worker_system_prompt="",
    )

    timeouts: list[float] = []
    async def fake_wait_for(awaitable, timeout):  # type: ignore[no-untyped-def]
        del awaitable
        timeouts.append(float(timeout))
        await asyncio.sleep(0)
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.services.direct_execution.executor.LmStudioToolLoop", _NeverFinishingLoop)
    monkeypatch.setattr("app.services.direct_execution.executor.asyncio.wait_for", fake_wait_for)

    result = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=_slice(),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
        )
    )

    assert result.action is not None
    assert result.action.action_type == "checkpoint"
    assert result.action.reason_code == "direct_model_stalled_between_actions"
    assert len(timeouts) >= 2
    assert timeouts[0] > timeouts[-1]
