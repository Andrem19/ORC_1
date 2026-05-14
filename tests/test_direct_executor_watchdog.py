from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.adapters.base import AdapterResponse
from app.adapters.base import BaseAdapter
from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice
from app.services.direct_execution.executor import DirectSliceExecutor
from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoopResult
from tests.mcp_catalog_fixtures import make_catalog_snapshot


class _NeverFinishingLoop:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs

    async def invoke(self, **kwargs):  # type: ignore[no-untyped-def]
        on_progress = kwargs.get("on_progress")
        if callable(on_progress):
            on_progress("tool_result", {"tool_name": "features_catalog"})
        await asyncio.Event().wait()


class _NeverFinishingLoopWithTranscript:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs

    async def invoke(self, **kwargs):  # type: ignore[no-untyped-def]
        on_progress = kwargs.get("on_progress")
        if callable(on_progress):
            on_progress(
                "tool_result",
                {
                    "tool_name": "research_map",
                    "tool_call_count": 3,
                    "expensive_tool_call_count": 1,
                    "transcript_len": 2,
                    "transcript": [
                        {"kind": "tool_result", "tool": "research_map"},
                        {"kind": "assistant_response", "payload": {"content": "thinking"}},
                    ],
                },
            )
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
            slice_obj=PlanSlice(
                slice_id="slice_progress",
                title="slice",
                hypothesis="h",
                objective="o",
                success_criteria=["c1"],
                allowed_tools=["research_map"],
                evidence_requirements=["e1"],
                policy_tags=[],
                max_turns=2,
                max_tool_calls=24,
                max_expensive_calls=6,
                parallel_slot=1,
            ),
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


class _ArtifactRecorder:
    def __init__(self) -> None:
        self.saved: list[dict[str, object]] = []

    def save_direct_attempt(self, *, plan_id: str, slice_id: str, payload: dict[str, object]) -> str:
        self.saved.append({"plan_id": plan_id, "slice_id": slice_id, "payload": payload})
        return f"/tmp/{plan_id}_{slice_id}_{len(self.saved)}.json"


class _ProgressLoop:
    def __init__(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs

    async def invoke(self, **kwargs):  # type: ignore[no-untyped-def]
        on_progress = kwargs.get("on_progress")
        if callable(on_progress):
            on_progress(
                "tool_result",
                {
                    "tool_name": "research_map",
                    "tool_call_count": 3,
                    "expensive_tool_call_count": 1,
                    "transcript_len": 7,
                },
            )
        return LmStudioToolLoopResult(
            response=AdapterResponse(
                success=True,
                raw_output='```json\n{"type":"final_report","summary":"ok","verdict":"COMPLETE","findings":[],"facts":{"research.project_id":"proj_1"},"evidence_refs":["transcript:1:research_map"],"confidence":0.8}\n```',
                duration_seconds=0.01,
            ),
            transcript=[{"kind": "tool_result", "tool": "research_map"}],
            tool_call_count=3,
            expensive_tool_call_count=1,
        )


def test_executor_heartbeat_persists_real_progress_metadata(monkeypatch) -> None:
    adapter = LmStudioWorkerApi(base_url="http://localhost:1234", model="qwen/qwen3.5-9b")
    artifact_store = _ArtifactRecorder()
    direct_cfg = SimpleNamespace(
        provider="lmstudio",
        timeout_seconds=30,
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
        artifact_store=artifact_store,  # type: ignore[arg-type]
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=direct_cfg,
        worker_system_prompt="",
        catalog_snapshot=make_catalog_snapshot(),
    )

    monkeypatch.setattr("app.services.direct_execution.executor.LmStudioToolLoop", _ProgressLoop)

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
    heartbeat_payload = next(
        item["payload"]
        for item in artifact_store.saved
        if item["payload"].get("status") == "in_progress" and item["payload"].get("heartbeat")
    )
    assert heartbeat_payload["tool_call_count"] == 3
    assert heartbeat_payload["expensive_tool_call_count"] == 1
    assert heartbeat_payload["transcript_len"] == 7
    assert heartbeat_payload["last_tool_name"] == "research_map"
    assert heartbeat_payload["last_progress_kind"] == "tool_result"
    assert heartbeat_payload["response"] == {"last_progress_kind": "tool_result"}
    assert heartbeat_payload["transcript"] == [{"kind": "heartbeat", "transcript_len": 7}]


def test_stalled_watchdog_preserves_transcript_progress_for_salvage(monkeypatch) -> None:
    adapter = LmStudioWorkerApi(base_url="http://localhost:1234", model="qwen/qwen3.5-9b")
    artifact_store = _ArtifactRecorder()
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
        artifact_store=artifact_store,  # type: ignore[arg-type]
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=direct_cfg,
        worker_system_prompt="",
        catalog_snapshot=make_catalog_snapshot(),
    )

    timeouts: list[float] = []

    async def fake_wait_for(awaitable, timeout):  # type: ignore[no-untyped-def]
        del awaitable
        timeouts.append(float(timeout))
        await asyncio.sleep(0)
        raise asyncio.TimeoutError

    monkeypatch.setattr("app.services.direct_execution.executor.LmStudioToolLoop", _NeverFinishingLoopWithTranscript)
    monkeypatch.setattr("app.services.direct_execution.executor.asyncio.wait_for", fake_wait_for)

    result = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=PlanSlice(
                slice_id="slice_progress",
                title="slice",
                hypothesis="h",
                objective="o",
                success_criteria=["c1"],
                allowed_tools=["research_map"],
                evidence_requirements=["e1"],
                policy_tags=[],
                max_turns=2,
                max_tool_calls=24,
                max_expensive_calls=6,
                parallel_slot=1,
            ),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
        )
    )

    assert result.tool_call_count == 3
    assert result.expensive_tool_call_count == 1
    completed_payload = next(item["payload"] for item in artifact_store.saved if item["payload"].get("status") == "completed")
    assert completed_payload["tool_call_count"] == 3
    assert completed_payload["expensive_tool_call_count"] == 1
    assert completed_payload["transcript"] == [
        {"kind": "tool_result", "tool": "research_map"},
        {"kind": "assistant_response", "payload": {"content": "thinking"}},
    ]
    assert len(timeouts) >= 2


class _StaticAdapter(BaseAdapter):
    def invoke(self, prompt: str, timeout: int = 120, **kwargs):  # type: ignore[no-untyped-def]
        del prompt, timeout, kwargs
        raise AssertionError("executor should use custom invoker in this test")

    def is_available(self) -> bool:
        return True

    def name(self) -> str:
        return "qwen_worker_cli"


def test_executor_hydrates_successful_tool_names_from_cli_metadata(tmp_path) -> None:
    artifact_store = ExecutionArtifactStore(tmp_path)
    direct_cfg = SimpleNamespace(
        provider="qwen_cli",
        timeout_seconds=30,
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

    async def fake_invoker(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return AdapterResponse(
            success=True,
            raw_output='```json\n{"type":"final_report","summary":"ok","verdict":"WATCHLIST","findings":[],"facts":{},"evidence_refs":["transcript:1:features_dataset"],"confidence":0.8}\n```',
            metadata={
                "tool_call_count": 5,
                "tool_names": [
                    "research_memory",
                    "features_custom",
                    "features_dataset",
                    "features_analytics",
                    "events",
                ],
            },
            duration_seconds=0.01,
        )

    executor = DirectSliceExecutor(
        adapter=_StaticAdapter(),
        artifact_store=artifact_store,
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=direct_cfg,
        worker_system_prompt="",
        invoker=fake_invoker,
        provider_name="qwen_cli",
        catalog_snapshot=make_catalog_snapshot(),
    )

    result = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=PlanSlice(
                slice_id="slice_cli",
                title="Data contract and feature contract",
                hypothesis="h",
                objective="o",
                success_criteria=["Custom features validated and published."],
                allowed_tools=["features_custom", "features_dataset", "features_analytics", "research_memory"],
                evidence_requirements=["e1"],
                policy_tags=["feature_contract"],
                max_turns=2,
                max_tool_calls=24,
                max_expensive_calls=6,
                parallel_slot=1,
            ),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
        )
    )

    assert result.action is not None
    assert result.action.facts["direct.successful_tool_names"] == [
        "research_memory",
        "features_custom",
        "features_dataset",
        "features_analytics",
    ]
    assert result.action.facts["direct.successful_tool_count"] == 5
