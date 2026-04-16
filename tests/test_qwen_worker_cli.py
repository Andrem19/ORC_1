from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.adapters.qwen_worker_cli import QwenWorkerCli
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import PlanSlice
from app.services.direct_execution.executor import DirectSliceExecutor


def test_qwen_worker_invoke_renders_stream_json_result(monkeypatch) -> None:
    transcript = "\n".join(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":"..."},{"type":"text","text":"{\\"type\\": \\"tool_call\\", \\"tool\\": \\"events\\", \\"arguments\\": {\\"view\\": \\"catalog\\"}, \\"reason\\": \\"check\\", \\"expected_evidence\\": [\\"ok\\"]}"}]}}',
            '{"type":"result","subtype":"success","result":"{\\"type\\": \\"tool_call\\", \\"tool\\": \\"events\\", \\"arguments\\": {\\"view\\": \\"catalog\\"}, \\"reason\\": \\"check\\", \\"expected_evidence\\": [\\"ok\\"]}"}',
        ]
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=transcript, stderr="")

    adapter = QwenWorkerCli(cli_path="/bin/echo")
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    monkeypatch.setattr("app.adapters.qwen_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.success is True
    assert response.raw_output.startswith("{")
    assert '"type": "tool_call"' in response.raw_output
    assert '"tool": "events"' in response.raw_output


def test_qwen_worker_build_command_avoids_plan_mode_when_tool_use_disabled(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=False)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--approval-mode" not in command
    assert "--yolo" not in command
    assert "--exclude-tools" in command
    assert "-e" in command
    assert "none" in command
    excluded = command[command.index("--exclude-tools") + 1]
    assert "write_file" in excluded
    assert "run_shell_command" in excluded


def test_qwen_worker_build_command_uses_yolo_only_when_tool_use_enabled(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--yolo" in command
    # Direct-safe tools are always excluded, even in yolo mode
    assert "--exclude-tools" in command
    excluded = command[command.index("--exclude-tools") + 1]
    assert "run_shell_command" in excluded
    assert "none" not in command


def test_qwen_worker_build_command_supports_exclude_tools_in_direct_mode(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True, exclude_tools=["read_file"])
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt", exclude_tools=["research_project", "read_file"])

    assert "--yolo" in command
    assert "--exclude-tools" in command
    excluded = command[command.index("--exclude-tools") + 1]
    assert "read_file" in excluded
    assert "research_project" in excluded


def test_qwen_worker_build_command_can_disable_tool_use_per_call(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt", allow_tool_use=False)

    assert "--yolo" not in command
    assert "-e" in command
    assert "none" in command


def test_qwen_tool_registry_preflight_caches_visible_tools(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    calls = {"count": 0}

    def _fake_invoke(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        calls["count"] += 1
        return SimpleNamespace(
            raw_output='{"visible_tools":["mcp__dev_space1__research_record","research_map"]}',
            success=True,
        )

    monkeypatch.setattr(adapter, "invoke", _fake_invoke)

    first = adapter.preflight_tool_registry(required_tools=["research_record"])
    second = adapter.preflight_tool_registry(required_tools=["research_record"])

    assert first["available"] is True
    assert "research_record" in first["visible_tools"]
    assert "mcp__dev_space1__research_record" in first["exact_visible_tools"]
    assert first["canonical_to_visible"]["research_record"] == "mcp__dev_space1__research_record"
    assert calls["count"] == 1
    assert second == first


def test_qwen_tool_registry_preflight_timeout_is_inconclusive(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    calls = {"count": 0}

    def _fake_invoke(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        calls["count"] += 1
        return SimpleNamespace(
            raw_output="",
            success=False,
            timed_out=True,
            error="Timed out after 30s",
        )

    monkeypatch.setattr(adapter, "invoke", _fake_invoke)

    first = adapter.preflight_tool_registry(required_tools=["research_project"])
    second = adapter.preflight_tool_registry(required_tools=["research_project"])

    assert first["available"] is True
    assert first["preflight_inconclusive"] is True
    assert first["reason"] == "probe_timeout"
    assert calls["count"] == 2
    assert second["reason"] == "probe_timeout"


def test_qwen_worker_invoke_counts_tool_calls_from_stream(monkeypatch) -> None:
    """Tool calls with mcp__dev_space1__ prefix are counted in metadata."""
    import json

    transcript = "\n".join(
        [
            '{"type":"system","subtype":"init"}',
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "checking..."},
                        {"type": "tool_use", "id": "call_1", "name": "mcp__dev_space1__research_project", "input": {}},
                    ],
                },
            }),
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "call_2", "name": "mcp__dev_space1__features_catalog", "input": {}},
                    ],
                },
            }),
            '{"type":"result","subtype":"success","result":"done"}',
        ]
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=transcript, stderr="")

    adapter = QwenWorkerCli(cli_path="/bin/echo")
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    monkeypatch.setattr("app.adapters.qwen_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.success is True
    assert response.metadata["tool_call_count"] == 2
    assert "research_project" in response.metadata["tool_names"]
    assert "features_catalog" in response.metadata["tool_names"]


def test_qwen_worker_invoke_no_tool_calls_returns_zero(monkeypatch) -> None:
    """When no dev_space1 tools are called, tool_call_count is 0."""
    transcript = "\n".join(
        [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"no tools needed"}]}}',
            '{"type":"result","subtype":"success","result":"done"}',
        ]
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=transcript, stderr="")

    adapter = QwenWorkerCli(cli_path="/bin/echo")
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    monkeypatch.setattr("app.adapters.qwen_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.success is True
    assert response.metadata["tool_call_count"] == 0
    assert response.metadata["tool_names"] == []


def _executor_slice() -> PlanSlice:
    return PlanSlice(
        slice_id="slice_1",
        title="slice",
        hypothesis="h",
        objective="o",
        success_criteria=["done"],
        allowed_tools=["research_project"],
        evidence_requirements=[],
        policy_tags=[],
        max_turns=2,
        max_tool_calls=4,
        max_expensive_calls=0,
        parallel_slot=1,
    )


def test_qwen_primary_preflight_retries_until_tools_appear(tmp_path, monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    calls = {"count": 0}

    def _fake_preflight(*, required_tools, timeout=60):  # type: ignore[no-untyped-def]
        del required_tools, timeout
        calls["count"] += 1
        if calls["count"] < 3:
            return {
                "available": False,
                "visible_tools": [],
                "exact_visible_tools": [],
                "canonical_to_visible": {},
                "missing_required_tools": ["research_project"],
            }
        return {
            "available": True,
            "visible_tools": ["research_project"],
            "exact_visible_tools": ["mcp__dev_space1__research_project"],
            "canonical_to_visible": {"research_project": "mcp__dev_space1__research_project"},
            "missing_required_tools": [],
        }

    monkeypatch.setattr(adapter, "preflight_tool_registry", _fake_preflight)
    monkeypatch.setattr(adapter, "invoke", lambda *args, **kwargs: SimpleNamespace(
        success=True,
        raw_output='```json\n{"type":"final_report","summary":"ok","verdict":"COMPLETE","findings":[],"facts":{},"evidence_refs":["node_1"],"confidence":0.8}\n```',
        error="",
        timed_out=False,
        finish_reason="completed",
        metadata={"tool_call_count": 1, "tool_names": ["research_project"]},
        duration_seconds=0.01,
    ))
    async def _fake_invoker(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        return adapter.invoke("prompt")

    executor = DirectSliceExecutor(
        adapter=adapter,
        artifact_store=ExecutionArtifactStore(tmp_path),
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=SimpleNamespace(
            provider="qwen_cli",
            timeout_seconds=30,
            max_attempts_per_slice=1,
            max_tool_calls_per_slice=4,
            max_expensive_tool_calls_per_slice=0,
            qwen_primary_preflight_enabled=True,
            qwen_tool_registry_preflight=True,
            qwen_primary_preflight_max_attempts=3,
            qwen_primary_preflight_retry_delay_seconds=0.0,
            qwen_preflight_timeout_seconds=1,
            safe_exclude_tools=[],
        ),
        worker_system_prompt="",
        provider_name="qwen_cli",
        invoker=_fake_invoker,
    )

    result = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=_executor_slice(),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
        )
    )

    assert calls["count"] == 3
    assert result.error == ""
    assert result.action is not None
    assert result.action.action_type == "final_report"


def test_qwen_primary_preflight_returns_infra_signal_after_exhausted_retries(tmp_path, monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    calls = {"count": 0}

    def _fake_preflight(*, required_tools, timeout=60):  # type: ignore[no-untyped-def]
        del required_tools, timeout
        calls["count"] += 1
        return {
            "available": False,
            "visible_tools": [],
            "exact_visible_tools": [],
            "canonical_to_visible": {},
            "missing_required_tools": ["research_project"],
        }

    monkeypatch.setattr(adapter, "preflight_tool_registry", _fake_preflight)

    executor = DirectSliceExecutor(
        adapter=adapter,
        artifact_store=ExecutionArtifactStore(tmp_path),
        incident_store=SimpleNamespace(record=lambda **_: None),
        direct_config=SimpleNamespace(
            provider="qwen_cli",
            timeout_seconds=30,
            max_attempts_per_slice=1,
            max_tool_calls_per_slice=4,
            max_expensive_tool_calls_per_slice=0,
            qwen_primary_preflight_enabled=True,
            qwen_tool_registry_preflight=True,
            qwen_primary_preflight_max_attempts=3,
            qwen_primary_preflight_retry_delay_seconds=0.0,
            qwen_preflight_timeout_seconds=1,
            safe_exclude_tools=[],
        ),
        worker_system_prompt="",
        provider_name="qwen_cli",
    )

    result = asyncio.run(
        executor.execute(
            plan_id="plan_1",
            slice_obj=_executor_slice(),
            baseline_bootstrap={},
            known_facts={},
            required_output_facts=[],
            recent_turn_summaries=[],
            checkpoint_summary="",
        )
    )

    assert calls["count"] == 3
    assert result.error == "qwen_mcp_tools_unavailable"
    assert result.action is None
