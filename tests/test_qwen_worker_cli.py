from __future__ import annotations

from types import SimpleNamespace

from app.adapters.qwen_worker_cli import QwenWorkerCli


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
    assert calls["count"] == 1
    assert second == first


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
