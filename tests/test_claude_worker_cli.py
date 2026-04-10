from __future__ import annotations

from types import SimpleNamespace

from app.adapters.claude_worker_cli import ClaudeWorkerCli


def test_claude_worker_build_command_yolo_when_tool_use_enabled(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--yolo" in command
    assert "--output-format" in command
    assert "stream-json" in command
    assert "-p" in command
    assert "prompt" in command


def test_claude_worker_build_command_no_yolo_when_tool_use_disabled(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=False)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--yolo" not in command
    assert "--output-format" in command


def test_claude_worker_build_command_exclude_tools(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        exclude_tools=["read_file", "write_file"],
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--exclude-tools" in command
    idx = command.index("--exclude-tools")
    excluded = command[idx + 1]
    assert "read_file" in excluded
    assert "write_file" in excluded


def test_claude_worker_build_command_runtime_exclude_tools(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=True, exclude_tools=["read_file"])
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt", exclude_tools=["research_project", "read_file"])

    assert "--exclude-tools" in command
    idx = command.index("--exclude-tools")
    excluded = command[idx + 1]
    assert "read_file" in excluded
    assert "research_project" in excluded


def test_claude_worker_build_command_model_flag(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", model="sonnet", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--model" in command
    idx = command.index("--model")
    assert command[idx + 1] == "sonnet"


def test_claude_worker_build_command_no_model_when_empty(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", model="", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--model" not in command


def test_claude_worker_name() -> None:
    adapter = ClaudeWorkerCli()
    assert adapter.name() == "claude_worker_cli"


def test_claude_worker_invoke_renders_stream_json(monkeypatch) -> None:
    transcript = "\n".join(
        [
            '{"type":"message_start","message":{"id":"msg_1"}}',
            '{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            '{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}',
            '{"type":"content_block_stop","index":0}',
            '{"type":"message_stop"}',
            '{"type":"result","subtype":"success","result":"Hello world"}',
        ]
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=transcript, stderr="")

    adapter = ClaudeWorkerCli(cli_path="/bin/echo")
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    monkeypatch.setattr("app.adapters.claude_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.success is True
    assert "Hello world" in response.raw_output
