from __future__ import annotations

from types import SimpleNamespace

from app.adapters.claude_worker_cli import ClaudeWorkerCli


def test_claude_worker_build_command_dangerously_skip_permissions(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=True)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--dangerously-skip-permissions" in command
    assert "--output-format" in command
    assert "stream-json" in command
    assert "-p" in command
    assert "prompt" in command


def test_claude_worker_build_command_no_skip_permissions_when_disabled(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=False)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--dangerously-skip-permissions" not in command
    assert "--output-format" in command


def test_claude_worker_build_command_disallowed_tools(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        exclude_tools=["read_file", "write_file"],
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--disallowedTools" in command
    idx = command.index("--disallowedTools")
    # Space-separated tools after the flag
    after_flag = command[idx + 1:]
    assert "read_file" in after_flag
    assert "write_file" in after_flag


def test_claude_worker_build_command_runtime_exclude_tools(monkeypatch) -> None:
    adapter = ClaudeWorkerCli(cli_path="/bin/echo", allow_tool_use=True, exclude_tools=["read_file"])
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt", exclude_tools=["research_project", "read_file"])

    assert "--disallowedTools" in command
    idx = command.index("--disallowedTools")
    after_flag = command[idx + 1:]
    assert "read_file" in after_flag
    assert "research_project" in after_flag


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


def test_claude_worker_invoke_counts_tool_calls_from_stream(monkeypatch) -> None:
    """Tool calls in stream-json output are counted and stored in metadata."""
    import json

    transcript = "\n".join(
        [
            '{"type":"message_start","message":{"id":"msg_1"}}',
            json.dumps({
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "research_project", "input": {}},
            }),
            '{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{}"}}',
            '{"type":"content_block_stop","index":1}',
            json.dumps({
                "type": "content_block_start",
                "index": 2,
                "content_block": {"type": "tool_use", "id": "toolu_2", "name": "research_map", "input": {}},
            }),
            '{"type":"content_block_stop","index":2}',
            '{"type":"result","subtype":"success","result":"done"}',
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
    assert response.metadata["tool_call_count"] == 2
    assert "research_project" in response.metadata["tool_names"]
    assert "research_map" in response.metadata["tool_names"]


def test_claude_worker_invoke_ignores_local_tools(monkeypatch) -> None:
    """Local tools like Bash/read_file should not be counted."""
    import json

    transcript = "\n".join(
        [
            json.dumps({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "Bash", "input": {}},
            }),
            json.dumps({
                "type": "content_block_start",
                "content_block": {"type": "tool_use", "name": "research_project", "input": {}},
            }),
            '{"type":"result","subtype":"success","result":"done"}',
        ]
    )

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout=transcript, stderr="")

    adapter = ClaudeWorkerCli(cli_path="/bin/echo")
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")
    monkeypatch.setattr("app.adapters.claude_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.metadata["tool_call_count"] == 1
    assert response.metadata["tool_names"] == ["research_project"]


def test_claude_worker_build_command_with_mcp_config(monkeypatch, tmp_path) -> None:
    """MCP config file should be generated and --mcp-config flag added."""
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        mcp_servers={
            "dev_space1": {
                "type": "http",
                "url": "http://127.0.0.1:8766/mcp",
                "headers": {"Authorization": "Bearer token123"},
            },
        },
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--mcp-config" in command
    idx = command.index("--mcp-config")
    mcp_config_path = command[idx + 1]
    assert mcp_config_path.startswith("/tmp/")
    assert "claude_mcp_" in mcp_config_path

    import os
    assert os.path.exists(mcp_config_path)

    import json
    with open(mcp_config_path) as f:
        config = json.load(f)
    assert "mcpServers" in config
    assert "dev_space1" in config["mcpServers"]
    assert config["mcpServers"]["dev_space1"]["type"] == "http"
    assert config["mcpServers"]["dev_space1"]["url"] == "http://127.0.0.1:8766/mcp"


def test_claude_worker_build_command_with_allowed_mcp_tools(monkeypatch) -> None:
    """--allowedTools flag should be added with comma-separated tool names."""
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        allowed_mcp_tools=[
            "mcp__dev_space1__backtests_runs",
            "mcp__dev_space1__features_catalog",
            "mcp__dev_space1__research_memory",
        ],
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--allowedTools" in command
    idx = command.index("--allowedTools")
    allowed_tools_str = command[idx + 1]
    assert "mcp__dev_space1__backtests_runs" in allowed_tools_str
    assert "mcp__dev_space1__features_catalog" in allowed_tools_str
    assert "mcp__dev_space1__research_memory" in allowed_tools_str
    assert "," in allowed_tools_str


def test_claude_worker_build_command_runtime_mcp_tools(monkeypatch) -> None:
    """Runtime allowed_mcp_tools via kwargs should be used."""
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        allowed_mcp_tools=["mcp__dev_space1__default_tool"],
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command(
        "prompt",
        allowed_mcp_tools=[
            "mcp__dev_space1__backtests_runs",
            "mcp__dev_space1__features_catalog",
        ],
    )

    idx = command.index("--allowedTools")
    allowed_tools_str = command[idx + 1]
    # Runtime tools should override default
    assert "backtests_runs" in allowed_tools_str
    assert "features_catalog" in allowed_tools_str
    assert "default_tool" not in allowed_tools_str


def test_claude_worker_build_command_no_mcp_when_tool_use_disabled(monkeypatch) -> None:
    """MCP flags should not be added when tool use is disabled."""
    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=False,
        mcp_servers={"dev_space1": {"type": "http", "url": "http://localhost/mcp"}},
        allowed_mcp_tools=["mcp__dev_space1__backtests_runs"],
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--mcp-config" not in command
    assert "--allowedTools" not in command


def test_claude_worker_invoke_cleans_up_mcp_config(monkeypatch) -> None:
    """Temporary MCP config files should be cleaned up after invoke."""
    import tempfile
    created_files: list[str] = []

    original_mkstemp = tempfile.mkstemp
    def _track_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_files.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _track_mkstemp)

    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        mcp_servers={"dev_space1": {"type": "http", "url": "http://localhost/mcp"}},
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    def _fake_run(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(returncode=0, stdout='{"type":"result","result":"ok"}', stderr="")

    monkeypatch.setattr("app.adapters.claude_worker_cli.subprocess.run", _fake_run)

    response = adapter.invoke("prompt", timeout=5)

    assert response.success is True
    # Files should be cleaned up after invoke
    import os
    for path in created_files:
        assert not os.path.exists(path), f"Temp file {path} was not cleaned up"


def test_claude_worker_cleanup_mcp_configs_multiple(monkeypatch) -> None:
    """Multiple MCP config files should be tracked and cleaned up."""
    import tempfile
    created_files: list[str] = []

    original_mkstemp = tempfile.mkstemp
    def _track_mkstemp(*args, **kwargs):
        fd, path = original_mkstemp(*args, **kwargs)
        created_files.append(path)
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", _track_mkstemp)

    adapter = ClaudeWorkerCli(
        cli_path="/bin/echo",
        allow_tool_use=True,
        mcp_servers={"dev_space1": {"type": "http", "url": "http://localhost/mcp"}},
    )
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    # Build command multiple times to create multiple temp files
    adapter._build_command("prompt1")
    adapter._build_command("prompt2")
    adapter._build_command("prompt3")

    assert len(created_files) >= 3

    # Cleanup should remove all
    adapter._cleanup_mcp_configs()

    import os
    for path in created_files:
        assert not os.path.exists(path), f"Temp file {path} was not cleaned up"
