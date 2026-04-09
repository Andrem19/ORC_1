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


def test_qwen_worker_build_command_uses_plan_mode_when_tool_use_disabled(monkeypatch) -> None:
    adapter = QwenWorkerCli(cli_path="/bin/echo", allow_tool_use=False)
    monkeypatch.setattr(adapter, "_resolve_cli_path", lambda: "/bin/echo")

    command = adapter._build_command("prompt")

    assert "--approval-mode" in command
    assert "plan" in command
    assert "--yolo" not in command
