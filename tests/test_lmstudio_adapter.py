"""
Tests for LM Studio worker adapter.

Uses patched HTTPConnection objects so no real LM Studio instance is needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.adapters.lmstudio_worker_api import LmStudioWorkerApi


def _make_adapter(
    base_url: str = "http://127.0.0.1:1234",
    model: str = "qwen3-4b",
    reasoning_effort: str = "",
) -> LmStudioWorkerApi:
    return LmStudioWorkerApi(
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
    )


def _mock_http_connection(*, status: int = 200, payload: dict | None = None):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload or {}).encode()
    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    return patch("app.lmstudio_api.HTTPConnection", return_value=mock_conn)


def _mock_invoke_connection(*, status: int = 200, payload: dict | None = None):
    mock_resp = MagicMock()
    mock_resp.status = status
    mock_resp.read.return_value = json.dumps(payload or {}).encode()
    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp
    return patch("app.adapters.lmstudio_worker_api.HTTPConnection", return_value=mock_conn)


def test_adapter_name():
    adapter = LmStudioWorkerApi()
    assert adapter.name() == "lmstudio_worker_api"


def test_is_available_when_server_running():
    adapter = _make_adapter()
    with _mock_http_connection(payload={"data": [{"id": "qwen3-4b"}]}):
        assert adapter.is_available() is True


def test_is_available_when_server_down():
    adapter = _make_adapter("http://127.0.0.1:1")
    with patch("app.lmstudio_api.HTTPConnection") as mock_conn_cls:
        mock_conn_cls.return_value.request.side_effect = ConnectionError()
        assert adapter.is_available() is False


def test_is_available_false_when_model_missing():
    adapter = _make_adapter(model="missing-model")
    with _mock_http_connection(payload={"data": [{"id": "qwen3-4b"}]}):
        assert adapter.is_available() is False


def test_invoke_returns_success():
    adapter = _make_adapter()
    with _mock_invoke_connection(payload={
        "choices": [{
            "message": {
                "role": "assistant",
                "content": json.dumps({
                    "status": "success",
                    "summary": "Task completed by LM Studio model",
                    "artifacts": ["output.txt"],
                    "confidence": 0.88,
                }),
            },
        }],
    }):
        response = adapter.invoke("Write a hello world function")

    assert response.success is True
    assert response.exit_code == 0
    assert "Task completed by LM Studio model" in response.raw_output
    assert response.duration_seconds >= 0


def test_start_and_check_returns_async_result():
    adapter = _make_adapter()
    with _mock_invoke_connection(payload={
        "choices": [{"message": {"content": "Task completed by LM Studio model"}}],
    }):
        handle = adapter.start("Write a hello world function", task_id="t1", worker_id="w1")
        for _ in range(50):
            output, finished = adapter.check(handle)
            if finished:
                break
        else:
            pytest.fail("LM Studio async worker did not finish in time")

    assert finished is True
    assert "Task completed by LM Studio model" in output
    assert handle.partial_output == output


def test_start_uses_default_system_prompt_when_none_provided():
    adapter = _make_adapter(model="qwen3-4b", reasoning_effort="none")
    with _mock_invoke_connection(payload={"choices": [{"message": {"content": "ok"}}]}) as patched:
        handle = adapter.start("Run task", task_id="t1", worker_id="w1")
        for _ in range(50):
            output, finished = adapter.check(handle)
            if finished:
                break
        else:
            pytest.fail("LM Studio async worker did not finish in time")
        mock_conn = patched.return_value

    body = json.loads(mock_conn.request.call_args[0][2])
    assert body["messages"][0]["role"] == "system"
    assert isinstance(body["messages"][0]["content"], str)
    assert body["messages"][0]["content"]
    assert output == "ok"


def test_invoke_sends_correct_payload():
    adapter = _make_adapter(model="qwen3-4b", reasoning_effort="none")
    with _mock_invoke_connection(payload={"choices": [{"message": {"content": "ok"}}]}) as patched:
        adapter.invoke("Test prompt", system_prompt="Be helpful")
        mock_conn = patched.return_value

    call_args = mock_conn.request.call_args
    body = json.loads(call_args[0][2])
    assert body["model"] == "qwen3-4b"
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 4096
    assert body["reasoning_effort"] == "none"
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Be helpful"
    assert body["messages"][1]["role"] == "user"
    assert body["messages"][1]["content"] == "Test prompt"


def test_invoke_returns_error_on_empty_content_with_reasoning_only_output():
    adapter = _make_adapter(model="qwen3-4b")
    with _mock_invoke_connection(payload={
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "Thinking about the task",
            },
            "finish_reason": "length",
        }],
    }):
        response = adapter.invoke("Test")

    assert response.success is False
    assert "empty assistant content" in response.error
    assert "Thinking about the task" in response.raw_output


def test_invoke_handles_http_error():
    adapter = _make_adapter()
    with _mock_invoke_connection(status=500, payload={"error": {"message": "Internal Server Error"}}):
        response = adapter.invoke("Test")

    assert response.success is False
    assert response.exit_code == 500
    assert "500" in response.error


def test_invoke_handles_connection_error():
    adapter = _make_adapter("http://127.0.0.1:1")
    with patch("app.adapters.lmstudio_worker_api.HTTPConnection") as mock_conn_cls:
        mock_conn_cls.return_value.request.side_effect = ConnectionError("refused")
        response = adapter.invoke("Test", timeout=3)

    assert response.success is False
    assert response.exit_code == -1
    assert response.error


def test_invoke_with_no_model_defaults_to_empty():
    adapter = LmStudioWorkerApi(base_url="http://127.0.0.1:1234", model="")
    with _mock_invoke_connection(payload={"choices": [{"message": {"content": "ok"}}]}) as patched:
        adapter.invoke("Test")
        mock_conn = patched.return_value

    body = json.loads(mock_conn.request.call_args[0][2])
    assert "model" not in body
    assert "messages" in body


def test_extract_content_from_valid_response():
    data = {
        "choices": [{
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop",
        }],
    }
    assert LmStudioWorkerApi._extract_content(data) == "Hello world"


def test_extract_content_from_empty_choices():
    data = {"choices": []}
    result = LmStudioWorkerApi._extract_content(data)
    assert result == json.dumps(data)


def test_extract_content_from_malformed_response():
    data = {"error": "something went wrong"}
    result = LmStudioWorkerApi._extract_content(data)
    assert result == json.dumps(data)


def test_adapter_config_fields():
    from app.config import AdapterConfig

    cfg = AdapterConfig(
        name="lmstudio_worker_api",
        base_url="http://localhost:1234",
        model="qwen3-4b",
        api_key="lm-studio",
        temperature=0.5,
        max_tokens=2048,
    )
    assert cfg.name == "lmstudio_worker_api"
    assert cfg.base_url == "http://localhost:1234"
    assert cfg.model == "qwen3-4b"
    assert cfg.temperature == 0.5
    assert cfg.max_tokens == 2048


def test_load_config_with_lmstudio():
    from app.config import load_config_from_dict

    data = {
        "goal": "Test with LM Studio",
        "worker_adapter": {
            "name": "lmstudio_worker_api",
            "base_url": "http://localhost:1234",
            "model": "llama-3-8b",
            "temperature": 0.3,
            "max_tokens": 8192,
        },
    }
    cfg = load_config_from_dict(data)
    assert cfg.worker_adapter.name == "lmstudio_worker_api"
    assert cfg.worker_adapter.base_url == "http://localhost:1234"
    assert cfg.worker_adapter.model == "llama-3-8b"
    assert cfg.worker_adapter.temperature == 0.3
    assert cfg.worker_adapter.max_tokens == 8192
