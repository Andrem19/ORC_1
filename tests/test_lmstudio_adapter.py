"""
Tests for LM Studio worker adapter.

Uses a mock HTTP server to simulate LM Studio responses,
so no real LM Studio instance is needed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import patch

import pytest

from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.adapters.base import AdapterResponse


# ---------------------------------------------------------------------------
# Mock LM Studio HTTP server
# ---------------------------------------------------------------------------

class MockLmStudioHandler(BaseHTTPRequestHandler):
    """Simulates LM Studio's OpenAI-compatible API."""

    # Class-level config for controlling responses
    _response_override: dict[str, Any] | None = None
    _status_override: int | None = None

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({
                "data": [
                    {"id": "qwen3-4b", "object": "model"},
                    {"id": "llama-3-8b", "object": "model"},
                ]
            })
            self._send_json(200, body)
        else:
            self._send_json(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            status = self._status_override or 200
            if self._response_override:
                body = json.dumps(self._response_override)
            else:
                body = json.dumps({
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
                        "finish_reason": "stop",
                        "index": 0,
                    }],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
                })
            self._send_json(status, body)
        else:
            self._send_json(404, json.dumps({"error": "not found"}))

    def _send_json(self, status: int, body: str):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format, *args):
        pass  # Silence server logs in tests


@pytest.fixture
def mock_server():
    """Start a mock LM Studio server on a free port."""
    server = HTTPServer(("127.0.0.1", 0), MockLmStudioHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}", server
    server.shutdown()


def _make_adapter(base_url: str, model: str = "qwen3-4b") -> LmStudioWorkerApi:
    return LmStudioWorkerApi(base_url=base_url, model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_adapter_name():
    adapter = LmStudioWorkerApi()
    assert adapter.name() == "lmstudio_worker_api"


def test_is_available_when_server_running(mock_server):
    url, _ = mock_server
    adapter = _make_adapter(url)
    assert adapter.is_available() is True


def test_is_available_when_server_down():
    adapter = _make_adapter("http://127.0.0.1:1")
    assert adapter.is_available() is False


def test_invoke_returns_success(mock_server):
    url, _ = mock_server
    adapter = _make_adapter(url)
    response = adapter.invoke("Write a hello world function")

    assert response.success is True
    assert response.exit_code == 0
    assert "Task completed by LM Studio model" in response.raw_output
    assert response.duration_seconds > 0


def test_invoke_sends_correct_payload(mock_server):
    url, server = mock_server
    adapter = _make_adapter(url, model="qwen3-4b")

    captured_body = {}

    original_do_post = MockLmStudioHandler.do_POST

    def capturing_do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        captured_body.update(json.loads(raw))
        original_do_post(self)

    with patch.object(MockLmStudioHandler, "do_POST", capturing_do_post):
        adapter.invoke("Test prompt", system_prompt="Be helpful")

    assert captured_body["model"] == "qwen3-4b"
    assert captured_body["temperature"] == 0.7
    assert captured_body["max_tokens"] == 4096
    assert len(captured_body["messages"]) == 2
    assert captured_body["messages"][0]["role"] == "system"
    assert captured_body["messages"][1]["role"] == "user"
    assert captured_body["messages"][1]["content"] == "Test prompt"


def test_invoke_handles_http_error(mock_server):
    url, server = mock_server
    MockLmStudioHandler._status_override = 500
    MockLmStudioHandler._response_override = {"error": {"message": "Internal Server Error"}}
    try:
        adapter = _make_adapter(url)
        response = adapter.invoke("Test")

        assert response.success is False
        assert response.exit_code == 500
        assert "500" in response.error
    finally:
        MockLmStudioHandler._status_override = None
        MockLmStudioHandler._response_override = None


def test_invoke_handles_connection_error():
    adapter = _make_adapter("http://127.0.0.1:1")
    response = adapter.invoke("Test", timeout=3)

    assert response.success is False
    assert response.exit_code == -1
    assert response.error  # non-empty error message


def test_invoke_with_custom_system_prompt(mock_server):
    url, _ = mock_server
    adapter = _make_adapter(url)

    captured_body = {}
    original_do_post = MockLmStudioHandler.do_POST

    def capturing_do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        captured_body.update(json.loads(raw))
        original_do_post(self)

    with patch.object(MockLmStudioHandler, "do_POST", capturing_do_post):
        adapter.invoke("Do the thing", system_prompt="You are a code reviewer")

    assert captured_body["messages"][0]["content"] == "You are a code reviewer"


def test_invoke_with_no_model_defaults_to_empty(mock_server):
    url, _ = mock_server
    adapter = LmStudioWorkerApi(base_url=url, model="")

    captured_body = {}
    original_do_post = MockLmStudioHandler.do_POST

    def capturing_do_post(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        captured_body.update(json.loads(raw))
        original_do_post(self)

    with patch.object(MockLmStudioHandler, "do_POST", capturing_do_post):
        adapter.invoke("Test")

    # When model is empty, the API call should still work
    # (LM Studio uses whatever model is loaded)
    assert "messages" in captured_body


def test_extract_content_from_valid_response():
    data = {
        "choices": [{
            "message": {"role": "assistant", "content": "Hello world"},
            "finish_reason": "stop",
        }]
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
    """Verify AdapterConfig supports LM Studio fields."""
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
    """Verify config loading with LM Studio adapter."""
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
