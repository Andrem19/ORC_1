"""Tests for LMStudio assistant service."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
from unittest.mock import MagicMock

from app.config import LMStudioAssistantConfig, LMStudioConfig
from app.plan_models import PlanTask, ResearchPlan
from app.services.lmstudio_assistant import (
    ExecutionPrediction,
    LMStudioAssistant,
    LogAnalysisResult,
    StagePrediction,
    TaskHistoryEntry,
)


# ---------------------------------------------------------------------------
# Mock HTTP server
# ---------------------------------------------------------------------------

class MockLmStudioHandler(BaseHTTPRequestHandler):
    """Simulates LM Studio OpenAI-compatible API."""

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json_response(200, {"data": [{"id": "test-model"}]})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            content = self._generate_response(body)
            self._json_response(200, {
                "choices": [{"message": {"content": content}}],
            })
        else:
            self._json_response(404, {"error": "not found"})

    def _generate_response(self, body: dict) -> str:
        user_msg = body.get("messages", [{}])[-1].get("content", "")
        if "LOG START" in user_msg:
            return json.dumps({
                "frequent_errors": ["MCP timeout", "Worker stall"],
                "error_patterns": ["Timeout every 10 cycles"],
                "avg_task_seconds": 45.0,
                "trend": "stable",
                "recommendations": ["Increase MCP timeout"],
                "digest_summary": "System running stable with occasional MCP timeouts.",
            })
        elif "Execution History" in user_msg or "New Research Plan" in user_msg:
            return json.dumps({
                "stage_predictions": [
                    {"stage_number": 1, "stage_name": "Explore", "estimated_minutes": 8.0},
                    {"stage_number": 2, "stage_name": "Backtest", "estimated_minutes": 15.0},
                ],
                "total_estimated_minutes": 23.0,
                "summary": "Stage 1 (~8 min), Stage 2 (~15 min). Total ~23 min.",
            })
        return "ok"

    def _json_response(self, status: int, data: Any) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress request logging


def _start_mock_server() -> tuple[HTTPServer, int]:
    server = HTTPServer(("127.0.0.1", 0), MockLmStudioHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _make_config(port: int) -> LMStudioConfig:
    return LMStudioConfig(
        enabled=True,
        base_url=f"http://127.0.0.1:{port}",
        model="test-model",
        assistant=LMStudioAssistantConfig(timeout_seconds=5),
    )


def _make_plan() -> ResearchPlan:
    return ResearchPlan(
        version=3,
        tasks=[
            PlanTask(stage_number=1, stage_name="Explore features"),
            PlanTask(stage_number=2, stage_name="Run backtests"),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIsAvailable:
    def test_available_when_server_running(self) -> None:
        server, port = _start_mock_server()
        try:
            svc = LMStudioAssistant(_make_config(port))
            assert svc.is_available() is True
        finally:
            server.shutdown()

    def test_not_available_when_server_down(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(
            enabled=True,
            base_url="http://127.0.0.1:1",
        ))
        assert svc.is_available() is False

    def test_not_available_when_disabled(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(enabled=False))
        assert svc.is_available() is False


class TestAnalyzeLog:
    def test_returns_result_on_valid_response(self) -> None:
        server, port = _start_mock_server()
        try:
            svc = LMStudioAssistant(_make_config(port))
            result = svc.analyze_log(["[INFO] Cycle 1: started", "[ERROR] MCP timeout"])
            assert isinstance(result, LogAnalysisResult)
            assert result.trend == "stable"
            assert len(result.frequent_errors) == 2
            assert "MCP timeout" in result.recommendations[0]
        finally:
            server.shutdown()

    def test_returns_none_on_empty_lines(self) -> None:
        server, port = _start_mock_server()
        try:
            svc = LMStudioAssistant(_make_config(port))
            assert svc.analyze_log([]) is None
        finally:
            server.shutdown()

    def test_returns_none_when_server_down(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(
            enabled=True,
            base_url="http://127.0.0.1:1",
        ))
        result = svc.analyze_log(["some log line"])
        assert result is None

    def test_handles_non_json_response(self) -> None:
        """If LMStudio returns plain text, fallback to raw_digest."""
        server, port = _start_mock_server()
        # Override handler to return non-JSON
        class PlainTextHandler(MockLmStudioHandler):
            def _generate_response(self, body: dict) -> str:
                return "This is a plain text analysis of the logs."

        server2 = HTTPServer(("127.0.0.1", 0), PlainTextHandler)
        port2 = server2.server_address[1]
        threading.Thread(target=server2.serve_forever, daemon=True).start()
        try:
            svc = LMStudioAssistant(_make_config(port2))
            result = svc.analyze_log(["log line"])
            assert result is not None
            assert result.raw_digest == "This is a plain text analysis of the logs."
            assert result.frequent_errors == []
        finally:
            server2.shutdown()
            server.shutdown()


class TestPredictExecutionTime:
    def test_returns_prediction_on_valid_response(self) -> None:
        server, port = _start_mock_server()
        try:
            svc = LMStudioAssistant(_make_config(port))
            history = [
                TaskHistoryEntry(stage_number=1, stage_name="Explore", execution_minutes=7.5),
                TaskHistoryEntry(stage_number=2, stage_name="Backtest", execution_minutes=14.0),
            ]
            result = svc.predict_execution_time(history, _make_plan())
            assert isinstance(result, ExecutionPrediction)
            assert len(result.stage_predictions) == 2
            assert result.total_estimated_minutes == 23.0
        finally:
            server.shutdown()

    def test_works_with_empty_history(self) -> None:
        server, port = _start_mock_server()
        try:
            svc = LMStudioAssistant(_make_config(port))
            result = svc.predict_execution_time([], _make_plan())
            assert result is not None
        finally:
            server.shutdown()

    def test_returns_none_when_server_down(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(
            enabled=True,
            base_url="http://127.0.0.1:1",
        ))
        result = svc.predict_execution_time([], _make_plan())
        assert result is None


class TestJsonExtraction:
    def test_extracts_plain_json(self) -> None:
        text = '{"key": "value", "num": 42}'
        assert LMStudioAssistant._extract_json(text) == {"key": "value", "num": 42}

    def test_extracts_from_code_fence(self) -> None:
        text = '```json\n{"key": "value"}\n```'
        assert LMStudioAssistant._extract_json(text) == {"key": "value"}

    def test_extracts_from_mixed_text(self) -> None:
        text = 'Here is the result:\n{"key": "value"}\nThat is all.'
        assert LMStudioAssistant._extract_json(text) == {"key": "value"}

    def test_returns_none_for_no_json(self) -> None:
        text = "No JSON here, just plain text."
        assert LMStudioAssistant._extract_json(text) is None


class TestFormatHelpers:
    def test_format_history_with_entries(self) -> None:
        entries = [
            TaskHistoryEntry(stage_number=1, stage_name="Explore", execution_minutes=5.0),
            TaskHistoryEntry(stage_number=2, stage_name="Backtest", execution_minutes=12.3),
        ]
        result = LMStudioAssistant._format_history(entries)
        assert "Stage 1 (Explore): 5.0 min" in result
        assert "Stage 2 (Backtest): 12.3 min" in result

    def test_format_history_empty(self) -> None:
        result = LMStudioAssistant._format_history([])
        assert "No previous" in result

    def test_format_history_caps_at_30(self) -> None:
        entries = [
            TaskHistoryEntry(stage_number=i, stage_name=f"S{i}", execution_minutes=float(i))
            for i in range(50)
        ]
        result = LMStudioAssistant._format_history(entries)
        lines = result.strip().split("\n")
        assert len(lines) == 30

    def test_format_plan_stages(self) -> None:
        plan = _make_plan()
        result = LMStudioAssistant._format_plan_stages(plan)
        assert "Stage 1" in result
        assert "Stage 2" in result
        assert "Explore features" in result


class TestParseFallback:
    def test_parse_log_analysis_with_invalid_json_uses_raw(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(enabled=True))
        result = svc._parse_log_analysis("Not JSON but useful text")
        assert result.raw_digest == "Not JSON but useful text"
        assert result.frequent_errors == []
        assert result.trend == "stable"

    def test_parse_execution_prediction_with_invalid_json_uses_raw(self) -> None:
        svc = LMStudioAssistant(LMStudioConfig(enabled=True))
        result = svc._parse_execution_prediction("Plain text prediction")
        assert result.raw_response == "Plain text prediction"
        assert result.stage_predictions == []
