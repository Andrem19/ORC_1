"""Tests for the LM Studio report compressor service."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from app.config import ReportCompressorConfig, load_config_from_dict
from app.plan_models import TaskReport
from app.services.report_compressor import ReportCompressorService


def _make_report(**overrides) -> TaskReport:
    defaults = dict(
        task_id="t1",
        worker_id="qwen-1",
        status="success",
        verdict="PROMOTE",
        plan_version=1,
        what_was_done="Created cf_alpha_rsi feature",
        key_metrics={"net_pnl": -45, "sharpe": 0.3},
        results_table=[{"net_pnl": -45, "trades": 12}],
    )
    defaults.update(overrides)
    return TaskReport(**defaults)


def _mock_lmstudio_ok(response_text: str = "Compressed summary here."):
    """Patch HTTPConnection to simulate a successful LM Studio response."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "choices": [{"message": {"content": response_text}}]
    }).encode()

    mock_conn = MagicMock()
    mock_conn.getresponse.return_value = mock_resp

    return patch(
        "app.services.report_compressor.HTTPConnection",
        return_value=mock_conn,
    )


class TestConfig:
    def test_defaults(self):
        cfg = ReportCompressorConfig()
        assert cfg.enabled is False
        assert cfg.base_url == "http://localhost:1234"
        assert cfg.model == ""
        assert cfg.max_tokens == 200
        assert cfg.timeout_seconds == 30

    def test_from_dict(self):
        data = {"report_compressor": {"enabled": True, "base_url": "http://localhost:9999"}}
        cfg = load_config_from_dict(data)
        assert cfg.report_compressor.enabled is True
        assert cfg.report_compressor.base_url == "http://localhost:9999"

    def test_missing_key(self):
        cfg = load_config_from_dict({})
        assert cfg.report_compressor.enabled is False


class TestFallback:
    def test_disabled_uses_compact(self):
        svc = ReportCompressorService(ReportCompressorConfig(enabled=False))
        reports = [_make_report()]
        result = svc.compress_reports(reports)
        # Should use compact_reports_for_revision format
        assert "stage plan_v1" in result
        assert "done:" in result

    def test_unavailable_uses_compact(self):
        cfg = ReportCompressorConfig(enabled=True)
        svc = ReportCompressorService(cfg)
        with patch.object(svc, "is_available", return_value=False):
            reports = [_make_report()]
            result = svc.compress_reports(reports)
            assert "stage plan_v1" in result

    def test_empty_reports(self):
        svc = ReportCompressorService()
        assert svc.compress_reports([]) == ""


class TestCompression:
    def test_compress_reports_returns_string(self):
        svc = ReportCompressorService(ReportCompressorConfig(enabled=True))
        reports = [_make_report()]
        with _mock_lmstudio_ok("Created RSI feature. PnL=-$45, 12 trades. PROMOTE."):
            result = svc.compress_reports(reports)
        assert "stage plan_v1" in result
        assert "RSI" in result

    def test_compress_multiple_reports(self):
        svc = ReportCompressorService(ReportCompressorConfig(enabled=True))
        r1 = _make_report(task_id="t1", verdict="PROMOTE")
        r2 = _make_report(task_id="t2", verdict="REJECT", what_was_done="Different test")
        with _mock_lmstudio_ok("Summary text."):
            result = svc.compress_reports([r1, r2])
        assert result.count("stage plan_v1") == 2

    def test_single_report_error_fallback(self):
        """If LM Studio fails for one report, fallback to compact for that one."""
        svc = ReportCompressorService(ReportCompressorConfig(enabled=True))
        reports = [_make_report()]

        # Simulate LM Studio returning non-200
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.read.return_value = b'{"error":"internal"}'
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch(
            "app.services.report_compressor.HTTPConnection",
            return_value=mock_conn,
        ):
            result = svc.compress_reports(reports)
        # Should fallback to compact format
        assert "stage plan_v1" in result


class TestFormatReport:
    def test_format_includes_all_fields(self):
        report = _make_report(error="test error")
        text = ReportCompressorService._format_report(report)
        data = json.loads(text)
        assert data["worker"] == "qwen-1"
        assert data["verdict"] == "PROMOTE"
        assert data["error"] == "test error"
        assert data["key_metrics"]["net_pnl"] == -45

    def test_format_limits_results_table(self):
        report = _make_report(results_table=[{"a": 1}, {"b": 2}, {"c": 3}, {"d": 4}])
        text = ReportCompressorService._format_report(report)
        data = json.loads(text)
        assert len(data["results_table"]) == 3
