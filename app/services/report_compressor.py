"""
Report compressor — uses LM Studio to compress worker reports into short summaries.

If LM Studio is unavailable, falls back to the existing compact_reports_for_revision().
"""

from __future__ import annotations

import json
import logging
import time
from http.client import HTTPConnection
from urllib.parse import urlparse

from app.config import ReportCompressorConfig
from app.plan_models import TaskReport
from app.plan_prompt_budget import compact_reports_for_revision

logger = logging.getLogger("orchestrator.report_compressor")

_COMPRESS_SYSTEM_PROMPT = """You are a research report compressor.
Given a worker's research report, produce a 2-3 sentence summary that captures:
1. What was done (which tools/features were used)
2. Key results (metrics: net_pnl, sharpe, trades, drawdown)
3. The worker's verdict and why

Be specific with numbers. Do NOT add commentary or suggestions."""


class ReportCompressorService:
    """Compresses TaskReports via LM Studio for smaller revision prompts."""

    def __init__(self, config: ReportCompressorConfig | None = None) -> None:
        self.config = config or ReportCompressorConfig()
        self._available_checked: bool = False
        self._available: bool = False

    def is_available(self) -> bool:
        """Check if LM Studio server is reachable."""
        if not self.config.enabled:
            return False
        try:
            parsed = urlparse(self.config.base_url)
            conn = HTTPConnection(parsed.hostname, parsed.port or 1234, timeout=5)
            conn.request("GET", "/v1/models")
            resp = conn.getresponse()
            conn.close()
            ok = resp.status == 200
            if ok and not self._available_checked:
                logger.info("LM Studio report compressor available at %s", self.config.base_url)
            self._available = ok
            self._available_checked = True
            return ok
        except Exception:
            self._available = False
            self._available_checked = True
            return False

    def compress_reports(self, reports: list[TaskReport]) -> str:
        """Compress all reports. Falls back to compact_reports_for_revision if LM Studio is down."""
        if not reports:
            return ""

        if not self.config.enabled or not self.is_available():
            return compact_reports_for_revision(reports)

        start = time.monotonic()
        lines: list[str] = []
        compressed_count = 0

        for report in reports:
            header = (
                f"- stage plan_v{report.plan_version} | worker={report.worker_id} "
                f"| status={report.status} | verdict={report.verdict}"
            )
            compressed = self._compress_single(report)
            if compressed:
                lines.append(header)
                lines.append(f"  {compressed}")
                compressed_count += 1
            else:
                # Fallback for this single report
                fallback = compact_reports_for_revision([report])
                lines.append(fallback)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Compressed %d/%d reports via LM Studio (%.0fms)",
            compressed_count, len(reports), elapsed_ms,
        )
        return "\n".join(lines)

    def _compress_single(self, report: TaskReport) -> str | None:
        """Compress one report via LM Studio. Returns None on failure."""
        report_text = self._format_report(report)
        body: dict = {
            "messages": [
                {"role": "system", "content": _COMPRESS_SYSTEM_PROMPT},
                {"role": "user", "content": report_text},
            ],
            "temperature": 0.3,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.model:
            body["model"] = self.config.model

        try:
            parsed = urlparse(self.config.base_url)
            conn = HTTPConnection(
                parsed.hostname, parsed.port or 1234,
                timeout=self.config.timeout_seconds,
            )
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            conn.close()

            if resp.status != 200:
                logger.warning(
                    "LM Studio compression returned HTTP %d: %s",
                    resp.status, resp_body[:200],
                )
                return None

            data = json.loads(resp_body)
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            ).strip()
            return content if content else None

        except Exception as e:
            logger.warning("LM Studio compression failed: %s", e)
            return None

    @staticmethod
    def _format_report(report: TaskReport) -> str:
        """Format a TaskReport as compact JSON for the compression prompt."""
        payload = {
            "worker": report.worker_id,
            "status": report.status,
            "what_was_done": report.what_was_done,
            "results_table": report.results_table[:3] if report.results_table else [],
            "key_metrics": report.key_metrics,
            "artifacts": report.artifacts[:5] if report.artifacts else [],
            "verdict": report.verdict,
            "confidence": report.confidence,
            "error": report.error,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)
