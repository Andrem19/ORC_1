"""LM Studio assistant — log analysis and execution time prediction.

Uses the local LM Studio OpenAI-compatible API for two features:
- Feature 13: Periodic log analysis with diagnostic digest
- Feature 15: Execution time prediction for new research plans

Gracefully degrades: if LM Studio is unavailable, all methods return None
and the orchestrator continues normally.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from http.client import HTTPConnection
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from app.config import LMStudioConfig

if TYPE_CHECKING:
    from app.plan_models import ResearchPlan

logger = logging.getLogger("orchestrator.lmstudio_assistant")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogAnalysisResult:
    frequent_errors: list[str]
    error_patterns: list[str]
    avg_task_seconds: float
    trend: str  # "improving" | "degrading" | "stable"
    recommendations: list[str]
    raw_digest: str


@dataclass
class StagePrediction:
    stage_number: int
    stage_name: str
    estimated_minutes: float


@dataclass
class ExecutionPrediction:
    stage_predictions: list[StagePrediction]
    total_estimated_minutes: float
    raw_response: str


@dataclass
class TaskHistoryEntry:
    stage_number: int
    stage_name: str
    execution_minutes: float


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_LOG_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert DevOps analyst. Analyze the provided application log lines and produce a JSON diagnostic report.

Return ONLY a JSON object (no markdown, no code fences) with these fields:
{
  "frequent_errors": ["error description 1", "error description 2"],
  "error_patterns": ["pattern 1", "pattern 2"],
  "avg_task_seconds": 45.0,
  "trend": "improving|degrading|stable",
  "recommendations": ["rec 1", "rec 2"],
  "digest_summary": "2-3 sentence human-readable summary for the operator"
}

Rules:
- frequent_errors: top 3-5 most common error messages or error categories
- error_patterns: recurring patterns you notice (e.g. "MCP timeouts every 10 cycles")
- avg_task_seconds: estimate average task execution time from timestamps if visible
- trend: is the system improving (fewer errors over time), degrading, or stable?
- recommendations: actionable suggestions to fix issues
- digest_summary: concise paragraph suitable for a Telegram notification

If the log is too short or lacks enough data, still provide your best analysis with available information."""


_EXECUTION_PREDICTION_SYSTEM_PROMPT = """\
You are an expert project manager for automated trading research tasks.
Given historical execution times and a new research plan, predict how long each stage will take.

Return ONLY a JSON object (no markdown, no code fences) with these fields:
{
  "stage_predictions": [
    {"stage_number": 1, "stage_name": "Feature exploration", "estimated_minutes": 12.5},
    {"stage_number": 2, "stage_name": "Backtesting", "estimated_minutes": 18.0}
  ],
  "total_estimated_minutes": 45.5,
  "summary": "Stage 1 (~12 min) is quick exploration. Stage 2 (~18 min) involves heavy backtesting. \
Total plan estimated at ~45 min. Expected completion by 22:30."
}

Rules:
- Estimate based on similar past stages (similar names, similar complexity)
- If no history exists, estimate based on stage description complexity:
  simple stages = 5-10 min, medium = 10-20 min, complex (integration/backtesting) = 15-30 min
- Parallel stages should not be summed — use the longest parallel branch
- total_estimated_minutes should account for parallelism (sum of sequential stages only)
- summary should be a human-readable paragraph for Telegram notification
- Be realistic — don't underestimate"""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class LMStudioAssistant:
    """LM Studio-powered assistant for log analysis and execution prediction."""

    def __init__(self, config: LMStudioConfig) -> None:
        self.config = config
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
                logger.info("LM Studio assistant available at %s", self.config.base_url)
            self._available = ok
            self._available_checked = True
            return ok
        except Exception:
            self._available = False
            self._available_checked = True
            return False

    def analyze_log(self, log_lines: list[str]) -> LogAnalysisResult | None:
        """Analyze log lines and return a diagnostic result.

        Returns None if LM Studio is unavailable or the call fails.
        """
        if not log_lines:
            return None

        user_prompt = (
            f"Analyze these {len(log_lines)} log lines from an automated trading research orchestrator.\n"
            f"The orchestrator runs in cycles, dispatching research tasks to AI workers "
            f"that call MCP tools (backtests, features, models).\n\n"
            f"--- LOG START ---\n"
            + "\n".join(log_lines)
            + "\n--- LOG END ---"
        )

        raw = self._call_lmstudio(_LOG_ANALYSIS_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            return None

        return self._parse_log_analysis(raw)

    def predict_execution_time(
        self,
        history: list[TaskHistoryEntry],
        plan: ResearchPlan,
    ) -> ExecutionPrediction | None:
        """Predict execution time for each stage of a new plan.

        Returns None if LM Studio is unavailable or the call fails.
        """
        history_text = self._format_history(history)
        stages_text = self._format_plan_stages(plan)

        user_prompt = (
            f"## Task Execution History\n{history_text}\n\n"
            f"## New Research Plan (v{plan.version})\n{stages_text}\n\n"
            f"Predict execution time for each stage of this new plan."
        )

        raw = self._call_lmstudio(_EXECUTION_PREDICTION_SYSTEM_PROMPT, user_prompt)
        if raw is None:
            return None

        return self._parse_execution_prediction(raw)

    # ---------------------------------------------------------------
    # HTTP client (reuses pattern from ReportCompressorService)
    # ---------------------------------------------------------------

    def _call_lmstudio(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: int | None = None,
    ) -> str | None:
        """Call LM Studio chat completions API. Returns response text or None."""
        timeout = timeout or self.config.timeout_seconds

        body: dict = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
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
                timeout=timeout,
            )
            headers = {"Content-Type": "application/json"}
            conn.request("POST", "/v1/chat/completions", json.dumps(body), headers)
            resp = conn.getresponse()
            resp_body = resp.read().decode("utf-8")
            conn.close()

            if resp.status != 200:
                logger.warning(
                    "LM Studio returned HTTP %d: %s",
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
            logger.warning("LM Studio call failed: %s", e)
            return None

    # ---------------------------------------------------------------
    # Parsing helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract JSON from model response, handling markdown code fences."""
        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        return None

    def _parse_log_analysis(self, raw: str) -> LogAnalysisResult:
        """Parse LM Studio response into LogAnalysisResult."""
        data = self._extract_json(raw)

        if data is not None:
            return LogAnalysisResult(
                frequent_errors=data.get("frequent_errors", []),
                error_patterns=data.get("error_patterns", []),
                avg_task_seconds=float(data.get("avg_task_seconds", 0)),
                trend=data.get("trend", "stable"),
                recommendations=data.get("recommendations", []),
                raw_digest=data.get("digest_summary", raw[:1000]),
            )

        # Fallback: use raw text as digest
        return LogAnalysisResult(
            frequent_errors=[],
            error_patterns=[],
            avg_task_seconds=0,
            trend="stable",
            recommendations=[],
            raw_digest=raw[:1500],
        )

    def _parse_execution_prediction(self, raw: str) -> ExecutionPrediction:
        """Parse LM Studio response into ExecutionPrediction."""
        data = self._extract_json(raw)

        if data is not None:
            predictions = []
            for sp in data.get("stage_predictions", []):
                predictions.append(StagePrediction(
                    stage_number=int(sp.get("stage_number", 0)),
                    stage_name=str(sp.get("stage_name", "")),
                    estimated_minutes=float(sp.get("estimated_minutes", 0)),
                ))
            total = float(data.get("total_estimated_minutes", 0))
            summary = data.get("summary", raw[:1500])
            return ExecutionPrediction(
                stage_predictions=predictions,
                total_estimated_minutes=total,
                raw_response=summary,
            )

        # Fallback: use raw text
        return ExecutionPrediction(
            stage_predictions=[],
            total_estimated_minutes=0,
            raw_response=raw[:1500],
        )

    # ---------------------------------------------------------------
    # Formatting helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _format_history(history: list[TaskHistoryEntry]) -> str:
        """Format task execution history for the prediction prompt."""
        if not history:
            return "No previous execution history available."

        # Take last 30 entries
        entries = history[-30:]
        lines = [
            f"- Stage {e.stage_number} ({e.stage_name}): {e.execution_minutes:.1f} min"
            for e in entries
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_plan_stages(plan: ResearchPlan) -> str:
        """Format plan stages for the prediction prompt."""
        lines = []
        for task in plan.tasks:
            deps = f" (depends on stages {task.depends_on})" if task.depends_on else ""
            steps_count = len(task.steps)
            lines.append(
                f"- Stage {task.stage_number}: {task.stage_name}{deps} "
                f"({steps_count} steps)"
            )
            if task.theory:
                lines.append(f"  Theory: {task.theory[:200]}")
        return "\n".join(lines)
