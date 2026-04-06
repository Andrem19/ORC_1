"""
MCP problem scanner -- periodic review of MCP tool problems.

Determines when a review is due, builds the review prompt,
calls the planner to analyze results, and processes the response.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from app.adapters.base import BaseAdapter
from app.mcp_problem_store import McpProblem, McpProblemStore
from app.models import OrchestratorState, TaskResult
from app.result_parser import _extract_json_block

logger = logging.getLogger("orchestrator.mcp_problem_scanner")


# ---------------------------------------------------------------------------
# Review prompt template
# ---------------------------------------------------------------------------

MCP_REVIEW_PROMPT = """You are reviewing recent worker results to identify problems related to MCP dev_space1 tool usage.

## Recent Worker Results (last {count} results)
{formatted_results}

## Known MCP Tool Constraints
- anchor_timeframe is POLICY-LOCKED to '1h' — no other value accepted
- execution_timeframe is POLICY-LOCKED to '5m' — no other value accepted
- Classifier columns (cl_*) require force=True for rebuild
- backtests_plan should be called before backtests_runs
- features_custom requires validate before publish
- Async tools need polling via action='status' with job_id
- models_dataset -> models_train -> models_registry is the required workflow order

## Task
Identify any results where the worker had problems with MCP dev_space1 tools. Look for:
- Wrong parameters (e.g., wrong timeframe, missing required fields)
- Wrong tool call order (e.g., running backtest without plan first)
- Tool failures (MCP server errors, timeouts)
- Misunderstood contracts (e.g., using wrong action values, missing required params)
- Any error messages mentioning MCP tools

Output ONLY a JSON object with this schema:
{{
  "problems": [
    {{
      "tool_name": "MCP tool name involved",
      "problem_type": "contract_error|wrong_params|wrong_order|tool_failure|misuse|unknown",
      "description": "clear description of what went wrong",
      "evidence": "relevant error or output excerpt (keep short)",
      "suggestion": "how to avoid this problem in future tasks",
      "severity": "low|medium|high"
    }}
  ]
}}

If no MCP problems are present in the results, return: {{"problems": []}}"""


class McpProblemScanner:
    """Periodic scanner that asks the planner to review worker results for MCP problems."""

    def __init__(
        self,
        planner_adapter: BaseAdapter,
        problem_store: McpProblemStore,
        review_every_n_cycles: int = 10,
        review_after_n_errors: int = 3,
        planner_timeout: int = 180,
    ) -> None:
        self.planner_adapter = planner_adapter
        self.problem_store = problem_store
        self.review_every_n_cycles = review_every_n_cycles
        self.review_after_n_errors = review_after_n_errors
        self.planner_timeout = planner_timeout

        self._last_review_cycle: int = 0
        self._mcp_errors_since_review: int = 0

    def should_review(self, state: OrchestratorState, new_results: list[TaskResult]) -> bool:
        """Check if a periodic MCP problem review should be triggered.

        Triggers when:
        - Cycles since last review >= review_every_n_cycles
        - OR MCP-related errors since last review >= review_after_n_errors
        """
        cycles_since = state.current_cycle - self._last_review_cycle
        if cycles_since >= self.review_every_n_cycles:
            return True

        # Count MCP-related errors in new results
        for r in new_results:
            if r.status in ("error", "partial") and self._looks_mcp_related(r):
                self._mcp_errors_since_review += 1

        if self._mcp_errors_since_review >= self.review_after_n_errors:
            return True

        return False

    def extract_worker_problems(self, result: TaskResult) -> list[McpProblem]:
        """Extract problems reported by the worker in its mcp_problems field."""
        if not result.mcp_problems:
            return []

        problems: list[McpProblem] = []
        for raw in result.mcp_problems:
            if not isinstance(raw, dict):
                continue
            problems.append(McpProblem(
                problem_id=uuid.uuid4().hex[:12],
                timestamp=datetime.now(timezone.utc).isoformat(),
                cycle=0,  # filled by caller
                task_id=result.task_id,
                worker_id=result.worker_id,
                tool_name=raw.get("tool_name", ""),
                problem_type=raw.get("problem_type", "unknown"),
                description=str(raw.get("description", ""))[:500],
                evidence=str(raw.get("evidence", ""))[:500],
                suggestion=str(raw.get("suggestion", ""))[:300],
                source="worker_report",
                severity=raw.get("severity", "medium"),
            ))
        return problems

    def run_review(
        self,
        state: OrchestratorState,
        recent_results: list[TaskResult],
    ) -> list[McpProblem]:
        """Run a full MCP problem review cycle.

        1. Build review prompt from recent results
        2. Call planner adapter
        3. Parse response
        4. Merge with worker-reported problems
        5. Save report
        """
        # Build prompt
        prompt = self._build_review_prompt(recent_results)

        # Call planner
        logger.info("Running MCP problem review at cycle %d", state.current_cycle)
        response = self.planner_adapter.invoke(prompt, timeout=self.planner_timeout)

        if not response.success:
            logger.warning("MCP review planner call failed: %s", response.error[:200])
            return []

        # Parse response
        planner_problems = self._parse_review_response(
            response.raw_output,
            cycle=state.current_cycle,
        )

        # Also collect worker-reported problems from recent results
        worker_problems: list[McpProblem] = []
        for r in recent_results:
            for p in self.extract_worker_problems(r):
                p.cycle = state.current_cycle
                worker_problems.append(p)

        all_problems = planner_problems + worker_problems

        # Save
        if all_problems:
            self.problem_store.save_report(all_problems, cycle=state.current_cycle)

        # Reset counters
        self._last_review_cycle = state.current_cycle
        self._mcp_errors_since_review = 0

        logger.info(
            "MCP review found %d problems (%d from planner, %d from workers)",
            len(all_problems), len(planner_problems), len(worker_problems),
        )
        return all_problems

    def get_context_for_planner(self, max_items: int = 10) -> str:
        """Get a formatted problem summary for injection into the planner prompt."""
        problems = self.problem_store.load_latest()
        if not problems:
            return ""
        return self.problem_store.format_summary_for_planner(problems, max_items=max_items)

    # ---------------------------------------------------------------
    # Heuristic structural problem detection
    # ---------------------------------------------------------------

    def detect_structural_problems(self, state: OrchestratorState) -> list[McpProblem]:
        """Detect structural problems using heuristics (no LLM call needed)."""
        problems: list[McpProblem] = []
        now = datetime.now(timezone.utc).isoformat()

        # 1. Repeated failures: same error pattern 3+ times in a row
        recent_errors = [
            r for r in state.results[-10:]
            if r.status == "error" and r.error
        ]
        if len(recent_errors) >= 3:
            common = recent_errors[-1].error[:30]
            if all(common in other.error for other in recent_errors[-3:]):
                problems.append(McpProblem(
                    problem_id=uuid.uuid4().hex[:12],
                    timestamp=now,
                    cycle=state.current_cycle,
                    task_id="",
                    worker_id="",
                    tool_name="worker",
                    problem_type="repeated_failure",
                    description=f"Same error repeated 3+ times: '{common}...'",
                    evidence="; ".join(r.error[:60] for r in recent_errors[-3:])[:300],
                    suggestion="The approach causing this error should be abandoned or modified",
                    source="heuristic",
                    severity="high",
                ))

        # 2. Empty cycle spiral: many empty cycles without progress
        if state.empty_cycles >= 8:
            problems.append(McpProblem(
                problem_id=uuid.uuid4().hex[:12],
                timestamp=now,
                cycle=state.current_cycle,
                task_id="",
                worker_id="",
                tool_name="orchestrator",
                problem_type="empty_spiral",
                description=f"Orchestrator has had {state.empty_cycles} empty cycles without progress",
                evidence=f"empty_cycles={state.empty_cycles}, total_errors={state.total_errors}",
                suggestion="Consider changing research direction or checking if workers are functional",
                source="heuristic",
                severity="medium",
            ))

        return problems

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _build_review_prompt(self, results: list[TaskResult]) -> str:
        """Build the review prompt from recent results."""
        formatted: list[str] = []
        for r in results[-20:]:  # limit to last 20 results
            lines = [
                f"Task {r.task_id} by {r.worker_id}: status={r.status}",
            ]
            if r.summary:
                lines.append(f"  Summary: {r.summary[:300]}")
            if r.error:
                lines.append(f"  Error: {r.error[:300]}")
            if r.raw_output:
                # Include a snippet of raw output for error diagnosis
                lines.append(f"  Raw: {r.raw_output[:200]}")
            formatted.append("\n".join(lines))

        return MCP_REVIEW_PROMPT.format(
            count=len(results[-20:]),
            formatted_results="\n\n".join(formatted) if formatted else "(no results)",
        )

    def _parse_review_response(
        self,
        raw: str,
        cycle: int,
    ) -> list[McpProblem]:
        """Parse the planner's review response into McpProblem objects."""
        json_str = _extract_json_block(raw)
        if not json_str:
            logger.warning("MCP review: no parseable JSON in planner response")
            return []

        try:
            data: dict[str, Any] = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("MCP review: invalid JSON: %s", e)
            return []

        raw_problems = data.get("problems", [])
        if not isinstance(raw_problems, list):
            logger.warning("MCP review: 'problems' is not a list")
            return []

        now = datetime.now(timezone.utc).isoformat()
        problems: list[McpProblem] = []
        for rp in raw_problems:
            if not isinstance(rp, dict):
                continue
            problems.append(McpProblem(
                problem_id=uuid.uuid4().hex[:12],
                timestamp=now,
                cycle=cycle,
                task_id=rp.get("task_id", ""),
                worker_id=rp.get("worker_id", ""),
                tool_name=str(rp.get("tool_name", "")),
                problem_type=str(rp.get("problem_type", "unknown")),
                description=str(rp.get("description", ""))[:500],
                evidence=str(rp.get("evidence", ""))[:500],
                suggestion=str(rp.get("suggestion", ""))[:300],
                source="planner_review",
                severity=str(rp.get("severity", "medium")),
            ))
        return problems

    @staticmethod
    def _looks_mcp_related(result: TaskResult) -> bool:
        """Heuristic: check if an error result is likely MCP-related."""
        text = f"{result.error} {result.summary} {result.raw_output}".lower()
        mcp_keywords = [
            "mcp", "backtest", "snapshot", "feature", "model",
            "strategy", "dataset", "signal", "timeout", "tool",
            "plan", "run", "publish", "validate", "train",
        ]
        return any(kw in text for kw in mcp_keywords)
