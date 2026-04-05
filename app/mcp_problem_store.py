"""
MCP problem store -- persistence layer for MCP problem reports.

Handles saving, loading, and querying MCP problem reports from the fixes/ directory.
Uses atomic writes (tempfile + rename) matching the state_store.py pattern.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("orchestrator.mcp_problem_store")


@dataclass
class McpProblem:
    """A single MCP-related problem identified during worker execution or planner review."""

    problem_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cycle: int = 0
    task_id: str = ""
    worker_id: str = ""
    tool_name: str = ""
    problem_type: str = "unknown"  # contract_error | wrong_params | wrong_order | tool_failure | misuse | unknown
    description: str = ""
    evidence: str = ""
    suggestion: str = ""
    source: str = ""  # planner_review | worker_report
    severity: str = "medium"  # low | medium | high


class McpProblemStore:
    """Manages MCP problem report persistence in the fixes/ directory."""

    def __init__(self, fixes_dir: str | Path) -> None:
        self.fixes_dir = Path(fixes_dir)
        self.fixes_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self,
        problems: list[McpProblem],
        cycle: int,
    ) -> Path:
        """Save a timestamped problem report and update latest.json.

        Returns the path of the saved report.
        """
        if not problems:
            logger.debug("No MCP problems to save")
            return Path("")

        now = datetime.now(timezone.utc)
        report_data = {
            "generated_at": now.isoformat(),
            "cycle": cycle,
            "total_problems": len(problems),
            "problems": [asdict(p) for p in problems],
        }

        # Timestamped report
        ts = now.strftime("%Y%m%dT%H%M%S")
        report_path = self.fixes_dir / f"mcp_problems_{ts}.json"
        self._atomic_write(report_path, report_data)

        # latest.json (always mirrors the most recent report)
        latest_path = self.fixes_dir / "latest.json"
        self._atomic_write(latest_path, report_data)

        logger.info("Saved %d MCP problems to %s", len(problems), report_path.name)
        return report_path

    def load_latest(self) -> list[McpProblem] | None:
        """Load the most recent problem report from latest.json."""
        latest_path = self.fixes_dir / "latest.json"
        if not latest_path.exists():
            return None
        return self._load_report_file(latest_path)

    def list_reports(self) -> list[Path]:
        """Return sorted list of all timestamped report paths."""
        reports = sorted(self.fixes_dir.glob("mcp_problems_*.json"))
        return reports

    def format_summary_for_planner(self, problems: list[McpProblem], max_items: int = 10) -> str:
        """Format problems into a concise prompt section for the planner.

        Deduplicates by (tool_name, problem_type), keeping the most recent occurrence.
        """
        if not problems:
            return ""

        # Deduplicate: keep latest per (tool_name, problem_type)
        seen: dict[tuple[str, str], McpProblem] = {}
        for p in problems:
            key = (p.tool_name or "unknown", p.problem_type)
            seen[key] = p  # last one wins (most recent)

        unique = list(seen.values())[:max_items]
        lines: list[str] = []
        for p in unique:
            severity_tag = p.severity.upper()
            lines.append(
                f"- [{severity_tag}] {p.tool_name}: {p.description}"
            )
            if p.suggestion:
                lines.append(f"  Fix: {p.suggestion}")
            if p.evidence:
                lines.append(f"  Evidence: {p.evidence[:150]}")

        return "\n".join(lines)

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _load_report_file(self, path: Path) -> list[McpProblem] | None:
        """Load a report file and return a list of McpProblem objects."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load report %s: %s", path, e)
            return None

        raw_problems = data.get("problems", [])
        problems: list[McpProblem] = []
        for rp in raw_problems:
            if isinstance(rp, dict):
                problems.append(McpProblem(
                    problem_id=rp.get("problem_id", uuid.uuid4().hex[:12]),
                    timestamp=rp.get("timestamp", ""),
                    cycle=rp.get("cycle", 0),
                    task_id=rp.get("task_id", ""),
                    worker_id=rp.get("worker_id", ""),
                    tool_name=rp.get("tool_name", ""),
                    problem_type=rp.get("problem_type", "unknown"),
                    description=rp.get("description", ""),
                    evidence=rp.get("evidence", ""),
                    suggestion=rp.get("suggestion", ""),
                    source=rp.get("source", ""),
                    severity=rp.get("severity", "medium"),
                ))
        return problems

    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON data atomically using tempfile + rename."""
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.fixes_dir),
                prefix=".tmp_",
                suffix=".json",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
