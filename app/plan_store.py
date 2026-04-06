"""
Plan persistence — dual-format storage for research plans.

Saves plans as:
  - Markdown: plans/plan_v1.md (human-readable, feeds back into planner)
  - JSON: plans/plan_v1.json (machine-readable, for state restoration)
  - Reports: plans/reports/{task_id}.json
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.plan_models import ResearchPlan, TaskReport

logger = logging.getLogger("orchestrator.plan_store")


class PlanStore:
    """Manages plan and report persistence on disk."""

    def __init__(self, plans_dir: str = "plans") -> None:
        self.plans_dir = Path(plans_dir)
        self.reports_dir = self.plans_dir / "reports"
        self.rejected_dir = self.plans_dir / "rejected"
        self.planner_runs_dir = self.plans_dir / "planner_runs"

    def ensure_dirs(self) -> None:
        """Create plan directories if they don't exist."""
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_dir.mkdir(parents=True, exist_ok=True)
        self.planner_runs_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Plan persistence
    # ---------------------------------------------------------------

    def save_plan(self, plan: ResearchPlan) -> Path:
        """Save plan as both markdown and JSON. Returns the markdown path."""
        self.ensure_dirs()

        md_path = self.plans_dir / f"plan_v{plan.version}.md"
        json_path = self.plans_dir / f"plan_v{plan.version}.json"

        # Save markdown (the planner's raw output, or rendered)
        md_content = plan.plan_markdown or _render_plan_markdown(plan)
        md_path.write_text(md_content, encoding="utf-8")

        # Save JSON (full structured data, but WITHOUT plan_markdown to save space)
        json_data = _plan_to_dict(plan)
        json_data["plan_markdown"] = ""  # stored separately as .md
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        logger.info("Plan v%d saved to %s (%d chars) and %s",
                     plan.version, md_path, len(md_content), json_path)
        return md_path

    def load_plan(self, version: int) -> ResearchPlan | None:
        """Load a plan by version number (from JSON, restores markdown from .md)."""
        json_path = self.plans_dir / f"plan_v{version}.json"
        if not json_path.exists():
            return None
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            plan = _plan_from_dict(data)
            # Restore plan_markdown from the separate .md file
            if not plan.plan_markdown:
                md = self.load_plan_markdown(version)
                if md:
                    plan.plan_markdown = md
            return plan
        except Exception as e:
            logger.error("Failed to load plan v%d: %s", version, e)
            return None

    def load_latest_plan(self) -> ResearchPlan | None:
        """Load the most recent plan version."""
        versions = self.list_plan_versions()
        if not versions:
            return None
        return self.load_plan(max(versions))

    def list_plan_versions(self) -> list[int]:
        """List all plan version numbers available on disk, sorted ascending."""
        versions: list[int] = []
        if not self.plans_dir.exists():
            return versions
        for p in self.plans_dir.glob("plan_v*.json"):
            try:
                ver = int(p.stem.replace("plan_v", ""))
                versions.append(ver)
            except ValueError:
                continue
        return sorted(versions)

    def load_plan_markdown(self, version: int) -> str | None:
        """Load the markdown text of a plan version."""
        md_path = self.plans_dir / f"plan_v{version}.md"
        if not md_path.exists():
            return None
        return md_path.read_text(encoding="utf-8")

    def load_all_anti_patterns(self) -> list[dict[str, Any]]:
        """Load anti-patterns from all plan versions (for planner context)."""
        all_patterns: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for ver in self.list_plan_versions():
            plan = self.load_plan(ver)
            if plan is None:
                continue
            for ap in plan.anti_patterns:
                if ap.pattern_id not in seen_ids:
                    seen_ids.add(ap.pattern_id)
                    all_patterns.append(asdict(ap))
        return all_patterns

    def load_cumulative_summary(self) -> str:
        """Load the cumulative summary from the latest plan."""
        versions = self.list_plan_versions()
        if not versions:
            return ""
        plan = self.load_plan(max(versions))
        if plan is None:
            return ""
        return plan.cumulative_summary

    # ---------------------------------------------------------------
    # Report persistence
    # ---------------------------------------------------------------

    def save_report(self, report: TaskReport) -> Path:
        """Save a task report."""
        self.ensure_dirs()
        path = self.reports_dir / f"{report.task_id}.json"
        path.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Report saved for task %s (plan v%d)", report.task_id, report.plan_version)
        return path

    def save_rejected_plan_attempt(
        self,
        plan_version: int,
        attempt_number: int,
        attempt_type: str,
        raw_output: str,
        parsed_data: dict[str, Any],
        validation_errors: list[dict[str, Any]],
        planner_run_artifact: str | None = None,
    ) -> Path:
        """Persist one invalid planner attempt for later debugging."""
        self.ensure_dirs()
        path = self.rejected_dir / f"plan_v{plan_version}_attempt_{attempt_number}.json"
        payload = {
            "plan_version": plan_version,
            "attempt_number": attempt_number,
            "attempt_type": attempt_type,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "raw_output": raw_output,
            "parsed_data": parsed_data,
            "validation_errors": validation_errors,
            "planner_run_artifact": planner_run_artifact,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Rejected plan attempt saved: v%d attempt=%d type=%s",
            plan_version, attempt_number, attempt_type,
        )
        return path

    def save_planner_run(
        self,
        *,
        request_type: str,
        request_version: int,
        attempt_number: int,
        payload: dict[str, Any],
    ) -> Path:
        """Persist one planner execution trace for diagnostics."""
        self.ensure_dirs()
        filename = f"{request_type}_v{request_version}_attempt_{attempt_number}.json"
        path = self.planner_runs_dir / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Planner run saved: type=%s v%d attempt=%d",
            request_type, request_version, attempt_number,
        )
        return path

    def load_report(self, task_id: str) -> TaskReport | None:
        """Load a task report by task_id."""
        path = self.reports_dir / f"{task_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return TaskReport(**data)
        except Exception as e:
            logger.error("Failed to load report for task %s: %s", task_id, e)
            return None

    def load_reports_for_plan(self, version: int) -> list[TaskReport]:
        """Load all reports for a specific plan version."""
        reports: list[TaskReport] = []
        if not self.reports_dir.exists():
            return reports
        for path in self.reports_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                report = TaskReport(**data)
                if report.plan_version == version:
                    reports.append(report)
            except Exception:
                continue
        return reports

    # ---------------------------------------------------------------
    # Reset support
    # ---------------------------------------------------------------

    def clear_all(self) -> None:
        """Remove all plan files and reports from disk."""
        for p in list(self.plans_dir.glob("plan_v*")):
            p.unlink()
            logger.info("Removed plan file: %s", p)
        if self.reports_dir.exists():
            for p in list(self.reports_dir.glob("*.json")):
                p.unlink()
                logger.info("Removed report: %s", p)
        if self.rejected_dir.exists():
            for p in list(self.rejected_dir.glob("*.json")):
                p.unlink()
                logger.info("Removed rejected plan artifact: %s", p)
        if self.planner_runs_dir.exists():
            for p in list(self.planner_runs_dir.glob("*.json")):
                p.unlink()
                logger.info("Removed planner run artifact: %s", p)

    def archive_to(self, target_dir: Path) -> int:
        """Copy all plan files and reports to target_dir. Returns file count."""
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for p in self.plans_dir.glob("plan_v*"):
            shutil.copy2(p, target_dir / p.name)
            count += 1
        reports_target = target_dir / "reports"
        if self.reports_dir.exists():
            reports_target.mkdir(parents=True, exist_ok=True)
            for p in self.reports_dir.glob("*.json"):
                shutil.copy2(p, reports_target / p.name)
                count += 1
        rejected_target = target_dir / "rejected"
        if self.rejected_dir.exists():
            rejected_target.mkdir(parents=True, exist_ok=True)
            for p in self.rejected_dir.glob("*.json"):
                shutil.copy2(p, rejected_target / p.name)
                count += 1
        planner_runs_target = target_dir / "planner_runs"
        if self.planner_runs_dir.exists():
            planner_runs_target.mkdir(parents=True, exist_ok=True)
            for p in self.planner_runs_dir.glob("*.json"):
                shutil.copy2(p, planner_runs_target / p.name)
                count += 1
        return count


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _plan_to_dict(plan: ResearchPlan) -> dict[str, Any]:
    """Serialize a ResearchPlan to a plain dict."""
    return asdict(plan)


def _plan_from_dict(data: dict[str, Any]) -> ResearchPlan:
    """Deserialize a ResearchPlan from a plain dict."""
    # Rebuild nested objects
    tasks = []
    for t_data in data.get("tasks", []):
        gates = []
        for g_data in t_data.get("decision_gates", []):
            gates.append(_dict_to_dataclass(g_data))
        t_data["decision_gates"] = gates
        depends_on = t_data.get("depends_on", [])
        if not isinstance(depends_on, list):
            depends_on = []
        t_data["depends_on"] = [int(dep) for dep in depends_on if isinstance(dep, int)]
        t_data["steps"] = t_data.get("steps", [])
        t_data["results_table_rows"] = t_data.get("results_table_rows", [])
        t_data["results_table_columns"] = t_data.get("results_table_columns", [])
        t_data["agent_instructions"] = t_data.get("agent_instructions", [])
        # Ensure status is valid
        status_str = t_data.get("status", "pending")
        try:
            from app.models import TaskStatus
            t_data["status"] = TaskStatus(status_str)
        except ValueError:
            t_data["status"] = TaskStatus.PENDING
        tasks.append(_dict_to_dataclass(t_data, target_class_name="PlanTask"))

    anti_patterns = []
    for ap_data in data.get("anti_patterns", []):
        anti_patterns.append(_dict_to_dataclass(ap_data, target_class_name="AntiPattern"))

    data["tasks"] = tasks
    data["anti_patterns"] = anti_patterns
    data["schema_version"] = int(data.get("schema_version", 1) or 1)
    data["principles"] = data.get("principles", [])
    data["execution_order"] = data.get("execution_order", [])
    baseline_metrics = data.get("baseline_metrics", {})
    data["baseline_metrics"] = baseline_metrics if isinstance(baseline_metrics, dict) else {}

    # Remove fields that don't exist in the dataclass
    valid_keys = {f.name for f in __import__("dataclasses").fields(ResearchPlan)}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    return ResearchPlan(**filtered)


def _dict_to_dataclass(data: dict[str, Any], target_class_name: str = "") -> Any:
    """Best-effort dict → dataclass conversion."""
    if target_class_name == "PlanTask":
        from app.plan_models import PlanTask, DecisionGate, PlanStep
        gates = []
        for g in data.get("decision_gates", []):
            if isinstance(g, dict):
                gates.append(DecisionGate(**g))
            elif isinstance(g, DecisionGate):
                gates.append(g)
        steps = []
        for s in data.get("steps", []):
            if isinstance(s, dict):
                steps.append(PlanStep(**{
                    "step_id": s.get("step_id", ""),
                    "kind": s.get("kind", "work"),
                    "instruction": s.get("instruction", ""),
                    "tool_name": s.get("tool_name"),
                    "args": s.get("args", {}) if isinstance(s.get("args"), dict) else {},
                    "binds": s.get("binds", []) if isinstance(s.get("binds"), list) else [],
                    "decision_outputs": s.get("decision_outputs", []) if isinstance(s.get("decision_outputs"), list) else [],
                    "notes": s.get("notes", ""),
                }))
            elif isinstance(s, PlanStep):
                steps.append(s)
        valid_keys = {f.name for f in __import__("dataclasses").fields(PlanTask)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        filtered["decision_gates"] = gates
        filtered["steps"] = steps
        return PlanTask(**filtered)
    elif target_class_name == "AntiPattern":
        from app.plan_models import AntiPattern
        valid_keys = {f.name for f in __import__("dataclasses").fields(AntiPattern)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return AntiPattern(**filtered)
    else:
        # DecisionGate or simple dataclass
        from app.plan_models import DecisionGate
        return DecisionGate(**data)


def _render_plan_markdown(plan: ResearchPlan) -> str:
    """Render a ResearchPlan into markdown (fallback if plan_markdown is empty)."""
    parts: list[str] = []
    parts.append(f"# Plan v{plan.version}")
    parts.append("")
    parts.append(f"**Created**: {plan.created_at}")
    if plan.frozen_base:
        parts.append(f"**Frozen base**: {plan.frozen_base}")
    parts.append(f"**Goal**: {plan.goal}")
    parts.append("")

    if plan.cumulative_summary:
        parts.append("## Cumulative Summary")
        parts.append(plan.cumulative_summary)
        parts.append("")

    if plan.anti_patterns:
        parts.append("## Anti-Patterns (do NOT repeat)")
        for ap in plan.anti_patterns:
            parts.append(f"- **{ap.category}**: {ap.description} ({ap.evidence_count} failures)")
        parts.append("")

    for task in plan.tasks:
        parts.append(f"## ETAP {task.stage_number}: {task.stage_name}")
        parts.append("")
        if task.theory:
            parts.append("### Theory")
            parts.append(task.theory)
            parts.append("")
        parts.append("### Instructions")
        for i, step in enumerate(task.normalized_steps(), 1):
            parts.append(f"{i}. [{step.step_id}] {step.kind}")
            for line in step.render_prompt_block()[1:]:
                parts.append(f"   - {line}")
        if task.results_table_columns:
            parts.append("")
            parts.append("### Results")
            cols = task.results_table_columns
            parts.append("| " + " | ".join(cols) + " |")
            parts.append("| " + " | ".join("---" for _ in cols) + " |")
            for row in task.results_table_rows:
                parts.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        parts.append("")

    return "\n".join(parts)
