"""
Persistence for the canonical markdown wave runtime.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.plan_models import Plan, PlanExecutionSlice, PlanReport
from app.run_context import ensure_current_run, read_current_run_id, resolve_run_dir

logger = logging.getLogger("orchestrator.plan_store")


class PlanStore:
    """Stores plans, plan reports, planner runs, and wave summaries."""

    def __init__(self, plans_dir: str = "plans", *, run_id: str = "") -> None:
        self.plans_dir = Path(plans_dir)
        self.run_id = run_id or read_current_run_id(self.plans_dir)
        self.reports_dir = self.active_root / "reports"
        self.planner_runs_dir = self.active_root / "planner_runs"
        self.wave_runs_dir = self.active_root / "waves"
        self.attempts_dir = self.active_root / "attempts"

    @property
    def active_root(self) -> Path:
        if not self.run_id:
            return self.plans_dir
        return ensure_current_run(self.plans_dir, self.run_id)

    def ensure_dirs(self) -> None:
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.active_root.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.planner_runs_dir.mkdir(parents=True, exist_ok=True)
        self.wave_runs_dir.mkdir(parents=True, exist_ok=True)
        self.attempts_dir.mkdir(parents=True, exist_ok=True)

    def save_plan(self, plan: Plan) -> Path:
        self.ensure_dirs()
        md_path = self.active_root / f"plan_v{plan.version}.md"
        meta_path = self.active_root / f"plan_v{plan.version}_meta.json"
        md_path.write_text(plan.markdown, encoding="utf-8")
        payload = asdict(plan)
        payload["markdown"] = ""
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Plan v%d saved", plan.version)
        return md_path

    def load_plan(self, version: int) -> Plan | None:
        meta_path = self._preferred_path(f"plan_v{version}_meta.json")
        if not meta_path.exists():
            return None
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["markdown"] = self.load_plan_markdown(version) or ""
            slices = payload.get("slices", [])
            if isinstance(slices, list):
                payload["slices"] = [
                    item if isinstance(item, PlanExecutionSlice) else PlanExecutionSlice(**item)
                    for item in slices
                    if isinstance(item, dict)
                ]
            filtered = {k: v for k, v in payload.items() if k in Plan.__dataclass_fields__}
            return Plan(**filtered)
        except Exception as exc:
            logger.error("Failed to load plan v%d: %s", version, exc)
            return None

    def load_plan_markdown(self, version: int) -> str | None:
        md_path = self._preferred_path(f"plan_v{version}.md")
        if not md_path.exists():
            return None
        return md_path.read_text(encoding="utf-8")

    def list_plan_versions(self) -> list[int]:
        versions: list[int] = []
        search_root = self.active_root if self.active_root.exists() else self.plans_dir
        if not search_root.exists():
            return versions
        for path in search_root.glob("plan_v*_meta.json"):
            try:
                versions.append(int(path.stem.replace("plan_v", "").replace("_meta", "")))
            except ValueError:
                continue
        return sorted(versions)

    def load_latest_plan(self) -> Plan | None:
        versions = self.list_plan_versions()
        if not versions:
            return None
        return self.load_plan(versions[-1])

    def save_report(self, report: PlanReport) -> Path:
        self.ensure_dirs()
        path = self.reports_dir / f"plan_v{report.plan_version}_report.json"
        path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Plan report saved for v%d", report.plan_version)
        return path

    def load_report(self, version: int) -> PlanReport | None:
        path = self._preferred_subpath("reports", f"plan_v{version}_report.json")
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            filtered = {k: v for k, v in payload.items() if k in PlanReport.__dataclass_fields__}
            return PlanReport(**filtered)
        except Exception as exc:
            logger.error("Failed to load report for v%d: %s", version, exc)
            return None

    def load_cumulative_findings(self) -> str:
        path = self.plans_dir / "cumulative_findings.txt"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def append_findings(self, text: str) -> None:
        self.ensure_dirs()
        path = self.plans_dir / "cumulative_findings.txt"
        existing = self.load_cumulative_findings()
        updated = (existing + "\n" + text).strip()
        if len(updated) > 3000:
            updated = updated[-3000:]
        path.write_text(updated, encoding="utf-8")

    def save_planner_run(
        self,
        *,
        request_type: str,
        request_version: int,
        attempt_number: int,
        execution_seq: int,
        payload: dict[str, Any],
    ) -> Path:
        self.ensure_dirs()
        filename = f"{request_type}_v{request_version}_attempt_{attempt_number}_exec_{execution_seq}.json"
        path = self.planner_runs_dir / filename
        normalized = self._externalize_planner_payload(filename=filename, payload=payload)
        path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            "Planner run saved: type=%s v%d attempt=%d exec=%d",
            request_type,
            request_version,
            attempt_number,
            execution_seq,
        )
        return path

    def save_wave_summary(self, *, wave_id: int, payload: dict[str, Any]) -> Path:
        self.ensure_dirs()
        path = self.wave_runs_dir / f"wave_{wave_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wave summary saved for wave %d", wave_id)
        return path

    def load_wave_summary(self, wave_id: int) -> dict[str, Any] | None:
        path = self.wave_runs_dir / f"wave_{wave_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to load wave %d: %s", wave_id, exc)
            return None

    def load_latest_wave_summary(self) -> dict[str, Any] | None:
        if not self.wave_runs_dir.exists():
            return None
        wave_ids: list[int] = []
        for path in self.wave_runs_dir.glob("wave_*.json"):
            try:
                wave_ids.append(int(path.stem.replace("wave_", "")))
            except ValueError:
                continue
        if not wave_ids:
            return None
        return self.load_wave_summary(max(wave_ids))

    def clear_all(self) -> None:
        targets = list(self.plans_dir.glob("plan_v*.md")) + list(self.plans_dir.glob("plan_v*_meta.json"))
        if self.plans_dir.exists():
            for subdir in ("reports", "planner_runs", "waves", "attempts", "runs"):
                path = self.plans_dir / subdir
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        path.unlink(missing_ok=True)
        findings_path = self.plans_dir / "cumulative_findings.txt"
        if findings_path.exists():
            targets.append(findings_path)
        current_run_path = self.plans_dir / "current_run.json"
        if current_run_path.exists():
            targets.append(current_run_path)
        current_link = self.plans_dir / "current"
        if current_link.exists() or current_link.is_symlink():
            targets.append(current_link)
        for path in targets:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def archive_to(self, target_dir: Path) -> int:
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for path in self.plans_dir.glob("plan_v*"):
            shutil.copy2(path, target_dir / path.name)
            count += 1
        for subdir_name in ("reports", "planner_runs", "waves"):
            src_dir = self.plans_dir / subdir_name
            if not src_dir.exists():
                continue
            dst_dir = target_dir / subdir_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for path in src_dir.glob("*.json"):
                shutil.copy2(path, dst_dir / path.name)
                count += 1
        findings_path = self.plans_dir / "cumulative_findings.txt"
        if findings_path.exists():
            shutil.copy2(findings_path, target_dir / findings_path.name)
            count += 1
        return count

    def save_worker_attempt_artifact(
        self,
        *,
        task_id: str,
        plan_version: int,
        payload: dict[str, Any],
    ) -> Path:
        self.ensure_dirs()
        safe_task = task_id.replace("/", "_")
        filename = f"plan_v{plan_version}_{safe_task}.json"
        path = self.attempts_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _preferred_path(self, filename: str) -> Path:
        candidate = self.active_root / filename
        if candidate.exists():
            return candidate
        return self.plans_dir / filename

    def _preferred_subpath(self, dirname: str, filename: str) -> Path:
        candidate = self.active_root / dirname / filename
        if candidate.exists():
            return candidate
        return self.plans_dir / dirname / filename

    def _externalize_planner_payload(self, *, filename: str, payload: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(payload)
        transcripts_dir = self.planner_runs_dir / "transcripts"
        transcripts_dir.mkdir(parents=True, exist_ok=True)
        for field_name in ("raw_stdout", "raw_stderr"):
            raw = str(normalized.get(field_name, "") or "")
            if not raw:
                normalized[f"{field_name}_chars"] = 0
                continue
            ref = transcripts_dir / f"{Path(filename).stem}_{field_name}.log"
            ref.write_text(raw, encoding="utf-8")
            normalized[f"{field_name}_chars"] = len(raw)
            normalized[f"{field_name}_ref"] = str(ref)
            normalized[f"{field_name}_preview"] = raw[:1200]
            normalized[f"{field_name}_tail"] = raw[-1200:]
            normalized[field_name] = _compact_preview(raw)
        rendered = str(normalized.get("rendered_output", "") or "")
        normalized["rendered_output_chars"] = len(rendered)
        if len(rendered) > 6000:
            normalized["rendered_output"] = _compact_preview(rendered)
        normalized["saved_at"] = datetime.now(timezone.utc).isoformat()
        return normalized


def _compact_preview(text: str, *, head: int = 1200, tail: int = 800) -> str:
    if len(text) <= head + tail + 32:
        return text
    return f"{text[:head]}\n\n--- [TRUNCATED] ---\n\n{text[-tail:]}"
