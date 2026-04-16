"""
Persistence for compiled raw-plan sequences and plannerless runtime manifests.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice
from app.raw_plan_models import CompiledPlanManifest, CompiledPlanSequence
from app.raw_plan_ordering import raw_plan_sort_key
from app.services.direct_execution.budgeting import normalize_plan_budgets


class CompiledPlanStore:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir)

    def save_sequence(self, sequence: CompiledPlanSequence) -> Path:
        target_dir = self.root_dir / Path(sequence.source_file).stem
        target_dir.mkdir(parents=True, exist_ok=True)
        semantic_path = target_dir / "semantic.json"
        report_path = target_dir / "compile_report.json"
        plans_dir = target_dir / "plans"
        plans_dir.mkdir(parents=True, exist_ok=True)
        plan_files: list[str] = []
        for plan in sequence.plans:
            plan_path = plans_dir / f"{plan.plan_id}.json"
            plan_path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2), encoding="utf-8")
            plan_files.append(str(plan_path.relative_to(target_dir)))
        semantic_payload = asdict(sequence.semantic_plan) if sequence.semantic_plan is not None else None
        semantic_path.write_text(json.dumps(semantic_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(asdict(sequence.report), ensure_ascii=False, indent=2), encoding="utf-8")
        manifest = CompiledPlanManifest(
            source_file=sequence.source_file,
            source_hash=sequence.source_hash,
            compiled_at=sequence.report.compiled_at,
            sequence_id=sequence.sequence_id,
            mcp_catalog_hash=sequence.report.mcp_catalog_hash,
            compile_status=sequence.report.compile_status,
            warnings=list(sequence.report.warnings),
            semantic_method=sequence.report.semantic_method,
            semantic_path=str(semantic_path.relative_to(target_dir)),
            compile_report_path=str(report_path.relative_to(target_dir)),
            plan_files=plan_files,
        )
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
        return manifest_path

    def load_manifests(self) -> list[CompiledPlanManifest]:
        manifests: list[CompiledPlanManifest] = []
        if not self.root_dir.exists():
            return manifests
        for path in sorted(self.root_dir.glob("*/manifest.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            manifests.append(CompiledPlanManifest(**payload))
        manifests.sort(key=lambda item: raw_plan_sort_key(item.source_file))
        return manifests

    def load_plan(self, manifest: CompiledPlanManifest, plan_file: str) -> ExecutionPlan:
        manifest_dir = self.root_dir / Path(manifest.source_file).stem
        payload = json.loads((manifest_dir / plan_file).read_text(encoding="utf-8"))
        plan = _deserialize_plan(payload)
        semantic_rel = manifest.semantic_path or "semantic.json"
        compile_report_rel = manifest.compile_report_path or "compile_report.json"
        plan.plan_source_kind = "compiled_raw"
        plan.source_sequence_id = manifest.sequence_id
        plan.source_raw_plan = manifest.source_file
        plan.source_manifest_path = str((manifest_dir / "manifest.json"))
        plan.source_semantic_path = str((manifest_dir / semantic_rel)) if (manifest_dir / semantic_rel).exists() else ""
        plan.source_compile_report_path = str((manifest_dir / compile_report_rel)) if (manifest_dir / compile_report_rel).exists() else ""
        if not plan.sequence_batch_index:
            plan.sequence_batch_index = _batch_index_for(plan.plan_id)
        _backfill_slice_dependencies(plan, manifest_dir=manifest_dir, semantic_path=semantic_rel)
        return normalize_plan_budgets(plan)


def _deserialize_plan(payload: dict[str, object]) -> ExecutionPlan:
    baseline_payload = dict(payload.get("baseline_ref", {}) or {})
    baseline = BaselineRef(
        snapshot_id=str(baseline_payload.get("snapshot_id", "") or ""),
        version=baseline_payload.get("version", 1),
        symbol=str(baseline_payload.get("symbol", "BTCUSDT") or "BTCUSDT"),
        anchor_timeframe=str(baseline_payload.get("anchor_timeframe", "1h") or "1h"),
        execution_timeframe=str(baseline_payload.get("execution_timeframe", "5m") or "5m"),
    )
    slices = [
        PlanSlice(**{
            key: value
            for key, value in dict(item).items()
            if key in PlanSlice.__dataclass_fields__
        })
        for item in payload.get("slices", []) or []
        if isinstance(item, dict)
    ]
    return ExecutionPlan(
        plan_id=str(payload.get("plan_id", "") or ""),
        goal=str(payload.get("goal", "") or ""),
        baseline_ref=baseline,
        global_constraints=[str(item) for item in payload.get("global_constraints", []) or []],
        slices=slices,
        mcp_catalog_hash=str(payload.get("mcp_catalog_hash", "") or ""),
        plan_source_kind=str(payload.get("plan_source_kind", "planner") or "planner"),
        source_sequence_id=str(payload.get("source_sequence_id", "") or ""),
        source_raw_plan=str(payload.get("source_raw_plan", "") or ""),
        source_manifest_path=str(payload.get("source_manifest_path", "") or ""),
        source_semantic_path=str(payload.get("source_semantic_path", "") or ""),
        source_compile_report_path=str(payload.get("source_compile_report_path", "") or ""),
        sequence_batch_index=int(payload.get("sequence_batch_index", 0) or 0),
        status=str(payload.get("status", "draft") or "draft"),
        created_at=str(payload.get("created_at", "") or ""),
        updated_at=str(payload.get("updated_at", "") or ""),
    )


def _batch_index_for(plan_id: str) -> int:
    suffix = plan_id.rsplit("_batch_", 1)
    if len(suffix) != 2:
        return 0
    try:
        return int(suffix[1])
    except ValueError:
        return 0


def _backfill_slice_dependencies(plan: ExecutionPlan, *, manifest_dir: Path, semantic_path: str) -> None:
    if not semantic_path:
        return
    semantic_file = manifest_dir / semantic_path
    if not semantic_file.exists():
        return
    try:
        payload = json.loads(semantic_file.read_text(encoding="utf-8"))
    except Exception:
        return
    stages = payload.get("stages", []) if isinstance(payload, dict) else []
    if not isinstance(stages, list):
        return
    dependency_map: dict[str, list[str]] = {}
    for item in stages:
        if not isinstance(item, dict):
            continue
        stage_id = str(item.get("stage_id", "") or "").strip()
        depends_on = [str(dep).strip() for dep in item.get("depends_on", []) or [] if str(dep).strip()]
        if stage_id:
            dependency_map[stage_id] = depends_on
    sequence_prefix = f"{plan.source_sequence_id}_"
    for slice_obj in plan.slices:
        if slice_obj.depends_on:
            continue
        if not slice_obj.slice_id.startswith(sequence_prefix):
            continue
        stage_id = slice_obj.slice_id[len(sequence_prefix):]
        stage_dependencies = dependency_map.get(stage_id, [])
        if stage_dependencies:
            slice_obj.depends_on = [f"{plan.source_sequence_id}_{dep}" for dep in stage_dependencies]
