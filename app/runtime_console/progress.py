"""
Helpers for operator-facing sequence progress in the runtime console.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.execution_models import ExecutionPlan
from app.runtime_console.models import BatchProgressView, SequenceProgressView


def load_total_batches(*, plan: ExecutionPlan, cache: dict[str, int]) -> int:
    sequence_id = str(plan.source_sequence_id or "").strip()
    if not sequence_id:
        return 0
    if sequence_id in cache:
        return cache[sequence_id]
    report_path = str(plan.source_compile_report_path or "").strip()
    if not report_path:
        cache[sequence_id] = 0
        return 0
    try:
        payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
        total = int(payload.get("compiled_plan_count", 0) or 0)
    except Exception:
        total = 0
    cache[sequence_id] = total
    return total


def build_sequence_progress(
    *,
    plan: ExecutionPlan,
    all_plans: list[ExecutionPlan],
    active_slice_id: str | None,
    total_batches: int,
) -> SequenceProgressView:
    sequence_plans = [
        item for item in all_plans
        if str(item.source_sequence_id or "").strip() == str(plan.source_sequence_id or "").strip()
    ]
    sequence_plans.sort(key=lambda item: int(item.sequence_batch_index or 0))
    batch_statuses = [
        _build_batch_progress(item, active_slice_id=active_slice_id if item.plan_id == plan.plan_id else None)
        for item in sequence_plans
    ]
    seen = {item.batch_index for item in batch_statuses}
    if total_batches > 0:
        for batch_index in range(1, total_batches + 1):
            if batch_index in seen:
                continue
            batch_statuses.append(BatchProgressView(batch_index=batch_index, status="pending"))
        batch_statuses.sort(key=lambda item: item.batch_index)
    current_batch = next((item for item in batch_statuses if item.batch_index == int(plan.sequence_batch_index or 0)), None)
    completed_batches = sum(1 for item in batch_statuses if item.status == "completed")
    return SequenceProgressView(
        raw_plan_label=_raw_plan_label(plan),
        sequence_id=str(plan.source_sequence_id or "").strip(),
        current_batch_index=int(plan.sequence_batch_index or 0),
        total_batches=total_batches,
        current_slice_index=current_batch.current_slice_index if current_batch else 0,
        total_slices_in_batch=current_batch.total_slices if current_batch else len(plan.slices),
        current_slice_id=_current_slice_id(plan, active_slice_id=active_slice_id),
        current_slice_title=current_batch.current_slice_label if current_batch else "",
        completed_batches=completed_batches,
        batch_statuses=batch_statuses,
    )


def _build_batch_progress(plan: ExecutionPlan, *, active_slice_id: str | None) -> BatchProgressView:
    completed_slices = sum(1 for item in plan.slices if item.status == "completed")
    total_slices = len(plan.slices)
    current_index, current_label = _current_slice_pointer(plan, active_slice_id=active_slice_id)
    return BatchProgressView(
        batch_index=int(plan.sequence_batch_index or 0),
        status=_batch_status(plan),
        completed_slices=completed_slices,
        total_slices=total_slices,
        current_slice_index=current_index,
        current_slice_label=current_label,
        slice_statuses=[
            f"S{index + 1} {item.status or 'pending'}"
            for index, item in enumerate(plan.slices)
        ],
    )


def _batch_status(plan: ExecutionPlan) -> str:
    if plan.status == "failed" or any(item.status == "failed" for item in plan.slices):
        return "failed"
    if any(item.status == "aborted" for item in plan.slices):
        return "aborted"
    if plan.slices and all(item.status == "completed" for item in plan.slices):
        return "completed"
    if any(item.status in {"running", "checkpointed", "completed"} for item in plan.slices):
        return "running"
    return "pending"


def _current_slice_pointer(plan: ExecutionPlan, *, active_slice_id: str | None) -> tuple[int, str]:
    if active_slice_id:
        for index, item in enumerate(plan.slices, start=1):
            if item.slice_id == active_slice_id:
                return index, item.title or item.slice_id
    for index, item in enumerate(plan.slices, start=1):
        if item.status not in {"completed", "failed", "aborted"}:
            return index, item.title or item.slice_id
    if plan.slices:
        last = plan.slices[-1]
        return len(plan.slices), last.title or last.slice_id
    return 0, ""


def _current_slice_id(plan: ExecutionPlan, *, active_slice_id: str | None) -> str:
    if active_slice_id:
        return active_slice_id
    for item in plan.slices:
        if item.status not in {"completed", "failed", "aborted"}:
            return item.slice_id
    return plan.slices[-1].slice_id if plan.slices else ""


def _raw_plan_label(plan: ExecutionPlan) -> str:
    source = str(plan.source_raw_plan or "").strip()
    if not source:
        return ""
    return Path(source).stem
