"""
In-memory render state for the direct rich runtime console.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SliceTrailEntry:
    """One completed slice outcome for the compact progress trail."""

    status: str
    execution_path: str = "direct"
    title: str = ""
    summary: str = ""


@dataclass
class TrailSliceSlot:
    """One slice position in the pre-built trail map."""

    slice_id: str = ""
    status: str = "pending"  # pending | completed | failed | aborted | skipped
    execution_path: str = "direct"  # direct | model


@dataclass
class TrailBatch:
    """One batch (compiled plan) in the trail map."""

    plan_id: str = ""
    slices: list[TrailSliceSlot] = field(default_factory=list)


@dataclass
class TrailPlanGroup:
    """One plan/sequence group in the trail map."""

    label: str = ""  # v1, v2, etc.
    sequence_id: str = ""
    batches: list[TrailBatch] = field(default_factory=list)


@dataclass
class TrailMap:
    """Full execution trail map — pre-built from plan structure."""

    plans: list[TrailPlanGroup] = field(default_factory=list)


@dataclass
class PlannerRuntimeView:
    status: str = "idle"
    adapter_name: str = ""
    attempt: int = 0
    max_attempts: int = 0
    started_at_monotonic: float = 0.0
    last_error: str = ""


@dataclass
class SliceRuntimeView:
    slot: int
    plan_id: str = ""
    slice_id: str = ""
    title: str = ""
    worker_id: str = ""
    status: str = "idle"
    turns_used: int = 0
    turns_total: int = 0
    tool_calls_used: int = 0
    tool_calls_total: int = 0
    last_summary: str = ""
    active_operation_ref: str = ""
    active_operation_status: str = ""


@dataclass
class BatchProgressView:
    batch_index: int
    status: str = "pending"
    completed_slices: int = 0
    total_slices: int = 0
    current_slice_index: int = 0
    current_slice_label: str = ""
    slice_statuses: list[str] = field(default_factory=list)


@dataclass
class SequenceProgressView:
    raw_plan_label: str = ""
    sequence_id: str = ""
    current_batch_index: int = 0
    total_batches: int = 0
    current_slice_index: int = 0
    total_slices_in_batch: int = 0
    current_slice_id: str = ""
    current_slice_title: str = ""
    completed_batches: int = 0
    batch_statuses: list[BatchProgressView] = field(default_factory=list)


@dataclass
class DirectToolRuntimeView:
    slot: int
    plan_id: str = ""
    slice_id: str = ""
    tool_name: str = ""
    phase: str = ""
    retryable: bool = False
    response_status: str = ""
    tool_response_status: str = ""
    operation_ref: str = ""
    error_class: str = ""
    started_at_monotonic: float = 0.0
    duration_ms: int = 0
    warning_count: int = 0


@dataclass
class ConsoleRuntimeState:
    runtime_status: str = "idle"
    goal: str = ""
    active_plan_id: str = ""
    current_cycle: int = 0
    total_errors: int = 0
    direct_status: str = "unknown"
    direct_summary: str = ""
    drain_mode: bool = False
    stop_reason: str = ""
    last_warning: str = ""
    last_warning_at: float = 0.0
    last_error: str = ""
    last_error_at: float = 0.0
    started_at_monotonic: float = 0.0
    planner: PlannerRuntimeView = field(default_factory=PlannerRuntimeView)
    sequence_progress: SequenceProgressView = field(default_factory=SequenceProgressView)
    slices: dict[int, SliceRuntimeView] = field(default_factory=dict)
    direct_calls: dict[int, DirectToolRuntimeView] = field(default_factory=dict)
    slice_trail: list[SliceTrailEntry] = field(default_factory=list)
    trail_map: TrailMap = field(default_factory=TrailMap)
