"""
In-memory render state for the broker-only rich runtime console.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
class ToolCallRuntimeView:
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
    broker_status: str = "unknown"
    broker_summary: str = ""
    broker_warnings: list[str] = field(default_factory=list)
    broker_tool_count: int = 0
    drain_mode: bool = False
    stop_reason: str = ""
    last_warning: str = ""
    last_warning_at: float = 0.0
    last_error: str = ""
    last_error_at: float = 0.0
    started_at_monotonic: float = 0.0
    planner: PlannerRuntimeView = field(default_factory=PlannerRuntimeView)
    slices: dict[int, SliceRuntimeView] = field(default_factory=dict)
    broker_calls: dict[int, ToolCallRuntimeView] = field(default_factory=dict)
