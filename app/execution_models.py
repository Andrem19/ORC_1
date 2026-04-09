"""
Structured models for brokered planner/worker execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class BaselineRef:
    snapshot_id: str
    version: int | str
    symbol: str = "BTCUSDT"
    anchor_timeframe: str = "1h"
    execution_timeframe: str = "5m"


@dataclass
class PlanSlice:
    slice_id: str
    title: str
    hypothesis: str
    objective: str
    success_criteria: list[str]
    allowed_tools: list[str]
    evidence_requirements: list[str]
    policy_tags: list[str]
    max_turns: int
    max_tool_calls: int
    max_expensive_calls: int
    parallel_slot: int
    status: str = "pending"  # pending | running | checkpointed | completed | failed | aborted
    turn_count: int = 0
    tool_call_count: int = 0
    expensive_call_count: int = 0
    assigned_worker_id: str | None = None
    last_checkpoint_turn_id: str = ""
    last_checkpoint_summary: str = ""
    final_report_turn_id: str = ""
    last_error: str = ""
    last_summary: str = ""
    latest_tool_result_summary: str = ""
    active_operation_tool: str = ""
    active_operation_ref: str = ""
    active_operation_status: str = ""
    active_operation_arguments: dict[str, Any] = field(default_factory=dict)
    active_resume_tool: str = ""
    active_resume_token: str = ""
    last_tool_response_status: str = ""
    artifacts: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed", "aborted"}


@dataclass
class ExecutionPlan:
    plan_id: str
    goal: str
    baseline_ref: BaselineRef
    global_constraints: list[str]
    slices: list[PlanSlice]
    plan_source_kind: str = "planner"
    source_sequence_id: str = ""
    source_raw_plan: str = ""
    source_manifest_path: str = ""
    source_semantic_path: str = ""
    source_compile_report_path: str = ""
    sequence_batch_index: int = 0
    status: str = "draft"  # draft | running | completed | failed
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "failed"}

    def active_slices(self) -> list[PlanSlice]:
        return [item for item in self.slices if not item.is_terminal]


@dataclass
class WorkerReportableIssue:
    summary: str
    severity: str = "medium"
    details: str = ""
    affected_tool: str = ""
    category: str = "runtime"


@dataclass
class WorkerAction:
    action_id: str
    action_type: str  # tool_call | checkpoint | final_report | abort
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    expected_evidence: list[str] = field(default_factory=list)
    status: str = ""
    summary: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    pending_questions: list[str] = field(default_factory=list)
    reportable_issues: list[WorkerReportableIssue] = field(default_factory=list)
    verdict: str = ""
    key_metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    rejected_findings: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason_code: str = ""
    retryable: bool = False


@dataclass
class ToolPolicy:
    tool_name: str
    expensive: bool = False
    async_resumable: bool = False
    mutating: bool = False
    autopoll_enabled: bool = False
    allowed_wait_modes: list[str] = field(default_factory=list)
    default_wait_mode: str = ""


@dataclass
class ToolDefinition:
    tool_name: str
    description: str
    title: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    policy: ToolPolicy = field(default_factory=lambda: ToolPolicy(tool_name=""))


@dataclass
class ToolResultEnvelope:
    call_id: str
    tool: str
    ok: bool
    retryable: bool
    duration_ms: int
    summary: str
    key_facts: dict[str, Any] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error_class: str = ""
    raw_result_ref: str = ""
    request_arguments: dict[str, Any] = field(default_factory=dict)
    response_status: str = ""
    tool_response_status: str = ""
    operation_ref: str = ""
    resume_tool: str = ""
    resume_token: str = ""
    resume_arguments: dict[str, Any] = field(default_factory=dict)
    plan_id: str = ""
    slice_id: str = ""
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class ExecutionTurn:
    turn_id: str
    plan_id: str
    slice_id: str
    worker_id: str
    turn_index: int
    action: WorkerAction
    tool_result: ToolResultEnvelope | None = None
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class BrokerHealth:
    endpoint_url: str = ""
    bootstrapped_at: str = ""
    session_id: str = ""
    tool_count: int = 0
    status: str = "unknown"
    summary: str = ""
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExecutionStateV2:
    goal: str
    status: str = "idle"  # idle | running | finished | error
    plans: list[ExecutionPlan] = field(default_factory=list)
    turn_history: list[ExecutionTurn] = field(default_factory=list)
    tool_call_ledger: list[ToolResultEnvelope] = field(default_factory=list)
    broker_health: BrokerHealth = field(default_factory=BrokerHealth)
    stop_reason: str = ""
    current_plan_id: str = ""
    completed_plan_count: int = 0
    consecutive_failed_plans: int = 0
    no_progress_cycles: int = 0
    broker_failure_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def touch(self) -> None:
        self.updated_at = utc_now_iso()

    def active_plan(self) -> ExecutionPlan | None:
        for plan in reversed(self.plans):
            if plan.plan_id == self.current_plan_id:
                return plan
        for plan in reversed(self.plans):
            if not plan.is_terminal:
                return plan
        return None

    def find_plan(self, plan_id: str) -> ExecutionPlan | None:
        for plan in self.plans:
            if plan.plan_id == plan_id:
                return plan
        return None

    def find_slice(self, plan_id: str, slice_id: str) -> PlanSlice | None:
        plan = self.find_plan(plan_id)
        if plan is None:
            return None
        for item in plan.slices:
            if item.slice_id == slice_id:
                return item
        return None
