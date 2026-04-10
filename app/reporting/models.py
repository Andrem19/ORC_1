"""
Canonical reporting models for post-run summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.execution_models import BaselineRef, utc_now_iso


@dataclass
class NarrativeSectionsRu:
    executive_summary_ru: str = ""
    key_findings_ru: list[str] = field(default_factory=list)
    important_failures_ru: list[str] = field(default_factory=list)
    recommended_next_actions_ru: list[str] = field(default_factory=list)
    operator_notes_ru: list[str] = field(default_factory=list)


@dataclass
class IncidentReference:
    incident_id: str
    summary: str
    severity: str
    source: str = ""
    affected_tool: str = ""
    path: str = ""


@dataclass
class DirectExecutionMetrics:
    direct_completed: int = 0
    direct_blocked: int = 0
    direct_failed: int = 0
    direct_parse_retries: int = 0
    direct_tool_calls_observed: int = 0
    direct_incidents: int = 0


@dataclass
class SliceResultReport:
    slice_id: str
    title: str
    status: str
    verdict: str = ""
    summary: str = ""
    facts: dict[str, Any] = field(default_factory=dict)
    key_metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    rejected_findings: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    confidence: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    last_error: str = ""
    incident_refs: list[IncidentReference] = field(default_factory=list)
    report_path: str = ""


@dataclass
class PlanBatchReport:
    plan_id: str
    plan_source_kind: str
    source_sequence_id: str = ""
    source_raw_plan: str = ""
    status: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    baseline_ref: BaselineRef | None = None
    global_constraints: list[str] = field(default_factory=list)
    slice_results: list[SliceResultReport] = field(default_factory=list)
    direct_metrics: DirectExecutionMetrics = field(default_factory=DirectExecutionMetrics)
    incident_summary: list[IncidentReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_verdict: str = ""
    final_summary: str = ""
    next_actions: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    stop_reason: str = ""
    supporting_paths: dict[str, Any] = field(default_factory=dict)
    narrative_status: str = "pending"
    narrative_sections_ru: NarrativeSectionsRu = field(default_factory=NarrativeSectionsRu)
    generated_at: str = field(default_factory=utc_now_iso)


@dataclass
class SequenceExecutionReport:
    source_file: str
    source_hash: str
    sequence_id: str
    manifest_path: str = ""
    semantic_path: str = ""
    compile_report_path: str = ""
    semantic_stage_count: int = 0
    compiled_plan_count: int = 0
    compile_status: str = ""
    compile_warnings: list[str] = field(default_factory=list)
    semantic_method: str = ""
    sequence_status: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    batch_results: list[dict[str, Any]] = field(default_factory=list)
    skipped_reason: str = ""
    confirmed_findings: list[str] = field(default_factory=list)
    rejected_findings: list[str] = field(default_factory=list)
    failed_branches: list[str] = field(default_factory=list)
    key_metrics_rollup: dict[str, Any] = field(default_factory=dict)
    slice_verdict_rollup: dict[str, int] = field(default_factory=dict)
    incidents: list[IncidentReference] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)
    executive_summary_ru: str = ""
    narrative_status: str = "pending"
    narrative_sections_ru: NarrativeSectionsRu = field(default_factory=NarrativeSectionsRu)
    plan_report_paths: list[str] = field(default_factory=list)
    compiled_plan_paths: list[str] = field(default_factory=list)
    generated_at: str = field(default_factory=utc_now_iso)


@dataclass
class RunSummaryReport:
    run_id: str
    plan_source: str
    goal: str
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    stop_reason: str = ""
    total_raw_plans: int = 0
    compiled_sequences: int = 0
    executed_sequences: int = 0
    completed_sequences: int = 0
    failed_sequences: int = 0
    partial_sequences: int = 0
    skipped_sequences: int = 0
    per_sequence_table: list[dict[str, Any]] = field(default_factory=list)
    best_candidates: list[str] = field(default_factory=list)
    best_outcomes: list[str] = field(default_factory=list)
    unresolved_blockers: list[str] = field(default_factory=list)
    recurring_incidents: list[dict[str, Any]] = field(default_factory=list)
    direct_metrics: DirectExecutionMetrics = field(default_factory=DirectExecutionMetrics)
    continue_items: list[str] = field(default_factory=list)
    drop_items: list[str] = field(default_factory=list)
    rerun_items: list[str] = field(default_factory=list)
    operator_notes_ru: list[str] = field(default_factory=list)
    executive_summary_ru: str = ""
    narrative_status: str = "pending"
    narrative_sections_ru: NarrativeSectionsRu = field(default_factory=NarrativeSectionsRu)
    sequence_report_paths: list[str] = field(default_factory=list)
    plan_report_paths: list[str] = field(default_factory=list)
    log_path: str = ""
    state_path: str = ""
    incident_dir: str = ""
    generated_at: str = field(default_factory=utc_now_iso)
