"""
Structured models for raw-plan conversion and compiled plan sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.execution_models import BaselineRef, ExecutionPlan, utc_now_iso


@dataclass
class RawPlanStageFragment:
    stage_id: str
    order_index: int
    heading: str
    title: str
    objective_hint: str = ""
    actions_hint: list[str] = field(default_factory=list)
    success_criteria_hint: list[str] = field(default_factory=list)
    result_table_fields: list[str] = field(default_factory=list)
    raw_markdown: str = ""
    section_titles: list[str] = field(default_factory=list)


@dataclass
class RawPlanDocument:
    source_file: str
    source_hash: str
    title: str
    version_label: str
    normalized_text: str
    pre_stage_context: str = ""
    baseline_ref_hint: dict[str, Any] = field(default_factory=dict)
    global_sections: dict[str, str] = field(default_factory=dict)
    candidate_stages: list[RawPlanStageFragment] = field(default_factory=list)
    parser_warnings: list[str] = field(default_factory=list)
    parse_confidence: float = 0.0


@dataclass
class SemanticStage:
    stage_id: str
    title: str
    objective: str
    actions: list[str]
    success_criteria: list[str]
    tool_hints: list[str] = field(default_factory=list)
    policy_tags: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    required: bool = True
    parallelizable: bool = False
    gate_hint: str = ""
    raw_stage_ref: str = ""


@dataclass
class SemanticRawPlan:
    source_file: str
    source_hash: str
    source_title: str
    goal: str
    baseline_ref: BaselineRef
    global_constraints: list[str]
    stages: list[SemanticStage]
    warnings: list[str] = field(default_factory=list)
    parse_confidence: float = 0.0


@dataclass
class CompileReport:
    source_file: str
    source_hash: str
    sequence_id: str
    compile_status: str
    parser_confidence: float
    semantic_method: str
    stage_count: int
    compiled_plan_count: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    compiled_at: str = field(default_factory=utc_now_iso)


@dataclass
class CompiledPlanManifest:
    source_file: str
    source_hash: str
    compiled_at: str
    sequence_id: str
    compile_status: str
    warnings: list[str]
    semantic_method: str
    semantic_path: str = ""
    compile_report_path: str = ""
    plan_files: list[str] = field(default_factory=list)


@dataclass
class CompiledPlanSequence:
    source_file: str
    source_hash: str
    sequence_id: str
    semantic_plan: SemanticRawPlan | None
    plans: list[ExecutionPlan]
    report: CompileReport
