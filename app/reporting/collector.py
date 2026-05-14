"""
Deterministic collection of canonical reports from persisted runtime artifacts.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.compiled_plan_store import CompiledPlanStore
from app.config import OrchestratorConfig
from app.execution_models import ExecutionPlan, ExecutionStateV2, PlanSlice, WorkerAction
from app.raw_plan_ordering import raw_plan_sort_key
from app.reporting.models import (
    IncidentReference,
    NarrativeSectionsRu,
    PlanBatchReport,
    RunSummaryReport,
    SequenceExecutionReport,
    SliceResultReport,
    DirectExecutionMetrics,
)
from app.run_context import resolve_run_dir

_BLOCKED_ABORT_REASON_CODES = {
    "infra_contract_blocker",
    "dependency_blocked",
    "worker_contract_recovery_exhausted",
    "tool_selection_ambiguous",
}


class ReportCollector:
    def __init__(self, *, config: OrchestratorConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.plan_run_dir = resolve_run_dir(config.plan_dir, run_id)
        self.state_run_dir = resolve_run_dir(config.state_dir, run_id)
        self.log_run_dir = resolve_run_dir(config.log_dir, run_id)
        self.compiled_plan_store = CompiledPlanStore(config.compiled_plan_dir)
        self._incidents = self._load_incidents()

    def collect_plan_reports(self, state: ExecutionStateV2) -> list[PlanBatchReport]:
        return [self._plan_report(plan=plan, state=state) for plan in state.plans]

    def collect_sequence_reports(self, *, state: ExecutionStateV2, plan_reports: list[PlanBatchReport]) -> list[SequenceExecutionReport]:
        if self.config.plan_source != "compiled_raw":
            return []
        plan_reports_by_sequence: dict[str, list[PlanBatchReport]] = {}
        for report in plan_reports:
            if report.source_sequence_id:
                plan_reports_by_sequence.setdefault(report.source_sequence_id, []).append(report)
        manifests = {item.sequence_id: item for item in self.compiled_plan_store.load_manifests()}
        raw_files = sorted(Path(self.config.raw_plan_dir).glob("*.md"), key=raw_plan_sort_key)
        reports: list[SequenceExecutionReport] = []
        for raw_file in raw_files:
            default_sequence_id = f"compiled_{raw_file.stem}"
            manifest = manifests.get(default_sequence_id)
            sequence_id = manifest.sequence_id if manifest is not None else default_sequence_id
            reports.append(
                self._sequence_report(
                    raw_file=raw_file,
                    sequence_id=sequence_id,
                    manifest=manifest,
                    state=state,
                    plan_reports=sorted(plan_reports_by_sequence.get(sequence_id, []), key=lambda item: item.plan_id),
                )
            )
        return reports

    def collect_run_report(
        self,
        *,
        state: ExecutionStateV2,
        plan_reports: list[PlanBatchReport],
        sequence_reports: list[SequenceExecutionReport],
    ) -> RunSummaryReport:
        recurring = _recurring_incidents(self._incidents)
        best_candidates = [
            f"{item.sequence_id}: {item.narrative_sections_ru.executive_summary_ru or item.executive_summary_ru or _sequence_status_ru(item.sequence_status)}"
            for item in sequence_reports
            if _sequence_top_verdict(item) == "PROMOTE"
        ]
        best_outcomes = [
            f"{item.sequence_id}: {_verdict_ru(_sequence_top_verdict(item))}"
            for item in sequence_reports
            if _sequence_top_verdict(item) in {"PROMOTE", "WATCHLIST"}
        ]
        continue_items = [
            f"{item.sequence_id}: продолжать исследование"
            for item in sequence_reports
            if item.sequence_status == "completed" and _sequence_top_verdict(item) in {"PROMOTE", "WATCHLIST"}
        ]
        drop_items = [
            f"{item.sequence_id}: прекратить ветку"
            for item in sequence_reports
            if item.sequence_status == "failed" or _sequence_top_verdict(item) == "REJECT"
        ]
        rerun_items = [
            f"{item.sequence_id}: повторить после исправления блокеров"
            for item in sequence_reports
            if item.sequence_status in {"partial", "skipped"}
        ]
        report = RunSummaryReport(
            run_id=self.run_id,
            plan_source=self.config.plan_source,
            goal=state.goal,
            started_at=state.created_at,
            finished_at=state.updated_at,
            duration_ms=_duration_ms(state.created_at, state.updated_at),
            stop_reason=state.stop_reason,
            total_raw_plans=len(sorted(Path(self.config.raw_plan_dir).glob("*.md"), key=raw_plan_sort_key)) if self.config.plan_source == "compiled_raw" else 0,
            compiled_sequences=sum(1 for item in sequence_reports if item.compile_status == "compiled"),
            executed_sequences=sum(1 for item in sequence_reports if item.sequence_status in {"completed", "failed", "partial"}),
            completed_sequences=sum(1 for item in sequence_reports if item.sequence_status == "completed"),
            failed_sequences=sum(1 for item in sequence_reports if item.sequence_status == "failed"),
            partial_sequences=sum(1 for item in sequence_reports if item.sequence_status == "partial"),
            skipped_sequences=sum(1 for item in sequence_reports if item.sequence_status == "skipped"),
            per_sequence_table=[
                {
                    "sequence_id": item.sequence_id,
                    "status": item.sequence_status,
                    "top_verdict": _sequence_top_verdict(item),
                    "summary": item.executive_summary_ru or item.narrative_sections_ru.executive_summary_ru or item.sequence_status,
                }
                for item in sequence_reports
            ],
            best_candidates=best_candidates,
            best_outcomes=best_outcomes,
            unresolved_blockers=_unique_preserve_order(
                blocker
                for item in sequence_reports
                for blocker in item.blockers
            ),
            recurring_incidents=recurring,
            direct_metrics=_direct_metrics_for_state(state=state, incidents=self._incidents),
            continue_items=continue_items,
            drop_items=drop_items,
            rerun_items=rerun_items,
            log_path=str(self.log_run_dir / self.config.log_file),
            state_path=str(self.state_run_dir / self.config.execution_state_file),
            incident_dir=str(self.state_run_dir / "incidents"),
        )
        report.executive_summary_ru = _run_fallback_summary(report)
        report.operator_notes_ru = _run_operator_notes(report)
        return report

    def _plan_report(self, *, plan: ExecutionPlan, state: ExecutionStateV2) -> PlanBatchReport:
        plan_reports_dir = self.plan_run_dir / "reports" / plan.plan_id
        slice_results = [self._slice_result(plan=plan, slice_obj=slice_obj, reports_dir=plan_reports_dir) for slice_obj in plan.slices]
        incidents = self._plan_incidents(plan.plan_id)
        plan_status = _reported_plan_status(plan=plan, state=state)
        report = PlanBatchReport(
            plan_id=plan.plan_id,
            plan_source_kind=plan.plan_source_kind,
            source_sequence_id=plan.source_sequence_id,
            source_raw_plan=plan.source_raw_plan,
            status=plan_status,
            started_at=plan.created_at,
            finished_at=plan.updated_at,
            duration_ms=_duration_ms(plan.created_at, plan.updated_at),
            baseline_ref=plan.baseline_ref,
            global_constraints=list(plan.global_constraints),
            slice_results=slice_results,
            direct_metrics=_direct_metrics_for_plan(plan=plan, state=state, incidents=self._incidents),
            incident_summary=incidents,
            warnings=[],
            final_verdict=_aggregate_plan_verdict(slice_results=slice_results, plan_status=plan_status),
            final_summary=_aggregate_plan_summary(slice_results),
            next_actions=_unique_preserve_order(action for item in slice_results for action in item.next_actions),
            artifacts=_unique_preserve_order(artifact for item in slice_results for artifact in item.artifacts),
            stop_reason=state.stop_reason if state.current_plan_id == plan.plan_id else "",
            supporting_paths={
                "plan_json": str(self.plan_run_dir / "plans" / f"{plan.plan_id}.json"),
                "slice_report_dir": str(self.plan_run_dir / "reports" / plan.plan_id),
                "worker_failure_dir": str(self.plan_run_dir / "worker_failures" / plan.plan_id),
            },
        )
        return report

    def _slice_result(self, *, plan: ExecutionPlan, slice_obj: PlanSlice, reports_dir: Path) -> SliceResultReport:
        payload, report_path = _load_slice_terminal_payload(reports_dir=reports_dir, slice_obj=slice_obj)
        incident_refs = self._slice_incidents(plan.plan_id, slice_obj.slice_id)
        action = _action_from_payload(payload)
        return SliceResultReport(
            slice_id=slice_obj.slice_id,
            title=slice_obj.title,
            status=slice_obj.status,
            acceptance_state=str(getattr(slice_obj, "acceptance_state", "") or ""),
            dependency_unblock_mode=str(getattr(slice_obj, "dependency_unblock_mode", "") or ""),
            verdict=action.verdict or _fallback_slice_verdict(slice_obj),
            summary=action.summary or slice_obj.last_summary or slice_obj.last_error,
            facts=dict(slice_obj.facts),
            key_metrics=dict(action.key_metrics),
            findings=list(action.findings),
            rejected_findings=list(action.rejected_findings),
            next_actions=list(action.next_actions),
            risks=list(action.risks),
            evidence_refs=list(action.evidence_refs),
            confidence=action.confidence,
            artifacts=list(slice_obj.artifacts),
            last_error=slice_obj.last_error,
            dependency_blocker_slice_id=str(getattr(slice_obj, "dependency_blocker_slice_id", "") or ""),
            dependency_blocker_reason_code=str(getattr(slice_obj, "dependency_blocker_reason_code", "") or ""),
            dependency_blocker_class=str(getattr(slice_obj, "dependency_blocker_class", "") or ""),
            root_failure_reason=str(slice_obj.facts.get("direct.root_failure_reason") or ""),
            root_failure_artifact=str(slice_obj.facts.get("direct.root_failure_artifact") or ""),
            best_failed_attempt_provider=str(slice_obj.facts.get("direct.best_failed_attempt_provider") or ""),
            best_failed_tool_call_count=int(slice_obj.facts.get("direct.best_failed_tool_call_count") or 0),
            last_provider_failure=str(slice_obj.facts.get("direct.last_provider_failure") or ""),
            incident_refs=incident_refs,
            report_path=str(report_path) if report_path is not None else "",
        )

    def _sequence_report(
        self,
        *,
        raw_file: Path,
        sequence_id: str,
        manifest: Any | None,
        state: ExecutionStateV2,
        plan_reports: list[PlanBatchReport],
    ) -> SequenceExecutionReport:
        compile_payload = _load_json(Path(self.config.compiled_plan_dir) / raw_file.stem / "compile_report.json") if manifest is not None else {}
        compiled_plan_paths = [
            str((Path(self.config.compiled_plan_dir) / raw_file.stem / item))
            for item in (manifest.plan_files if manifest is not None else [])
        ]
        sequence_status, skipped_reason = _sequence_status(
            manifest=manifest,
            plan_reports=plan_reports,
            state=state,
            compiled_plan_paths=compiled_plan_paths,
        )
        incidents = _unique_incidents(
            incident
            for report in plan_reports
            for incident in report.incident_summary
        )
        report = SequenceExecutionReport(
            source_file=str(raw_file),
            source_hash=str(getattr(manifest, "source_hash", "") or ""),
            sequence_id=sequence_id,
            manifest_path=str(Path(self.config.compiled_plan_dir) / raw_file.stem / "manifest.json") if manifest is not None else "",
            semantic_path=str((Path(self.config.compiled_plan_dir) / raw_file.stem / manifest.semantic_path)) if manifest is not None and manifest.semantic_path else "",
            compile_report_path=str((Path(self.config.compiled_plan_dir) / raw_file.stem / manifest.compile_report_path)) if manifest is not None and manifest.compile_report_path else "",
            semantic_stage_count=int(compile_payload.get("stage_count", 0) or 0),
            compiled_plan_count=int(compile_payload.get("compiled_plan_count", len(compiled_plan_paths)) or len(compiled_plan_paths)),
            compile_status=str(getattr(manifest, "compile_status", "") or compile_payload.get("compile_status", "")),
            compile_warnings=[str(item) for item in compile_payload.get("warnings", []) or getattr(manifest, "warnings", []) or []],
            semantic_method=str(getattr(manifest, "semantic_method", "") or compile_payload.get("semantic_method", "")),
            sequence_status=sequence_status,
            started_at=min((item.started_at for item in plan_reports if item.started_at), default=""),
            finished_at=max((item.finished_at for item in plan_reports if item.finished_at), default=""),
            duration_ms=_duration_ms(
                min((item.started_at for item in plan_reports if item.started_at), default=""),
                max((item.finished_at for item in plan_reports if item.finished_at), default=""),
            ),
            batch_results=[
                {
                    "plan_id": item.plan_id,
                    "status": item.status,
                    "final_verdict": item.final_verdict,
                    "summary": item.final_summary,
                }
                for item in plan_reports
            ],
            skipped_reason=skipped_reason,
            confirmed_findings=_unique_preserve_order(
                finding
                for item in plan_reports
                for slice_result in item.slice_results
                for finding in (slice_result.findings or ([slice_result.summary] if slice_result.summary and slice_result.status == "completed" else []))
            ),
            rejected_findings=_unique_preserve_order(
                finding
                for item in plan_reports
                for slice_result in item.slice_results
                for finding in (slice_result.rejected_findings or ([slice_result.summary] if slice_result.status in {"failed", "aborted"} and slice_result.summary else []))
            ),
            failed_branches=_unique_preserve_order(
                (
                    (
                        f"{slice_result.slice_id}: root {slice_result.root_failure_reason} "
                        f"({slice_result.best_failed_attempt_provider}, tools={slice_result.best_failed_tool_call_count})"
                        if slice_result.root_failure_reason
                        else f"{slice_result.slice_id}: {slice_result.summary or slice_result.last_error or slice_result.status}"
                    )
                    if not slice_result.dependency_blocker_slice_id
                    else (
                        f"{slice_result.slice_id}: blocked by {slice_result.dependency_blocker_slice_id} "
                        f"({slice_result.dependency_blocker_class or 'unknown'}:"
                        f"{slice_result.dependency_blocker_reason_code or slice_result.last_error or 'dependency_blocked'})"
                    )
                )
                for item in plan_reports
                for slice_result in item.slice_results
                if slice_result.status in {"failed", "aborted"} or (slice_result.status == "checkpointed" and slice_result.last_error)
            ),
            key_metrics_rollup=_rollup_key_metrics(plan_reports),
            slice_verdict_rollup=dict(Counter(
                slice_result.verdict or "PENDING"
                for item in plan_reports
                for slice_result in item.slice_results
            )),
            incidents=incidents,
            blockers=_unique_preserve_order(
                blocker
                for incident in incidents
                for blocker in [incident.summary]
            ),
            recommended_next_actions=_unique_preserve_order(
                action
                for item in plan_reports
                for action in item.next_actions
            ),
            executive_summary_ru=_sequence_fallback_summary(
                source_name=raw_file.name,
                status=sequence_status,
                plan_reports=plan_reports,
                skipped_reason=skipped_reason,
            ),
            compiled_plan_paths=compiled_plan_paths,
        )
        report.plan_report_paths = [
            str(self.plan_run_dir / "plan_reports" / f"{item.plan_id}.json")
            for item in plan_reports
        ]
        return report

    def _load_incidents(self) -> list[dict[str, Any]]:
        incidents_dir = self.state_run_dir / "incidents"
        if not incidents_dir.exists():
            return []
        return [
            _load_json(path) | {"_path": str(path)}
            for path in sorted(incidents_dir.glob("*.json"))
        ]

    def _plan_incidents(self, plan_id: str) -> list[IncidentReference]:
        result: list[IncidentReference] = []
        for payload in self._incidents:
            metadata = payload.get("metadata", {}) or {}
            if str(metadata.get("plan_id", "") or "") != plan_id:
                continue
            result.append(_incident_ref(payload))
        return _unique_incidents(result)

    def _slice_incidents(self, plan_id: str, slice_id: str) -> list[IncidentReference]:
        result: list[IncidentReference] = []
        for payload in self._incidents:
            metadata = payload.get("metadata", {}) or {}
            if str(metadata.get("plan_id", "") or "") != plan_id:
                continue
            if str(metadata.get("slice_id", "") or "") != slice_id:
                continue
            result.append(_incident_ref(payload))
        return _unique_incidents(result)


def _load_slice_terminal_payload(*, reports_dir: Path, slice_obj: PlanSlice) -> tuple[dict[str, Any], Path | None]:
    explicit_turns = [slice_obj.final_report_turn_id, slice_obj.last_checkpoint_turn_id]
    for turn_id in explicit_turns:
        if not turn_id:
            continue
        path = reports_dir / slice_obj.slice_id / f"{turn_id}.json"
        if path.exists():
            return _load_json(path), path
    candidates = list((reports_dir / slice_obj.slice_id).glob("*.json"))
    if not candidates:
        return {}, None
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return _load_json(latest), latest


def _action_from_payload(payload: dict[str, Any]) -> WorkerAction:
    return WorkerAction(
        action_id=str(payload.get("action_id", "") or ""),
        action_type=str(payload.get("type", payload.get("action_type", "")) or ""),
        summary=str(payload.get("summary", "") or ""),
        verdict=str(payload.get("verdict", "") or ""),
        facts=dict(payload.get("facts", {}) or {}),
        artifacts=[str(item) for item in payload.get("artifacts", []) or []],
        key_metrics=dict(payload.get("key_metrics", {}) or {}),
        findings=[str(item) for item in payload.get("findings", []) or []],
        rejected_findings=[str(item) for item in payload.get("rejected_findings", []) or []],
        next_actions=[str(item) for item in payload.get("next_actions", []) or []],
        risks=[str(item) for item in payload.get("risks", []) or []],
        evidence_refs=[str(item) for item in payload.get("evidence_refs", []) or []],
        confidence=float(payload.get("confidence", 0.0) or 0.0),
    )


def _incident_ref(payload: dict[str, Any]) -> IncidentReference:
    metadata = payload.get("metadata", {}) or {}
    return IncidentReference(
        incident_id=str(payload.get("incident_id", "") or ""),
        summary=str(payload.get("summary", "") or ""),
        severity=str(payload.get("severity", "medium") or "medium"),
        source=str(payload.get("source", "") or ""),
        affected_tool=str(metadata.get("affected_tool", metadata.get("tool_name", "")) or ""),
        path=str(payload.get("_path", "") or ""),
    )


def _direct_metrics_for_plan(*, plan: ExecutionPlan, state: ExecutionStateV2, incidents: list[dict[str, Any]]) -> DirectExecutionMetrics:
    metrics = DirectExecutionMetrics()
    slice_ids = {item.slice_id for item in plan.slices}
    for slice_obj in plan.slices:
        if slice_obj.status == "completed":
            metrics.direct_completed += 1
        elif slice_obj.status == "failed":
            metrics.direct_failed += 1
        elif slice_obj.status == "aborted" or (slice_obj.status == "checkpointed" and slice_obj.last_checkpoint_status == "blocked"):
            metrics.direct_blocked += 1
    for turn in state.turn_history:
        if turn.plan_id != plan.plan_id or turn.slice_id not in slice_ids:
            continue
        metrics.direct_parse_retries += int(turn.direct_attempt.parse_retry_count or 0)
        metrics.direct_tool_calls_observed += int(turn.direct_attempt.tool_call_count or 0)
    metrics.direct_incidents = sum(1 for item in incidents if (item.get("metadata", {}) or {}).get("plan_id") == plan.plan_id)
    return metrics


def _direct_metrics_for_state(*, state: ExecutionStateV2, incidents: list[dict[str, Any]]) -> DirectExecutionMetrics:
    metrics = DirectExecutionMetrics()
    for plan in state.plans:
        current = _direct_metrics_for_plan(plan=plan, state=state, incidents=incidents)
        metrics.direct_completed += current.direct_completed
        metrics.direct_blocked += current.direct_blocked
        metrics.direct_failed += current.direct_failed
        metrics.direct_parse_retries += current.direct_parse_retries
        metrics.direct_tool_calls_observed += current.direct_tool_calls_observed
        metrics.direct_incidents += current.direct_incidents
    return metrics


def _aggregate_plan_verdict(*, slice_results: list[SliceResultReport], plan_status: str) -> str:
    verdicts = [item.verdict for item in slice_results if item.verdict]
    if "PROMOTE" in verdicts:
        return "PROMOTE"
    if "WATCHLIST" in verdicts:
        return "WATCHLIST"
    if "REJECT" in verdicts:
        return "REJECT"
    if "FAILED" in verdicts:
        return "FAILED"
    if plan_status == "failed":
        return "FAILED"
    return "PENDING"


def _aggregate_plan_summary(slice_results: list[SliceResultReport]) -> str:
    texts = [item.summary for item in slice_results if item.summary]
    if not texts:
        return "Итоговая сводка по plan отсутствует."
    return " | ".join(texts[:3])


def _fallback_slice_verdict(slice_obj: PlanSlice) -> str:
    if slice_obj.status == "completed":
        return "WATCHLIST"
    if slice_obj.status == "aborted" and slice_obj.last_error in _BLOCKED_ABORT_REASON_CODES:
        return "PENDING"
    if slice_obj.status in {"failed", "aborted"}:
        return "REJECT"
    return "PENDING"


def _rollup_key_metrics(plan_reports: list[PlanBatchReport]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for report in plan_reports:
        for slice_result in report.slice_results:
            for key, value in slice_result.key_metrics.items():
                grouped.setdefault(key, []).append(
                    {
                        "plan_id": report.plan_id,
                        "slice_id": slice_result.slice_id,
                        "value": value,
                    }
                )
    merged: dict[str, Any] = {}
    for key, entries in grouped.items():
        unique_values = {_stable_metric_key(item["value"]) for item in entries}
        if len(unique_values) == 1:
            merged[key] = entries[0]["value"]
        else:
            merged[key] = entries
    return merged


def _sequence_status(
    *,
    manifest: Any | None,
    plan_reports: list[PlanBatchReport],
    state: ExecutionStateV2,
    compiled_plan_paths: list[str],
) -> tuple[str, str]:
    if manifest is None:
        return "skipped", "missing_compiled_manifest"
    if str(manifest.compile_status or "") != "compiled":
        return "skipped", f"compile_status={manifest.compile_status}"
    if any(not Path(path).exists() for path in compiled_plan_paths):
        return "skipped", "missing_compiled_plan_file"
    if not plan_reports:
        if state.stop_reason == "graceful_stop":
            return "partial", "run_stopped_before_sequence_started"
        return "skipped", "not_reached_before_run_end"
    statuses = {item.status for item in plan_reports}
    if "stopped" in statuses:
        return "partial", "sequence_incomplete_at_run_end"
    if "failed" in statuses:
        return "failed", ""
    expected = len(getattr(manifest, "plan_files", []) or [])
    if len(plan_reports) < expected:
        return "partial", "sequence_incomplete_at_run_end"
    if statuses == {"completed"}:
        return "completed", ""
    return "partial", ""


def _sequence_top_verdict(report: SequenceExecutionReport) -> str:
    counts = report.slice_verdict_rollup
    for verdict in ("PROMOTE", "WATCHLIST", "REJECT", "FAILED"):
        if counts.get(verdict):
            return verdict
    return "PENDING"


def _sequence_fallback_summary(*, source_name: str, status: str, plan_reports: list[PlanBatchReport], skipped_reason: str) -> str:
    if status == "completed":
        return f"Последовательность `{source_name}` завершена. Batch-планов: {len(plan_reports)}."
    if status == "failed":
        return f"Последовательность `{source_name}` завершилась с ошибкой."
    if status == "partial":
        return f"Последовательность `{source_name}` завершилась частично."
    return f"Последовательность `{source_name}` пропущена: {_skip_reason_ru(skipped_reason) or 'нет исполняемых batch-планов'}."


def _run_fallback_summary(report: RunSummaryReport) -> str:
    return (
        f"Прогон `{report.run_id}` завершён с причиной `{report.stop_reason or '-'}`. "
        f"Завершено sequence: {report.completed_sequences}, "
        f"с ошибкой: {report.failed_sequences}, "
        f"частично: {report.partial_sequences}."
    )


def _run_operator_notes(report: RunSummaryReport) -> list[str]:
    notes: list[str] = []
    if report.failed_sequences:
        notes.append("Есть sequence с execution failure, проверьте batch reports и incidents.")
    if report.partial_sequences:
        notes.append("Есть partial sequence, возможно нужен повторный прогон или продолжение после recovery.")
    if report.skipped_sequences:
        notes.append("Есть skipped sequence, проверьте compiled manifests и очередь compiled_raw.")
    return notes


def _sequence_status_ru(status: str) -> str:
    mapping = {
        "completed": "завершено",
        "failed": "ошибка",
        "stopped": "остановлено",
        "partial": "частично",
        "skipped": "пропущено",
        "pending": "ожидание",
    }
    return mapping.get(status, status or "-")


def _reported_plan_status(*, plan: ExecutionPlan, state: ExecutionStateV2) -> str:
    if plan.status == "running" and state.stop_reason == "graceful_stop":
        return "stopped"
    return plan.status


def _verdict_ru(verdict: str) -> str:
    mapping = {
        "PROMOTE": "продвигать",
        "WATCHLIST": "наблюдать",
        "REJECT": "отклонить",
        "FAILED": "ошибка",
        "PENDING": "ожидание",
    }
    return mapping.get(verdict, verdict or "-")


def _skip_reason_ru(reason: str) -> str:
    mapping = {
        "missing_compiled_manifest": "отсутствует manifest compiled-плана",
        "missing_compiled_plan_file": "отсутствует файл compiled batch-плана",
        "not_reached_before_run_end": "очередь не дошла до sequence до завершения run",
        "run_stopped_before_sequence_started": "run остановлен до старта sequence",
        "sequence_incomplete_at_run_end": "sequence не завершена к моменту окончания run",
    }
    if reason.startswith("compile_status="):
        return f"статус компиляции: {reason.split('=', 1)[1] or '-'}"
    return mapping.get(reason, reason)


def _unique_preserve_order(values: Any) -> list[Any]:
    result: list[Any] = []
    seen: set[str] = set()
    for item in values:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(item)
    return result


def _unique_incidents(items: Any) -> list[IncidentReference]:
    result: list[IncidentReference] = []
    seen: set[str] = set()
    for item in items:
        key = f"{item.incident_id}:{item.summary}:{item.path}"
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _recurring_incidents(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: dict[tuple[str, str], int] = {}
    for item in items:
        metadata = item.get("metadata", {}) or {}
        key = (
            str(item.get("summary", "") or ""),
            str(metadata.get("affected_tool", metadata.get("tool_name", "")) or ""),
        )
        counter[key] = counter.get(key, 0) + 1
    result = [
        {"summary": summary, "affected_tool": affected_tool, "count": count}
        for (summary, affected_tool), count in sorted(counter.items(), key=lambda item: (-item[1], item[0][0]))
        if count > 1
    ]
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _duration_ms(started_at: str, finished_at: str) -> int:
    if not started_at or not finished_at:
        return 0
    try:
        from datetime import datetime

        start = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
        finish = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except Exception:
        return 0
    return max(0, int((finish - start).total_seconds() * 1000))


def _stable_metric_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return repr(value)
