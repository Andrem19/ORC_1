"""
Compatibility helpers around the canonical PlanReport type.
"""

from __future__ import annotations

from app.models import TaskResult
from app.plan_models import PlanReport
from app.result_identity import ensure_result_fingerprint


TaskReport = PlanReport


def task_report_to_task_result(report: PlanReport, *, allow_plan_id_fallback: bool = False) -> TaskResult:
    """Convert the canonical worker report to the generic TaskResult."""

    task_id = report.task_id
    if not task_id and allow_plan_id_fallback:
        task_id = report.plan_id or f"plan_v{report.plan_version}"
    result = TaskResult(
        task_id=task_id,
        worker_id=report.worker_id,
        status=report.status,
        summary=report.what_was_done[:1000],
        artifacts=report.artifacts,
        confidence=report.confidence,
        error=report.error,
        raw_output=report.raw_output,
        mcp_problems=report.mcp_problems,
        plan_report=report,
    )
    ensure_result_fingerprint(result)
    return result


def compact_task_reports(reports: list[PlanReport], *, max_reports: int = 3) -> str:
    lines: list[str] = []
    for report in reports[:max_reports]:
        line = (
            f"- plan_v{report.plan_version} | worker={report.worker_id} "
            f"| status={report.status} | verdict={report.verdict}"
        )
        lines.append(line)
        if report.what_was_done:
            done = report.what_was_done.strip()
            if len(done) > 160:
                done = done[:145].rstrip() + "\n...[truncated]"
            lines.append(f"  done: {done}")
        if report.key_metrics:
            metrics = ", ".join(f"{k}={v}" for k, v in list(report.key_metrics.items())[:6])
            lines.append(f"  metrics: {metrics}")
        if report.results_table:
            row = report.results_table[0]
            row_view = ", ".join(f"{k}={row.get(k)}" for k in list(row.keys())[:6])
            lines.append(f"  row0: {row_view}")
        if report.error:
            error = report.error.strip()
            if len(error) > 180:
                error = error[:165].rstrip() + "\n...[truncated]"
            lines.append(f"  error: {error}")
    if len(reports) > max_reports:
        lines.append(f"- ... {len(reports) - max_reports} more reports omitted")
    return "\n".join(lines)
