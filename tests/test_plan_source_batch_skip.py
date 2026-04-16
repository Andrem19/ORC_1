"""Tests for granular batch skip after recoverable blocker (Fix 1)."""
from __future__ import annotations

from pathlib import Path

from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice
from app.models import StopReason
from app.plan_sources import CompiledPlanSource
from app.raw_plan_models import CompiledPlanManifest


def _compiled_source(
    *,
    manifest: CompiledPlanManifest,
    skip_failures: bool = True,
    infra_failure_never_skip_batches: bool = True,
) -> CompiledPlanSource:
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = skip_failures
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = infra_failure_never_skip_batches
    source._warned_sequences = set()
    source._manifest_map = {Path(manifest.source_file).stem: manifest}
    source._ordered_raw_files = [Path(manifest.source_file)]
    return source


def _failed_plan(plan_id: str, *, last_error: str = "zero_tool_calls") -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal="goal",
        baseline_ref=BaselineRef(snapshot_id="snap", version=1),
        global_constraints=[],
        slices=[
            PlanSlice(
                slice_id="s1",
                title="t",
                hypothesis="h",
                objective="o",
                success_criteria=[],
                allowed_tools=[],
                evidence_requirements=[],
                policy_tags=[],
                max_turns=5,
                max_tool_calls=5,
                max_expensive_calls=0,
                parallel_slot=1,
                last_error=last_error,
            ),
        ],
        status="failed",
        last_error=last_error,
    )


def _completed_plan(plan_id: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal="goal",
        baseline_ref=BaselineRef(snapshot_id="snap", version=1),
        global_constraints=[],
        slices=[],
        status="completed",
    )


def _manifest_with_5_batches() -> CompiledPlanManifest:
    return CompiledPlanManifest(
        source_file="raw_plans/plan_v2.md",
        source_hash="hash",
        compiled_at="2026-04-14T00:00:00Z",
        sequence_id="compiled_plan_v2",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=[
            "plans/compiled_plan_v2_batch_1.json",
            "plans/compiled_plan_v2_batch_2.json",
            "plans/compiled_plan_v2_batch_3.json",
            "plans/compiled_plan_v2_batch_4.json",
            "plans/compiled_plan_v2_batch_5.json",
        ],
    )


def test_recoverable_failure_has_pending_returns_true_for_remaining_batches() -> None:
    """After batch_4 fails with a recoverable error, _has_pending_plan returns True
    because batch_5 is still pending."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _completed_plan("compiled_plan_v2_batch_1"),
            _completed_plan("compiled_plan_v2_batch_2"),
            _completed_plan("compiled_plan_v2_batch_3"),
            _failed_plan("compiled_plan_v2_batch_4", last_error="zero_tool_calls"),
        ],
    )
    assert source._has_pending_plan(state) is True


def test_recoverable_failure_has_pending_returns_false_when_all_done() -> None:
    """When all batches are executed (completed or failed), _has_pending_plan
    returns False."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _completed_plan("compiled_plan_v2_batch_1"),
            _completed_plan("compiled_plan_v2_batch_2"),
            _completed_plan("compiled_plan_v2_batch_3"),
            _failed_plan("compiled_plan_v2_batch_4", last_error="zero_tool_calls"),
            _completed_plan("compiled_plan_v2_batch_5"),
        ],
    )
    assert source._has_pending_plan(state) is False


def test_semantic_abort_still_skips_remaining_batches() -> None:
    """Semantic abort should still skip the entire remaining sequence."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(manifest=manifest, skip_failures=True)
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _completed_plan("compiled_plan_v2_batch_1"),
            _completed_plan("compiled_plan_v2_batch_2"),
            _completed_plan("compiled_plan_v2_batch_3"),
            _failed_plan("compiled_plan_v2_batch_4", last_error="semantic_abort"),
        ],
    )
    # Semantic failure should NOT report pending plans even though batch_5 exists
    assert source._has_pending_plan(state) is False


def test_infra_failure_never_skip_false_returns_no_pending() -> None:
    """When infra_failure_never_skip_batches=False, old behavior: no pending."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(
        manifest=manifest,
        infra_failure_never_skip_batches=False,
    )
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _completed_plan("compiled_plan_v2_batch_1"),
            _failed_plan("compiled_plan_v2_batch_2", last_error="missing_domain_tool_evidence"),
        ],
    )
    # Old behavior: recoverable blocker + infra_failure_never_skip_batches=False
    # means no pending plans (return False).
    assert source._has_pending_plan(state) is False


def test_missing_domain_tool_evidence_is_recoverable() -> None:
    """missing_domain_tool_evidence should be classified as infra (recoverable),
    not semantic."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _completed_plan("compiled_plan_v2_batch_1"),
            _completed_plan("compiled_plan_v2_batch_2"),
            _completed_plan("compiled_plan_v2_batch_3"),
            _failed_plan("compiled_plan_v2_batch_4", last_error="missing_domain_tool_evidence"),
        ],
    )
    # Should see batch_5 as still pending (recoverable, not semantic)
    assert source._has_pending_plan(state) is True


def test_multiple_recoverable_failures_still_pending() -> None:
    """Multiple recoverable failures should still allow remaining batches."""
    manifest = _manifest_with_5_batches()
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _failed_plan("compiled_plan_v2_batch_1", last_error="zero_tool_calls"),
            _failed_plan("compiled_plan_v2_batch_2", last_error="missing_domain_tool_evidence"),
        ],
    )
    assert source._has_pending_plan(state) is True
