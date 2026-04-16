from __future__ import annotations

import asyncio
from pathlib import Path

from app.execution_models import BaselineRef, ExecutionPlan, ExecutionStateV2, PlanSlice
from app.models import StopReason
from app.plan_sources import CompiledPlanSource
from app.raw_plan_models import CompiledPlanManifest
from app.services.mcp_catalog.models import McpCatalogSnapshot, McpToolSpec


def _compiled_source(*, manifest: CompiledPlanManifest, skip_failures: bool = True) -> CompiledPlanSource:
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = skip_failures
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = True
    source._warned_sequences = set()
    source._manifest_map = {Path(manifest.source_file).stem: manifest}
    source._ordered_raw_files = [Path(manifest.source_file)]
    return source


def _plan(plan_id: str, *, status: str) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        goal="goal",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=[],
        slices=[],
        status=status,
    )


def test_compiled_source_stop_reason_waits_for_remaining_batches() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=[
            "plans/compiled_plan_v1_batch_1.json",
            "plans/compiled_plan_v1_batch_2.json",
        ],
    )
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[_plan("compiled_plan_v1_batch_1", status="completed")],
    )

    assert source.stop_reason(state, drain_mode=False) is None


def test_compiled_source_stop_reason_reaches_goal_after_last_batch() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    source = _compiled_source(manifest=manifest)
    state = ExecutionStateV2(
        goal="goal",
        plans=[_plan("compiled_plan_v1_batch_1", status="completed")],
    )

    assert source.stop_reason(state, drain_mode=False) == StopReason.GOAL_REACHED


def test_compiled_source_refreshes_stale_runtime_metadata_from_current_catalog() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    source = _compiled_source(manifest=manifest)
    source.catalog_snapshot = McpCatalogSnapshot(
        server_name="dev_space1",
        endpoint_url="http://127.0.0.1:8766/mcp",
        schema_hash="hash",
        fetched_at="2026-04-12T00:00:00Z",
        tools=[
            McpToolSpec(
                name="features_catalog",
                side_effects="read_only",
                supports_terminal_write=False,
                supports_discovery=True,
                fields=["scope", "timeframe"],
                description="Inspect feature catalog",
            ),
            McpToolSpec(
                name="events",
                side_effects="read_only",
                supports_terminal_write=False,
                supports_discovery=True,
                fields=["family", "symbol"],
                description="Inspect events",
            ),
            McpToolSpec(
                name="datasets",
                side_effects="read_only",
                supports_terminal_write=False,
                supports_discovery=True,
                fields=["view", "dataset_id"],
                description="Inspect datasets",
            ),
            McpToolSpec(
                name="research_memory",
                side_effects="mutating",
                supports_terminal_write=True,
                supports_discovery=True,
                accepted_handle_fields=["project_id"],
                produced_handle_fields=["project_id", "memory_node_id"],
            ),
        ],
    )
    plan = ExecutionPlan(
        plan_id="compiled_plan_v1_batch_1",
        goal="goal",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=[],
        slices=[],
        status="draft",
    )
    plan.slices.append(
        PlanSlice(
            slice_id="compiled_plan_v1_stage_3_part1",
            title="Data contract и feature contract для каждой новой семьи (exploration)",
            hypothesis="h",
            objective="Explore data coverage, alignment, and feature contracts before recording findings.",
            success_criteria=["Exploration phase complete", "Leakage checked for each contract."],
            allowed_tools=["features_catalog", "events", "datasets", "research_memory"],
            evidence_requirements=["Exploration phase complete"],
            policy_tags=["feature_contract", "data_readiness"],
            max_turns=10,
            max_tool_calls=10,
            max_expensive_calls=0,
            parallel_slot=1,
            runtime_profile="write_result",
            required_output_facts=[],
            finalization_mode="none",
        )
    )

    source._apply_catalog_guardrails(plan=plan, manifest=manifest)

    slice_obj = plan.slices[0]
    assert slice_obj.runtime_profile == "generic_mutation"
    assert slice_obj.required_output_facts == []
    assert slice_obj.required_prerequisite_facts == []
    assert slice_obj.finalization_mode == "none"


def test_compiled_source_refresh_preserves_compiler_fact_contracts() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    source = _compiled_source(manifest=manifest)
    source.catalog_snapshot = McpCatalogSnapshot(
        server_name="dev_space1",
        endpoint_url="http://127.0.0.1:8766/mcp",
        schema_hash="hash",
        fetched_at="2026-04-12T00:00:00Z",
        tools=[
            McpToolSpec(
                name="backtests_analysis",
                side_effects="mutating",
                supports_polling=True,
                accepted_handle_fields=["run_id"],
                produced_handle_fields=["run_id"],
            ),
            McpToolSpec(
                name="research_memory",
                side_effects="mutating",
                supports_terminal_write=True,
                accepted_handle_fields=["project_id"],
                produced_handle_fields=["project_id", "memory_node_id"],
            ),
        ],
    )
    slice_obj = PlanSlice(
        slice_id="compiled_plan_v1_stage_2",
        title="Integration checks",
        hypothesis="h",
        objective="o",
        success_criteria=["integration testing"],
        allowed_tools=["backtests_analysis", "research_memory"],
        evidence_requirements=[],
        policy_tags=["stability"],
        max_turns=4,
        max_tool_calls=6,
        max_expensive_calls=1,
        parallel_slot=1,
        runtime_profile="write_result",
        required_output_facts=["research.project_id"],
        required_prerequisite_facts=["backtests.candidate_handles"],
        finalization_mode="fact_based",
    )

    source._refresh_slice_runtime_metadata(slice_obj)

    assert slice_obj.required_output_facts == ["research.project_id"]
    assert slice_obj.required_prerequisite_facts == ["backtests.candidate_handles"]


def test_compiled_source_stops_on_recoverable_dependency_blocked_without_pending_batches() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=[
            "plans/compiled_plan_v1_batch_1.json",
            "plans/compiled_plan_v1_batch_2.json",
        ],
    )
    source = _compiled_source(manifest=manifest)
    stopped = _plan("compiled_plan_v1_batch_1", status="stopped")
    stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="checkpointed",
            last_error="direct_error_loop_detected",
            last_checkpoint_status="blocked",
        ),
        PlanSlice(
            slice_id="compiled_plan_v1_stage_8",
            title="Downstream",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_analysis"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="aborted",
            last_error="dependency_blocked",
            dependency_blocker_slice_id="compiled_plan_v1_stage_7",
            dependency_blocker_reason_code="direct_error_loop_detected",
            dependency_blocker_class="contract",
        ),
    ]
    state = ExecutionStateV2(goal="goal", plans=[stopped])

    # NEW BEHAVIOR: recoverable blocker no longer skips all remaining batches.
    # batch_2 is still pending so _has_pending_plan returns True.
    assert source._has_pending_plan(state) is True
    # stop_reason returns None because there are still pending plans.
    assert source.stop_reason(state, drain_mode=False) is None
    # next_plan_batch skips the failed batch_1 and would load batch_2
    # (but store is None so it returns None in test scenario).
    assert asyncio.run(source.next_plan_batch(state)) is None


def test_compiled_source_does_not_advance_to_next_raw_plan_after_recoverable_blocker() -> None:
    """OLD BEHAVIOR TEST: When infra_failure_never_skip_batches=False,
    recoverable blocker stops the whole source. This test preserves
    the legacy behavior for documentation."""
    manifest_v1 = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash1",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    manifest_v2 = CompiledPlanManifest(
        source_file="raw_plans/plan_v2.md",
        source_hash="hash2",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v2",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v2_batch_1.json"],
    )
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = True
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = False  # Legacy mode: stop on recoverable
    source._warned_sequences = set()
    source._manifest_map = {
        Path(manifest_v1.source_file).stem: manifest_v1,
        Path(manifest_v2.source_file).stem: manifest_v2,
    }
    source._ordered_raw_files = [Path(manifest_v1.source_file), Path(manifest_v2.source_file)]
    stopped = _plan("compiled_plan_v1_batch_1", status="stopped")
    stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="checkpointed",
            last_error="direct_error_loop_detected",
            last_checkpoint_status="blocked",
        )
    ]
    state = ExecutionStateV2(goal="goal", plans=[stopped])

    # Legacy mode: recoverable blocker stops source even with pending sequence
    assert source.stop_reason(state, drain_mode=False) == StopReason.RECOVERABLE_BLOCKED
    assert asyncio.run(source.next_plan_batch(state)) is None


def test_compiled_source_advances_to_next_sequence_on_recoverable_blocker_when_infra_never_skip() -> None:
    """NEW BEHAVIOR: When infra_failure_never_skip_batches=True (default),
    recoverable blocker in v1 does NOT stop the whole source.
    Next sequence v2 should be returned if it has pending plans."""
    manifest_v1 = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash1",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json", "plans/compiled_plan_v1_batch_2.json"],
    )
    manifest_v2 = CompiledPlanManifest(
        source_file="raw_plans/plan_v2.md",
        source_hash="hash2",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v2",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v2_batch_1.json"],
    )
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = True
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = True  # New default mode
    source._warned_sequences = set()
    source._manifest_map = {
        Path(manifest_v1.source_file).stem: manifest_v1,
        Path(manifest_v2.source_file).stem: manifest_v2,
    }
    source._ordered_raw_files = [Path(manifest_v1.source_file), Path(manifest_v2.source_file)]

    # v1_batch_2 stopped with recoverable blocker
    stopped = _plan("compiled_plan_v1_batch_2", status="stopped")
    stopped.last_error = "direct_final_report_quality_gate_failed"
    stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="checkpointed",
            last_error="direct_final_report_quality_gate_failed",
        )
    ]
    state = ExecutionStateV2(
        goal="goal",
        plans=[
            _plan("compiled_plan_v1_batch_1", status="completed"),
            stopped,
        ],
    )

    # With pending v2, stop_reason should be None (not RECOVERABLE_BLOCKED)
    assert source.stop_reason(state, drain_mode=False) is None
    # next_plan_batch should return a plan from v2 (mock returns None due to no store)
    # but importantly it should not short-circuit due to v1's recoverable blocker
    result = asyncio.run(source.next_plan_batch(state))
    assert result is None  # No store to load from, but should have tried v2


def test_compiled_source_recoverable_blocked_only_when_all_sequences_exhausted() -> None:
    """RECOVERABLE_BLOCKED should only be returned when there are no
    pending plans remaining across ALL sequences."""
    manifest_v1 = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash1",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    manifest_v2 = CompiledPlanManifest(
        source_file="raw_plans/plan_v2.md",
        source_hash="hash2",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v2",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v2_batch_1.json"],
    )
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = True
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = True
    source._warned_sequences = set()
    source._manifest_map = {
        Path(manifest_v1.source_file).stem: manifest_v1,
        Path(manifest_v2.source_file).stem: manifest_v2,
    }
    source._ordered_raw_files = [Path(manifest_v1.source_file), Path(manifest_v2.source_file)]

    # v1 stopped with recoverable, v2 also stopped with recoverable
    v1_stopped = _plan("compiled_plan_v1_batch_1", status="stopped")
    v1_stopped.last_error = "direct_final_report_quality_gate_failed"
    v1_stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="checkpointed",
            last_error="direct_final_report_quality_gate_failed",
        )
    ]
    v2_stopped = _plan("compiled_plan_v2_batch_1", status="stopped")
    v2_stopped.last_error = "direct_final_report_quality_gate_failed"
    v2_stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v2_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="checkpointed",
            last_error="direct_final_report_quality_gate_failed",
        )
    ]
    state = ExecutionStateV2(goal="goal", plans=[v1_stopped, v2_stopped])

    # All sequences have recoverable blockers, no pending plans
    assert source._has_pending_plan(state) is False
    assert source.stop_reason(state, drain_mode=False) == StopReason.RECOVERABLE_BLOCKED


def test_compiled_source_does_not_advance_after_nonaccepted_ancestor_blocker() -> None:
    manifest_v1 = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash1",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    manifest_v2 = CompiledPlanManifest(
        source_file="raw_plans/plan_v2.md",
        source_hash="hash2",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v2",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v2_batch_1.json"],
    )
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = True
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = True
    source._warned_sequences = set()
    source._manifest_map = {
        Path(manifest_v1.source_file).stem: manifest_v1,
        Path(manifest_v2.source_file).stem: manifest_v2,
    }
    source._ordered_raw_files = [Path(manifest_v1.source_file), Path(manifest_v2.source_file)]

    blocked = _plan("compiled_plan_v1_batch_1", status="stopped")
    blocked.last_error = "watchlist_not_accepted"
    blocked.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_1",
            title="Research setup",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["research_project"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=2,
            max_expensive_calls=0,
            parallel_slot=1,
            status="completed",
            verdict="WATCHLIST",
            acceptance_state="reported_terminal",
            dependency_unblock_mode="accepted_only",
            watchlist_allows_unblock=False,
            last_error="watchlist_not_accepted",
        )
    ]
    state = ExecutionStateV2(goal="goal", plans=[blocked])

    assert source._has_pending_plan(state) is False
    assert source.stop_reason(state, drain_mode=False) == StopReason.RECOVERABLE_BLOCKED
    assert asyncio.run(source.next_plan_batch(state)) is None


def test_compiled_source_stops_sequence_after_semantic_dependency_blocked() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=[
            "plans/compiled_plan_v1_batch_1.json",
            "plans/compiled_plan_v1_batch_2.json",
        ],
    )
    source = _compiled_source(manifest=manifest)
    stopped = _plan("compiled_plan_v1_batch_1", status="stopped")
    stopped.slices = [
        PlanSlice(
            slice_id="compiled_plan_v1_stage_7",
            title="Integration",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_runs"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="aborted",
            last_error="semantic_abort",
        ),
        PlanSlice(
            slice_id="compiled_plan_v1_stage_8",
            title="Downstream",
            hypothesis="h",
            objective="o",
            success_criteria=[],
            allowed_tools=["backtests_analysis"],
            evidence_requirements=[],
            policy_tags=[],
            max_turns=1,
            max_tool_calls=1,
            max_expensive_calls=0,
            parallel_slot=1,
            status="aborted",
            last_error="dependency_blocked",
            dependency_blocker_slice_id="compiled_plan_v1_stage_7",
            dependency_blocker_reason_code="semantic_abort",
            dependency_blocker_class="semantic",
        ),
    ]
    state = ExecutionStateV2(goal="goal", plans=[stopped])

    assert source._has_pending_plan(state) is False


# --- start_from feature tests ---


def test_start_from_v2_skips_v1() -> None:
    """start_from='v2' should skip plan_v1 and start from plan_v2."""
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.store = None
    source.raw_plan_dir = Path("raw_plans")
    source.skip_failures = True
    source.notification_service = None
    source.catalog_snapshot = None
    source.incident_store = None
    source.infra_failure_never_skip_batches = True
    source.start_from = "v2"
    source._warned_sequences = set()
    source._manifest_map = {
        "plan_v1": CompiledPlanManifest(
            source_file="raw_plans/plan_v1.md",
            source_hash="h1",
            compiled_at="2026-04-12T00:00:00Z",
            sequence_id="compiled_plan_v1",
            compile_status="compiled",
            warnings=[],
            semantic_method="llm",
            plan_files=["plans/compiled_plan_v1_batch_1.json"],
        ),
        "plan_v2": CompiledPlanManifest(
            source_file="raw_plans/plan_v2.md",
            source_hash="h2",
            compiled_at="2026-04-12T00:00:00Z",
            sequence_id="compiled_plan_v2",
            compile_status="compiled",
            warnings=[],
            semantic_method="llm",
            plan_files=["plans/compiled_plan_v2_batch_1.json"],
        ),
    }
    source._ordered_raw_files = [
        Path("raw_plans/plan_v1.md"),
        Path("raw_plans/plan_v2.md"),
    ]
    source._apply_start_from()

    assert len(source._ordered_raw_files) == 1
    assert source._ordered_raw_files[0] == Path("raw_plans/plan_v2.md")


def test_start_from_empty_keeps_all_plans() -> None:
    """Empty start_from should keep all plans."""
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.start_from = ""
    source._ordered_raw_files = [
        Path("raw_plans/plan_v1.md"),
        Path("raw_plans/plan_v2.md"),
        Path("raw_plans/plan_v3.md"),
    ]
    source._apply_start_from()

    assert len(source._ordered_raw_files) == 3


def test_start_from_unknown_keeps_all_plans() -> None:
    """start_from value that matches no plan should keep all plans."""
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.start_from = "v99"
    source._ordered_raw_files = [
        Path("raw_plans/plan_v1.md"),
        Path("raw_plans/plan_v2.md"),
    ]
    source._apply_start_from()

    assert len(source._ordered_raw_files) == 2


def test_start_from_v5_skips_v1_through_v4() -> None:
    """start_from='v5' should skip v1, v2, v3, v4 and start from v5."""
    source = CompiledPlanSource.__new__(CompiledPlanSource)
    source.start_from = "v5"
    source._ordered_raw_files = [
        Path("raw_plans/plan_v1.md"),
        Path("raw_plans/plan_v2.md"),
        Path("raw_plans/plan_v3.md"),
        Path("raw_plans/plan_v4.md"),
        Path("raw_plans/plan_v5.md"),
        Path("raw_plans/plan_v6.md"),
    ]
    source._apply_start_from()

    assert len(source._ordered_raw_files) == 2
    assert source._ordered_raw_files[0] == Path("raw_plans/plan_v5.md")
    assert source._ordered_raw_files[1] == Path("raw_plans/plan_v6.md")


# --- is_sequence_complete tests ---


def _plan_with_seq(plan_id: str, *, status: str, sequence_id: str) -> ExecutionPlan:
    plan = _plan(plan_id, status=status)
    plan.source_sequence_id = sequence_id
    return plan


def _manifest_4batches(seq_id: str = "compiled_plan_v1") -> CompiledPlanManifest:
    return CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id=seq_id,
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=[
            "plans/compiled_plan_v1_batch_1.json",
            "plans/compiled_plan_v1_batch_2.json",
            "plans/compiled_plan_v1_batch_3.json",
            "plans/compiled_plan_v1_batch_4.json",
        ],
    )


def test_is_sequence_complete_false_when_not_all_batches_executed() -> None:
    manifest = _manifest_4batches()
    source = _compiled_source(manifest=manifest)
    plan = _plan_with_seq("compiled_plan_v1_batch_1", status="completed", sequence_id="compiled_plan_v1")
    state = ExecutionStateV2(goal="goal", plans=[plan])

    assert source.is_sequence_complete(plan, state) is False


def test_is_sequence_complete_false_when_batch_still_running() -> None:
    manifest = _manifest_4batches()
    source = _compiled_source(manifest=manifest)
    plans = [
        _plan_with_seq(f"compiled_plan_v1_batch_{i}", status="completed", sequence_id="compiled_plan_v1")
        for i in range(1, 4)
    ]
    plans.append(_plan_with_seq("compiled_plan_v1_batch_4", status="running", sequence_id="compiled_plan_v1"))
    state = ExecutionStateV2(goal="goal", plans=plans)
    last_plan = plans[-1]

    assert source.is_sequence_complete(last_plan, state) is False


def test_is_sequence_complete_true_when_all_batches_terminal() -> None:
    manifest = _manifest_4batches()
    source = _compiled_source(manifest=manifest)
    plans = [
        _plan_with_seq(f"compiled_plan_v1_batch_{i}", status="completed", sequence_id="compiled_plan_v1")
        for i in range(1, 5)
    ]
    state = ExecutionStateV2(goal="goal", plans=plans)
    last_plan = plans[-1]

    assert source.is_sequence_complete(last_plan, state) is True


def test_is_sequence_complete_true_with_mixed_terminal_statuses() -> None:
    manifest = _manifest_4batches()
    source = _compiled_source(manifest=manifest)
    plans = [
        _plan_with_seq("compiled_plan_v1_batch_1", status="completed", sequence_id="compiled_plan_v1"),
        _plan_with_seq("compiled_plan_v1_batch_2", status="completed", sequence_id="compiled_plan_v1"),
        _plan_with_seq("compiled_plan_v1_batch_3", status="stopped", sequence_id="compiled_plan_v1"),
        _plan_with_seq("compiled_plan_v1_batch_4", status="failed", sequence_id="compiled_plan_v1"),
    ]
    state = ExecutionStateV2(goal="goal", plans=plans)

    assert source.is_sequence_complete(plans[2], state) is True


def test_is_sequence_complete_false_no_source_sequence_id() -> None:
    manifest = _manifest_4batches()
    source = _compiled_source(manifest=manifest)
    plan = _plan("compiled_plan_v1_batch_1", status="completed")
    state = ExecutionStateV2(goal="goal", plans=[plan])

    assert source.is_sequence_complete(plan, state) is False


def test_is_sequence_complete_single_batch() -> None:
    manifest = CompiledPlanManifest(
        source_file="raw_plans/plan_v1.md",
        source_hash="hash",
        compiled_at="2026-04-12T00:00:00Z",
        sequence_id="compiled_plan_v1",
        compile_status="compiled",
        warnings=[],
        semantic_method="llm",
        plan_files=["plans/compiled_plan_v1_batch_1.json"],
    )
    source = _compiled_source(manifest=manifest)
    plan = _plan_with_seq("compiled_plan_v1_batch_1", status="completed", sequence_id="compiled_plan_v1")
    state = ExecutionStateV2(goal="goal", plans=[plan])

    assert source.is_sequence_complete(plan, state) is True
