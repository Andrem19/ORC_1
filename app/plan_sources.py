"""
Plan source abstractions for planner-driven and compiled-raw runtimes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.compiled_plan_store import CompiledPlanStore
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ExecutionPlan, ExecutionStateV2
from app.models import StopReason
from app.raw_plan_ordering import raw_plan_sort_key
from app.runtime_incidents import LocalIncidentStore
from app.services.direct_execution.blocker_classification import blocker_class_from_slice, classify_blocker
from app.services.direct_execution.acceptance import slice_requires_strict_acceptance
from app.services.direct_execution.runtime_profiles import resolve_runtime_slice_metadata_with_prerequisites
from app.services.mcp_catalog.models import McpCatalogSnapshot
from app.services.direct_execution.planner import PlannerDecisionService

logger = logging.getLogger("orchestrator.plan_source")


# Explicit markers that mean the plan (or its slices) hit a semantic wall —
# continuing the rest of the compiled sequence would be meaningless.
_SEMANTIC_ABORT_REASON_MARKERS = (
    "semantic_abort",
    "user_stop",
    "goal_impossible",
    "plan_contract_violation",
    "explicit_abort",
)

# Recoverable markers — the plan failed because of a crash, timeout, gate
# rejection or contract issue. These are not semantic proof, but should stop
# the sequence for operator retry instead of auto-advancing.
_TRANSIENT_INFRA_REASON_MARKERS = (
    "lmstudio_model_crash",
    "lmstudio_http_timeout",
    "lmstudio__chat_timeout",
    "adapter_invoke_timeout",
    "adapter_timeout",
    "qwen_cli_timeout",
    "claude_cli_timeout",
    "checkpoint_blocked",
    "zero_tool_calls",
    "empty_evidence_refs",
    "auto_salvage_stub_rejected",
    "watchlist_confidence",
    "missing_domain_tool_evidence",
    "direct_output_parse_failed",
    "direct_model_stalled_before_first_action",
    "direct_model_stalled_between_actions",
    "direct_tool_budget_exhausted",
    "direct_expensive_tool_budget_exhausted",
)


def _classify_plan_failure(plan: ExecutionPlan) -> str:
    """Return ``"semantic"`` | recoverable class | ``"unknown"`` for a failed plan.

    Looks at both ``plan.last_error`` and per-slice ``last_error`` fields so we
    still classify correctly for plans created before ``last_error`` existed on
    ``ExecutionPlan``.
    """
    haystacks: list[str] = []
    plan_error = str(getattr(plan, "last_error", "") or "").strip().lower()
    if plan_error:
        haystacks.append(plan_error)
    for slice_obj in getattr(plan, "slices", []) or []:
        slice_error = str(getattr(slice_obj, "last_error", "") or "").strip().lower()
        if slice_error:
            haystacks.append(slice_error)
    if not haystacks:
        return "unknown"
    slice_classes = [blocker_class_from_slice(slice_obj) for slice_obj in getattr(plan, "slices", []) or []]
    if any(item == "semantic" for item in slice_classes):
        return "semantic"
    for item in ("provider_limit", "contract", "infra"):
        if any(slice_class == item for slice_class in slice_classes):
            return item
    combined = " | ".join(haystacks)
    direct_class = classify_blocker(reason_code=plan_error, summary=combined)
    if direct_class == "semantic":
        return "semantic"
    if direct_class in {"contract", "infra", "provider_limit"}:
        return direct_class
    if any(marker in combined for marker in _SEMANTIC_ABORT_REASON_MARKERS):
        return "semantic"
    if any(marker in combined for marker in _TRANSIENT_INFRA_REASON_MARKERS):
        return "infra"
    return "unknown"


def _plan_has_non_skippable_acceptance_blocker(plan: ExecutionPlan) -> bool:
    for slice_obj in getattr(plan, "slices", []) or []:
        if str(getattr(slice_obj, "acceptance_state", "") or "").strip().lower() != "reported_terminal":
            continue
        if not slice_requires_strict_acceptance(slice_obj):
            continue
        return True
    return False


class PlanSource(ABC):
    @abstractmethod
    async def next_plan_batch(self, state: ExecutionStateV2) -> ExecutionPlan | None:
        raise NotImplementedError

    @abstractmethod
    def mark_plan_complete(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_plan_failed(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop_reason(self, state: ExecutionStateV2, *, drain_mode: bool) -> StopReason | None:
        raise NotImplementedError

    @abstractmethod
    def summary(self) -> str:
        raise NotImplementedError


class PlannerPlanSource(PlanSource):
    def __init__(
        self,
        *,
        planner: PlannerDecisionService,
        goal: str,
        baseline_bootstrap: dict[str, Any],
        max_worker_count: int,
        max_plans_per_run: int,
        available_tools_provider: Any,
        state_summary_provider: Any,
    ) -> None:
        self.planner = planner
        self.goal = goal
        self.baseline_bootstrap = baseline_bootstrap
        self.max_worker_count = max_worker_count
        self.max_plans_per_run = max(1, int(max_plans_per_run or 1))
        self.available_tools_provider = available_tools_provider
        self.state_summary_provider = state_summary_provider

    async def next_plan_batch(self, state: ExecutionStateV2) -> ExecutionPlan | None:
        if len(state.plans) >= self.max_plans_per_run:
            return None
        return await self.planner.create_plan(
            goal=self.goal,
            baseline_bootstrap=self.baseline_bootstrap,
            available_tools=sorted(self.available_tools_provider()),
            worker_count=self.max_worker_count,
            plan_version=len(state.plans) + 1,
            previous_state_summary=self.state_summary_provider(),
            previous_blockers=self._recent_blockers(state),
        )

    def mark_plan_complete(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        del state
        self.planner.save_plan_snapshot(plan)

    def mark_plan_failed(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        del state
        self.planner.save_plan_snapshot(plan)

    def stop_reason(self, state: ExecutionStateV2, *, drain_mode: bool) -> StopReason | None:
        if drain_mode:
            return StopReason.GRACEFUL_STOP
        if len(state.plans) < self.max_plans_per_run:
            return None
        last_plan = state.plans[-1] if state.plans else None
        if last_plan is not None and last_plan.status == "completed":
            return StopReason.GOAL_REACHED
        return StopReason.GOAL_IMPOSSIBLE

    def summary(self) -> str:
        return f"planner:max_plans={self.max_plans_per_run}"

    @staticmethod
    def _recent_blockers(state: ExecutionStateV2) -> list[str]:
        blockers: list[str] = []
        for plan in state.plans[-3:]:
            if plan.status == "failed":
                blockers.append(f"plan_failed:{plan.plan_id}")
            for slice_obj in plan.slices:
                if slice_obj.last_error:
                    blockers.append(f"{slice_obj.slice_id}:{slice_obj.last_error}")
        return blockers[-8:]


class CompiledPlanSource(PlanSource):
    def __init__(
        self,
        *,
        store: CompiledPlanStore,
        raw_plan_dir: str,
        skip_failures: bool,
        notification_service: Any | None = None,
        catalog_snapshot: McpCatalogSnapshot | None = None,
        incident_store: LocalIncidentStore | None = None,
        infra_failure_never_skip_batches: bool = True,
        start_from: str = "",
    ) -> None:
        self.store = store
        self.raw_plan_dir = Path(raw_plan_dir)
        self.skip_failures = skip_failures
        self.infra_failure_never_skip_batches = infra_failure_never_skip_batches
        self.notification_service = notification_service
        self.catalog_snapshot = catalog_snapshot
        self.incident_store = incident_store
        self.start_from = start_from.strip()
        self._warned_sequences: set[str] = set()
        self._manifest_map = {Path(item.source_file).stem: item for item in self.store.load_manifests()}
        self._ordered_raw_files = sorted(self.raw_plan_dir.glob("*.md"), key=raw_plan_sort_key)
        self._apply_start_from()

    def _apply_start_from(self) -> None:
        """Filter ``_ordered_raw_files`` to start from the plan specified in config.

        Accepts shorthand values like ``"v2"`` which map to raw files matching
        ``plan_v2.md``.  If the value is empty or no matching file is found, the
        full ordered list is kept unchanged.
        """
        if not self.start_from:
            return
        target = self.start_from.lower().lstrip("v")
        # Build candidate stems: "v2" -> match files like "plan_v2.md"
        for raw_file in self._ordered_raw_files:
            stem = raw_file.stem.lower()
            # Extract the numeric version part from stems like "plan_v2"
            version_part = stem.split("_v", 1)[-1] if "_v" in stem else stem
            if version_part == target:
                idx = self._ordered_raw_files.index(raw_file)
                skipped = idx
                self._ordered_raw_files = self._ordered_raw_files[idx:]
                if skipped:
                    logger.info(
                        "start_from=%s: skipping %d plan(s), starting from %s",
                        self.start_from, skipped, raw_file.name,
                    )
                return
        logger.warning(
            "start_from=%s: no matching compiled plan found, using full sequence",
            self.start_from,
        )

    async def next_plan_batch(self, state: ExecutionStateV2) -> ExecutionPlan | None:
        executed = {plan.plan_id: plan.status for plan in state.plans}
        plans_by_id = {plan.plan_id: plan for plan in state.plans}
        for raw_file in self._ordered_raw_files:
            manifest = self._manifest_map.get(raw_file.stem)
            if manifest is None:
                self._warn_once(raw_file.stem, f"Skipping raw plan without compiled manifest: {raw_file.name}")
                continue
            if manifest.compile_status != "compiled":
                self._warn_once(manifest.sequence_id, f"Skipping compiled sequence {manifest.sequence_id}: compile_status={manifest.compile_status}")
                continue
            sequence_plan_ids = [plan_file.rsplit("/", 1)[-1].replace(".json", "") for plan_file in manifest.plan_files]
            failed_plan_ids = [pid for pid in sequence_plan_ids if executed.get(pid) in {"failed", "stopped"}]
            if failed_plan_ids:
                failed_plans = [plans_by_id[pid] for pid in failed_plan_ids if pid in plans_by_id]
                if any(_plan_has_non_skippable_acceptance_blocker(plan) for plan in failed_plans):
                    self._warn_once(
                        f"{manifest.sequence_id}:acceptance_blocked",
                        (
                            f"Acceptance blocker in {manifest.sequence_id} batches {failed_plan_ids}; "
                            "not skipping downstream batches because an ancestor slice was not accepted."
                        ),
                    )
                    return None
                classifications = [_classify_plan_failure(plan) for plan in failed_plans]
                semantic_hit = any(cls == "semantic" for cls in classifications)
                recoverable_hit = any(cls in {"contract", "infra", "provider_limit", "unknown"} for cls in classifications)
                if semantic_hit:
                    self._warn_once(
                        manifest.sequence_id,
                        (
                            f"Skipping remaining batches for {manifest.sequence_id}: "
                            f"semantic abort in {failed_plan_ids}"
                        ),
                    )
                    if self.skip_failures:
                        continue
                    return None
                if recoverable_hit:
                    self._warn_once(
                        f"{manifest.sequence_id}:recoverable_blocked",
                        (
                            f"Recoverable blocker in {manifest.sequence_id} batches {failed_plan_ids}; "
                            f"skipping failed batches and continuing with remaining batches."
                        ),
                    )
                    # Do NOT continue/return here.  Fall through to the inner loop
                    # (line below) which skips already-executed plan_ids and
                    # loads the next unexecuted batch in the same sequence.
                else:
                    self._warn_once(
                        manifest.sequence_id,
                        f"Skipping remaining batches for {manifest.sequence_id}: prior execution stopped or failed",
                    )
                    if self.skip_failures:
                        continue
                    return None
            for plan_file, plan_id in zip(manifest.plan_files, sequence_plan_ids):
                if plan_id in executed:
                    continue
                if self.store is None:
                    # Test scenario: no store available, cannot load plan
                    continue
                try:
                    plan = self.store.load_plan(manifest, plan_file)
                    self._apply_catalog_guardrails(plan=plan, manifest=manifest)
                    return plan
                except FileNotFoundError:
                    self._warn_once(manifest.sequence_id, f"Skipping missing compiled plan file: {manifest.sequence_id}/{plan_file}")
                    break
        return None

    def mark_plan_complete(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        del plan, state

    def is_sequence_complete(self, plan: ExecutionPlan, state: ExecutionStateV2) -> bool:
        """Return True when every compiled batch for the same sequence is terminal.

        Compares plans already recorded in *state* against the manifest's full
        batch list.  A sequence is considered complete only when every expected
        batch plan exists in the state **and** is terminal.
        """
        seq_id = getattr(plan, "source_sequence_id", "") or ""
        if not seq_id:
            return False
        manifest = None
        for m in self._manifest_map.values():
            if m.sequence_id == seq_id:
                manifest = m
                break
        if manifest is None:
            return False
        total_expected = len(manifest.plan_files)
        seq_plans = [p for p in state.plans if getattr(p, "source_sequence_id", "") == seq_id]
        if len(seq_plans) < total_expected:
            return False
        return all(p.is_terminal for p in seq_plans)

    def mark_plan_failed(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        if self.notification_service is not None:
            seq = getattr(plan, "source_sequence_id", "") or "?"
            self.notification_service.send_lifecycle(
                "plan_batch_failed",
                f"Batch {plan.plan_id} failed: {plan.last_error or 'unknown'}. Sequence {seq}.",
            )

    def stop_reason(self, state: ExecutionStateV2, *, drain_mode: bool) -> StopReason | None:
        if drain_mode:
            return StopReason.GRACEFUL_STOP
        if not self._ordered_raw_files:
            return StopReason.GOAL_IMPOSSIBLE
        # Recoverable blockers don't stop the run while other sequences still
        # have pending plans — we should advance and only report
        # RECOVERABLE_BLOCKED when nothing else is runnable.
        if self._has_pending_plan(state):
            return None
        if self._has_recoverable_blocker(state):
            return StopReason.RECOVERABLE_BLOCKED
        completed = any(plan.status == "completed" for plan in state.plans)
        return StopReason.GOAL_REACHED if completed else StopReason.GOAL_IMPOSSIBLE

    def summary(self) -> str:
        return f"compiled_raw:raw_files={len(self._ordered_raw_files)} manifests={len(self._manifest_map)} skip_failures={self.skip_failures}"

    def _has_pending_plan(self, state: ExecutionStateV2) -> bool:
        executed = {plan.plan_id: plan.status for plan in state.plans}
        plans_by_id = {plan.plan_id: plan for plan in state.plans}
        for raw_file in self._ordered_raw_files:
            manifest = self._manifest_map.get(raw_file.stem)
            if manifest is None or manifest.compile_status != "compiled":
                continue
            sequence_plan_ids = [plan_file.rsplit("/", 1)[-1].replace(".json", "") for plan_file in manifest.plan_files]
            failed_plan_ids = [pid for pid in sequence_plan_ids if executed.get(pid) in {"failed", "stopped"}]
            if failed_plan_ids:
                failed_plans = [plans_by_id[pid] for pid in failed_plan_ids if pid in plans_by_id]
                if any(_plan_has_non_skippable_acceptance_blocker(plan) for plan in failed_plans):
                    return False
                classifications = [_classify_plan_failure(plan) for plan in failed_plans]
                semantic_hit = any(cls == "semantic" for cls in classifications)
                recoverable_hit = any(cls in {"contract", "infra", "provider_limit", "unknown"} for cls in classifications)
                if semantic_hit:
                    if self.skip_failures:
                        continue
                    return False
                if recoverable_hit:
                    if self.infra_failure_never_skip_batches:
                        # Check if there are still unexecuted batches in this manifest
                        if any(plan_id not in executed for plan_id in sequence_plan_ids):
                            return True
                        continue
                    return False
                if self.skip_failures:
                    continue
                return False
            if any(plan_id not in executed for plan_id in sequence_plan_ids):
                return True
        return False

    def _has_recoverable_blocker(self, state: ExecutionStateV2) -> bool:
        for plan in state.plans:
            if plan.status not in {"failed", "stopped"}:
                continue
            if _classify_plan_failure(plan) in {"contract", "infra", "provider_limit", "unknown"}:
                return True
        return False

    def _warn_once(self, sequence_id: str, message: str) -> None:
        if sequence_id in self._warned_sequences:
            return
        self._warned_sequences.add(sequence_id)
        logger.warning(message)
        if self.notification_service is not None:
            self.notification_service.send_lifecycle("compiled_plan_warning", message)

    def _apply_catalog_guardrails(self, *, plan: ExecutionPlan, manifest: Any) -> None:
        if self.catalog_snapshot is None:
            return
        if getattr(manifest, "mcp_catalog_hash", "") and manifest.mcp_catalog_hash != self.catalog_snapshot.schema_hash:
            self._warn_once(
                f"{manifest.sequence_id}:catalog_hash",
                (
                    f"Compiled sequence {manifest.sequence_id} was built against MCP catalog "
                    f"{manifest.mcp_catalog_hash[:12]}, current startup catalog is {self.catalog_snapshot.schema_hash[:12]}."
                ),
            )
        current_tool_names = self.catalog_snapshot.tool_name_set()
        missing_by_slice: dict[str, list[str]] = {}
        for slice_obj in plan.slices:
            missing = sorted({tool for tool in slice_obj.allowed_tools if str(tool).strip() and str(tool).strip() not in current_tool_names})
            if missing:
                missing_by_slice[slice_obj.slice_id] = missing
                slice_obj.status = "checkpointed"
                slice_obj.last_checkpoint_status = "blocked"
                slice_obj.last_checkpoint_summary = (
                    "Compiled slice blocked by live MCP drift: "
                    f"missing tools -> {', '.join(missing)}."
                )
                slice_obj.last_summary = slice_obj.last_checkpoint_summary
                slice_obj.last_error = "mcp_catalog_tool_missing"
                slice_obj.facts["mcp_catalog.missing_tools"] = missing
                continue
            self._refresh_slice_runtime_metadata(slice_obj)
        if missing_by_slice and self.incident_store is not None:
            self.incident_store.record(
                summary="Compiled plan blocked by live MCP catalog drift",
                metadata={
                    "plan_id": plan.plan_id,
                    "sequence_id": getattr(manifest, "sequence_id", ""),
                    "mcp_catalog_hash": self.catalog_snapshot.schema_hash,
                    "compiled_mcp_catalog_hash": getattr(manifest, "mcp_catalog_hash", ""),
                    "missing_by_slice": missing_by_slice,
                },
                source="compiled_plan_source",
                severity="high",
            )

    def _refresh_slice_runtime_metadata(self, slice_obj: Any) -> None:
        if self.catalog_snapshot is None:
            return
        existing_profile = str(getattr(slice_obj, "runtime_profile", "") or "").strip()
        existing_output_facts = [
            str(item).strip()
            for item in list(getattr(slice_obj, "required_output_facts", None) or [])
            if str(item).strip()
        ]
        existing_prerequisites = [
            str(item).strip()
            for item in list(getattr(slice_obj, "required_prerequisite_facts", None) or [])
            if str(item).strip()
        ]
        existing_finalization = str(getattr(slice_obj, "finalization_mode", "") or "").strip()
        (
            slice_obj.runtime_profile,
            slice_obj.required_output_facts,
            slice_obj.required_prerequisite_facts,
            slice_obj.finalization_mode,
        ) = resolve_runtime_slice_metadata_with_prerequisites(
            runtime_profile=existing_profile,
            required_output_facts=existing_output_facts or None,
            required_prerequisite_facts=existing_prerequisites or None,
            finalization_mode=existing_finalization,
            allowed_tools=list(slice_obj.allowed_tools or []),
            catalog_snapshot=self.catalog_snapshot,
            title=str(slice_obj.title or ""),
            objective=str(slice_obj.objective or ""),
            success_criteria=list(slice_obj.success_criteria or []),
            policy_tags=list(slice_obj.policy_tags or []),
        )
