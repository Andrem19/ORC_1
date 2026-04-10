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
from app.services.direct_execution.planner import PlannerDecisionService

logger = logging.getLogger("orchestrator.plan_source")


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
    ) -> None:
        self.store = store
        self.raw_plan_dir = Path(raw_plan_dir)
        self.skip_failures = skip_failures
        self.notification_service = notification_service
        self._warned_sequences: set[str] = set()
        self._manifest_map = {Path(item.source_file).stem: item for item in self.store.load_manifests()}
        self._ordered_raw_files = sorted(self.raw_plan_dir.glob("*.md"), key=raw_plan_sort_key)

    async def next_plan_batch(self, state: ExecutionStateV2) -> ExecutionPlan | None:
        executed = {plan.plan_id: plan.status for plan in state.plans}
        for raw_file in self._ordered_raw_files:
            manifest = self._manifest_map.get(raw_file.stem)
            if manifest is None:
                self._warn_once(raw_file.stem, f"Skipping raw plan without compiled manifest: {raw_file.name}")
                continue
            if manifest.compile_status != "compiled":
                self._warn_once(manifest.sequence_id, f"Skipping compiled sequence {manifest.sequence_id}: compile_status={manifest.compile_status}")
                continue
            sequence_plan_ids = [plan_file.rsplit("/", 1)[-1].replace(".json", "") for plan_file in manifest.plan_files]
            if any(executed.get(plan_id) in {"failed", "stopped"} for plan_id in sequence_plan_ids):
                self._warn_once(manifest.sequence_id, f"Skipping remaining batches for {manifest.sequence_id}: prior execution stopped or failed")
                if self.skip_failures:
                    continue
                return None
            for plan_file, plan_id in zip(manifest.plan_files, sequence_plan_ids):
                if plan_id in executed:
                    continue
                try:
                    return self.store.load_plan(manifest, plan_file)
                except FileNotFoundError:
                    self._warn_once(manifest.sequence_id, f"Skipping missing compiled plan file: {manifest.sequence_id}/{plan_file}")
                    break
        return None

    def mark_plan_complete(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        del plan, state

    def mark_plan_failed(self, plan: ExecutionPlan, state: ExecutionStateV2) -> None:
        del plan, state

    def stop_reason(self, state: ExecutionStateV2, *, drain_mode: bool) -> StopReason | None:
        if drain_mode:
            return StopReason.GRACEFUL_STOP
        if not self._ordered_raw_files:
            return StopReason.GOAL_IMPOSSIBLE
        completed = any(plan.status == "completed" for plan in state.plans)
        return StopReason.GOAL_REACHED if completed else StopReason.GOAL_IMPOSSIBLE

    def summary(self) -> str:
        return f"compiled_raw:raw_files={len(self._ordered_raw_files)} manifests={len(self._manifest_map)} skip_failures={self.skip_failures}"

    def _warn_once(self, sequence_id: str, message: str) -> None:
        if sequence_id in self._warned_sequences:
            return
        self._warned_sequences.add(sequence_id)
        logger.warning(message)
        if self.notification_service is not None:
            self.notification_service.send_lifecycle("compiled_plan_warning", message)
