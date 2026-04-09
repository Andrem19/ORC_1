"""
Deterministic live coverage suite for broker <-> dev_space1 integration.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import monotonic, time
from typing import Any

from app.broker.service import BrokerService
from app.config import OrchestratorConfig
from app.execution_artifacts import ExecutionArtifactStore
from app.execution_models import ToolResultEnvelope
from app.runtime_incidents import LocalIncidentStore

_RUNNING_STATUSES = {
    "queued",
    "pending",
    "running",
    "started",
    "in_progress",
    "running_compute",
    "publishing",
    "publishing_result",
    "persisting",
}


@dataclass
class ScenarioResult:
    name: str
    ok: bool
    tool: str = ""
    summary: str = ""
    response_status: str = ""
    duration_ms: int = 0
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class CoverageCycleResult:
    cycle_index: int
    ok: bool
    scenarios: list[ScenarioResult] = field(default_factory=list)
    skipped_tools: list[str] = field(default_factory=list)
    duration_ms: int = 0


class LiveCoverageRunner:
    def __init__(self, *, config: OrchestratorConfig, run_id: str) -> None:
        self.config = config
        self.run_id = run_id
        self.artifact_store = ExecutionArtifactStore(config.plan_dir, run_id=run_id)
        self.incident_store = LocalIncidentStore(config.state_dir, run_id=run_id)
        self.broker = BrokerService(
            transport_config=config.broker,
            artifact_store=self.artifact_store,
            incident_store=self.incident_store,
            autopoll_budget_seconds=min(config.broker_autopoll_budget_seconds, 2.0),
            autopoll_interval_seconds=min(config.broker_autopoll_interval_seconds, 0.5),
        )
        self.dataset_ids: dict[str, str] = {}
        self.snapshot_id = "active-signal-v1"
        self.snapshot_version = 1
        self.project_id = ""
        self.model_id = ""
        self.model_version = ""
        self.dataset_materialized_id = ""
        self.backtest_run_id = ""
        self.published_custom_feature = ""
        self.signal_binding: dict[str, Any] = {}

    async def run_cycles(self, *, cycles: int) -> list[CoverageCycleResult]:
        await self.broker.bootstrap()
        try:
            return [await self._run_cycle(index + 1) for index in range(cycles)]
        finally:
            await self.broker.close()

    async def _run_cycle(self, cycle_index: int) -> CoverageCycleResult:
        started = monotonic()
        await self._discover_context()
        scenarios: list[ScenarioResult] = []
        for scenario in self._scenario_plan(cycle_index):
            scenarios.append(await scenario())
        skipped = [
            "backtests_run_report",
            "backtests_study_materialize",
            "notify_send",
            "notify_send",
            "notify_worker",
            "system_reset_space(action='apply')",
        ]
        return CoverageCycleResult(
            cycle_index=cycle_index,
            ok=all(item.ok for item in scenarios),
            scenarios=scenarios,
            skipped_tools=skipped,
            duration_ms=int((monotonic() - started) * 1000),
        )

    async def _discover_context(self) -> None:
        datasets = await self._call_and_parse("datasets", {"view": "list"}, scenario_name="_discover_datasets", expect_ok=True)
        for item in datasets[1].get("datasets", [])[:50]:
            descriptor = item.get("descriptor", {}) if isinstance(item, dict) else {}
            dataset_id = str(descriptor.get("dataset_id", "") or "").strip()
            symbol = str(descriptor.get("symbol", "") or "").strip()
            timeframe = str(descriptor.get("timeframe", "") or "").strip()
            if dataset_id and symbol == "BTCUSDT" and timeframe in {"1h", "5m"}:
                self.dataset_ids[timeframe] = dataset_id
        snapshots = await self._call_and_parse(
            "backtests_strategy",
            {"action": "inspect", "view": "list"},
            scenario_name="_discover_snapshots",
            expect_ok=True,
        )
        for item in snapshots[1].get("snapshots", []):
            snapshot_id = str(item.get("snapshot_id", "") or "").strip()
            latest_version = item.get("latest_version")
            if snapshot_id == "active-signal-v1" and latest_version:
                self.snapshot_id = snapshot_id
                self.snapshot_version = int(latest_version)
                break
        features_envelope, features_data = await self._call_and_parse(
            "features_custom",
            {"action": "inspect", "view": "list"},
            scenario_name="_discover_custom_features",
            expect_ok=True,
        )
        if features_envelope.ok:
            names = self._extract_custom_feature_names(features_data)
            if names:
                self.published_custom_feature = names[0]

    def _scenario_plan(self, cycle_index: int) -> list[Any]:
        scenarios = [
            lambda: self._tool_scenario("system_bootstrap", {"view": "summary"}, "system_bootstrap_summary"),
            lambda: self._tool_scenario("system_health", {"validate_tools": "true", "view": "summary"}, "system_health_summary"),
            lambda: self._tool_scenario("datasets", {"view": "catalog"}, "datasets_catalog"),
            lambda: self._tool_scenario("datasets_sync", {"action": "sync", "symbol": "BTCUSDT", "timeframes": ["1h", "5m"], "wait": "started"}, "datasets_sync_started"),
            lambda: self._tool_scenario("datasets_preview", {"dataset_id": self.dataset_ids["1h"], "view": "rows", "limit": "5"}, "datasets_preview_rows"),
            lambda: self._tool_scenario("events", {"view": "catalog", "family": "funding", "symbol": "BTCUSDT"}, "events_catalog_funding"),
            lambda: self._tool_scenario(
                "events",
                {"view": "align_preview", "family": "funding", "symbol": "BTCUSDT", "start_at": "2026-03-01T00:00:00+00:00", "end_at": "2026-03-05T00:00:00+00:00", "preview_rows": "5"},
                "events_align_preview",
            ),
            lambda: self._tool_scenario("events_sync", {"family": "funding", "scope": "incremental", "symbol": "BTCUSDT", "wait": "started"}, "events_sync_started"),
            lambda: self._tool_scenario("features_catalog", {"scope": "timeframe", "timeframe": "1h"}, "features_catalog_1h"),
            lambda: self._tool_scenario("features_dataset", {"action": "inspect", "view": "columns", "symbol": "BTCUSDT", "timeframe": "1h"}, "features_dataset_columns"),
            lambda: self._tool_scenario("features_analytics", {"action": "analytics", "symbol": "BTCUSDT", "feature_name": "rsi_1"}, "features_analytics_rsi"),
            lambda: self._tool_scenario("features_custom", {"action": "inspect", "view": "contract"}, "features_custom_contract"),
            lambda: self._features_cleanup_preview(),
            lambda: self._research_cycle(cycle_index),
            lambda: self._model_cycle(cycle_index),
            lambda: self._experiments_cycle(cycle_index),
            lambda: self._backtests_research_cycle(cycle_index),
            lambda: self._tool_scenario("experiments_run", {"action": "describe"}, "experiments_contract"),
            lambda: self._tool_scenario("experiments_registry_inspect", {"view": "summary"}, "experiments_registry_summary"),
            lambda: self._backtest_cycle(),
            lambda: self._tool_scenario("incidents", {"action": "inspect", "view": "summary", "include_closed": "true"}, "incidents_summary"),
            lambda: self._tool_scenario("gold_collection", {"action": "inspect", "view": "summary"}, "gold_collection_summary"),
            lambda: self._signal_binding_cycle(),
            lambda: self._tool_scenario("notify_status", {"view": "summary"}, "notify_status_summary"),
            lambda: self._tool_scenario("system_queue", {"action": "inspect", "view": "summary"}, "system_queue_summary"),
            lambda: self._tool_scenario("system_logs", {"service": "mcp_server", "view": "summary", "lines": "5"}, "system_logs_summary"),
            lambda: self._tool_scenario("system_reset_space", {"action": "preview"}, "system_reset_preview"),
        ]
        if cycle_index == 1:
            scenarios.append(lambda: self._tool_scenario("system_workspace", {"action": "inspect", "view": "summary"}, "system_workspace_summary"))
        return scenarios

    async def _research_cycle(self, cycle_index: int) -> ScenarioResult:
        project_name = f"coverage-project-{self.run_id}-{cycle_index}"
        project_envelope, project_data = await self._call_and_parse(
            "research_project",
            {"action": "create", "project": {"name": project_name, "goal": "Broker live coverage validation"}},
            scenario_name="research_project_create",
            expect_ok=True,
        )
        project_result = self._scenario_from_envelope("research_project_create", "research_project", project_envelope, project_data)
        project_block = project_data.get("project", {}) if isinstance(project_data.get("project"), dict) else {}
        projects = project_data.get("projects") or []
        if project_block:
            self.project_id = str(project_block.get("project_id", "") or self.project_id)
        elif projects and isinstance(projects[0], dict):
            self.project_id = str(projects[0].get("project_id", "") or self.project_id)
        if not self.project_id:
            self.project_id = str(project_data.get("project_id", "") or "")
        record_result = await self._await_async_scenario(
            "research_record",
            {
                "action": "create",
                "project_id": self.project_id,
                "kind": "note",
                "record": {"title": "Coverage note", "summary": "Live broker coverage note", "metadata": {"cycle": cycle_index}},
                "wait": "started",
            },
            "research_record_create",
        )
        search = await self._tool_scenario(
            "research_search",
            {"project_id": self.project_id, "query": "coverage note", "level": "normal", "limit": "5"},
            "research_search_normal",
        )
        map_result = await self._tool_scenario("research_map", {"action": "inspect", "project_id": self.project_id, "view": "summary"}, "research_map_summary")
        results = [project_result, record_result, search, map_result]
        return self._merge_results("research_cycle", results)

    async def _model_cycle(self, cycle_index: int) -> ScenarioResult:
        contract_envelope, contract_data = await self._call_and_parse(
            "models_dataset",
            {"action": "contract"},
            scenario_name="models_dataset_contract",
            expect_ok=True,
        )
        contract_result = self._scenario_from_envelope("models_dataset_contract", "models_dataset", contract_envelope, contract_data)
        spec = {
            "name": f"coverage_ds_{self.run_id}_{cycle_index}",
            "description": "Small dataset for live broker coverage.",
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "start_at": "2024-01-01T00:00:00+00:00",
            "end_at": "2024-01-10T23:00:00+00:00",
            "input_fields": [
                {"name": "open_now", "source_kind": "ohlcv", "column": "open", "timeframe": "1h", "transform": "current"},
                {"name": "close_now", "source_kind": "ohlcv", "column": "close", "timeframe": "1h", "transform": "current"},
                {"name": "rsi_now", "source_kind": "feature", "column": "rsi_1", "timeframe": "1h", "transform": "current"},
            ],
            "target": {
                "kind": "python_target",
                "code": "def compute_target(step):\n    return 1 if step.current('close') > step.current('open') else 0\n",
                "required_columns": ["open", "close"],
                "required_features": ["rsi_1"],
                "required_timeframes": ["1h"],
                "lookback_bars": 1,
            },
            "split": {"train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2},
        }
        materialize = await self._await_async_scenario(
            "models_dataset",
            {"action": "materialize", "spec": spec, "wait": "started"},
            "models_dataset_materialize",
        )
        data = materialize.metadata.get("data", {}) if materialize.ok else {}
        dataset = data.get("dataset", {}) if isinstance(data, dict) else {}
        self.dataset_materialized_id = str(dataset.get("dataset_id", "") or "")
        if not self.dataset_materialized_id:
            return self._merge_results("model_cycle", [contract_result, materialize])
        model_id = f"coverage_model_{self.run_id}_{cycle_index}"
        train = await self._await_async_scenario(
            "models_train",
            {
                "action": "start",
                "model_id": model_id,
                "name": f"coverage_model_{cycle_index}",
                "dataset_id": self.dataset_materialized_id,
                "library": "lightgbm",
                "task_type": "binary_classification",
                "primary_metric": "accuracy",
                "params": {"num_leaves": 8, "learning_rate": 0.1, "n_estimators": 20},
                "wait": "started",
            },
            "models_train_start",
        )
        train_data = train.metadata.get("data", {}) if train.ok else {}
        version = str((train_data.get("result", {}) or {}).get("version", "") or train_data.get("version", "") or "")
        self.model_id = model_id
        self.model_version = version
        train_second = await self._await_async_scenario(
            "models_train",
            {
                "action": "start",
                "model_id": model_id,
                "name": f"coverage_model_{cycle_index}_v2",
                "dataset_id": self.dataset_materialized_id,
                "library": "lightgbm",
                "task_type": "binary_classification",
                "primary_metric": "accuracy",
                "params": {"num_leaves": 12, "learning_rate": 0.08, "n_estimators": 24},
                "wait": "started",
            },
            "models_train_start_v2",
        )
        train_second_data = train_second.metadata.get("data", {}) if train_second.ok else {}
        second_version = str((train_second_data.get("result", {}) or {}).get("version", "") or train_second_data.get("version", "") or "")
        if second_version:
            self.model_version = second_version
        registry = await self._tool_scenario("models_registry", {"action": "inspect", "view": "list"}, "models_registry_list")
        version_detail = ScenarioResult(name="models_registry_version_detail", ok=True, summary="skipped_no_model_version")
        compare = ScenarioResult(name="models_compare_versions", ok=True, summary="skipped_no_second_version")
        to_feature_validate = ScenarioResult(name="models_to_feature_validate", ok=True, summary="skipped_no_model_version")
        if self.model_version:
            version_detail = await self._tool_scenario(
                "models_registry",
                {"action": "inspect", "view": "version_detail", "model_id": self.model_id, "version": self.model_version},
                "models_registry_version_detail",
            )
            to_feature_validate = await self._await_async_scenario(
                "models_to_feature",
                {
                    "action": "validate",
                    "model_id": self.model_id,
                    "version": self.model_version,
                    "feature_name": f"cf_cov_model_{cycle_index}",
                    "symbol": "BTCUSDT",
                    "timeframe": "1h",
                    "wait": "started",
                },
                "models_to_feature_validate",
                max_polls=5,
            )
        if second_version:
            compare = await self._tool_scenario("models_compare", {"model_id": self.model_id}, "models_compare_versions")
        to_feature = await self._tool_scenario(
            "models_to_feature",
            {"action": "inspect", "view": "contract", "model_id": self.model_id, "version": self.model_version or "1"},
            "models_to_feature_contract",
        )
        return self._merge_results(
            "model_cycle",
            [contract_result, materialize, train, train_second, registry, version_detail, compare, to_feature, to_feature_validate],
        )

    async def _experiments_cycle(self, cycle_index: int) -> ScenarioResult:
        code = (
            "def run(ctx):\n"
            "    payload = {'cycle': " + str(cycle_index) + ", 'status': 'ok'}\n"
            "    ctx.artifacts.write_json('summary.json', payload)\n"
            "    return payload\n"
        )
        started = await self._await_async_scenario(
            "experiments_run",
            {
                "action": "start",
                "task_summary": f"Live broker coverage experiment cycle {cycle_index}",
                "task_category": "validation",
                "name": f"coverage_experiment_{self.run_id}_{cycle_index}",
                "code": code,
                "wait": "started",
            },
            "experiments_run_start",
            max_polls=10,
        )
        data = started.metadata.get("data", {}) if started.ok else {}
        job = data.get("job", {}) if isinstance(data, dict) else {}
        job_id = str(job.get("job_id", "") or data.get("job_id", "") or "")
        if not job_id:
            return self._merge_results("experiments_cycle", [started])
        inspect_status = await self._tool_scenario("experiments_inspect", {"view": "status", "job_id": job_id}, "experiments_inspect_status")
        inspect_result = await self._tool_scenario("experiments_inspect", {"view": "result", "job_id": job_id}, "experiments_inspect_result")
        artifacts = await self._tool_scenario("experiments_read", {"view": "list", "job_id": job_id}, "experiments_read_list")
        artifact_json = ScenarioResult(name="experiments_read_json", ok=True, summary="skipped_no_json_artifact")
        artifact_path = self._select_experiment_json_artifact(artifacts)
        if artifact_path:
            artifact_json = await self._tool_scenario(
                "experiments_read",
                {"view": "json", "job_id": job_id, "artifact_path": artifact_path},
                "experiments_read_json",
            )
        return self._merge_results("experiments_cycle", [started, inspect_status, inspect_result, artifacts, artifact_json])

    @staticmethod
    def _select_experiment_json_artifact(artifacts: ScenarioResult) -> str:
        if not artifacts.ok:
            return ""
        metadata = artifacts.metadata.get("data", {}) if isinstance(artifacts.metadata, dict) else {}
        items = metadata.get("artifacts", []) if isinstance(metadata, dict) else []
        preferred = ""
        fallback = ""
        for item in items:
            if not isinstance(item, dict):
                continue
            relative_path = str(item.get("relative_path", "") or "")
            mime_type = str(item.get("mime_type", "") or "")
            if not relative_path:
                continue
            if relative_path.endswith(".json") and relative_path == "result.json":
                preferred = relative_path
                break
            if relative_path.endswith(".json") and mime_type == "application/json" and not fallback:
                fallback = relative_path
        return preferred or fallback

    async def _backtests_research_cycle(self, cycle_index: int) -> ScenarioResult:
        validate_signal = await self._tool_scenario(
            "backtests_strategy_validate",
            {"mode": "signal", "code": "result = rsi_1 <= 30", "feature_names": ["rsi_1"]},
            "backtests_strategy_validate_signal",
        )
        variants = [{"variant_id": f"baseline_copy_{cycle_index}", "name": "Baseline copy", "changes": []}]
        studies_preview = await self._tool_scenario(
            "backtests_studies",
            {
                "action": "preview",
                "base_snapshot_id": self.snapshot_id,
                "base_version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "signal_id": "feature-long",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-01-07T00:00:00+00:00",
                "variants": variants,
            },
            "backtests_studies_preview",
        )
        studies_start = await self._await_async_scenario(
            "backtests_studies",
            {
                "action": "start",
                "base_snapshot_id": self.snapshot_id,
                "base_version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "signal_id": "feature-long",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-01-07T00:00:00+00:00",
                "variants": variants,
                "wait": "started",
            },
            "backtests_studies_start",
            max_polls=1,
        )
        studies_status = await self._follow_up_job_status(
            tool_name="backtests_studies",
            result=studies_start,
            scenario_name="backtests_studies_status",
        )
        walkforward_start = await self._await_async_scenario(
            "backtests_walkforward",
            {
                "action": "start",
                "base_snapshot_id": self.snapshot_id,
                "base_version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-02-15T00:00:00+00:00",
                "train_days": 14,
                "validation_days": 7,
                "test_days": 7,
                "step_days": 7,
                "wait": "started",
            },
            "backtests_walkforward_start",
            max_polls=1,
        )
        walkforward_status = await self._follow_up_job_status(
            tool_name="backtests_walkforward",
            result=walkforward_start,
            scenario_name="backtests_walkforward_status",
        )
        return self._merge_results(
            "backtests_research_cycle",
            [validate_signal, studies_preview, studies_start, studies_status, walkforward_start, walkforward_status],
        )

    async def _backtest_cycle(self) -> ScenarioResult:
        detail = await self._tool_scenario(
            "backtests_strategy",
            {"action": "inspect", "view": "detail", "snapshot_id": self.snapshot_id, "version": str(self.snapshot_version)},
            "backtests_strategy_detail",
        )
        plan_envelope, plan_data = await self._call_and_parse(
            "backtests_plan",
            {
                "snapshot_id": self.snapshot_id,
                "version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-01-07T00:00:00+00:00",
            },
            scenario_name="backtests_plan_window",
            expect_ok=True,
        )
        plan_result = self._scenario_from_envelope("backtests_plan_window", "backtests_plan", plan_envelope, plan_data)
        run = await self._await_async_scenario(
            "backtests_runs",
            {
                "action": "start",
                "snapshot_id": self.snapshot_id,
                "version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-01-07T00:00:00+00:00",
                "wait": "started",
                "starting_capital_usd": "1000",
            },
            "backtests_runs_start",
        )
        data = run.metadata.get("data", {}) if run.ok else {}
        run_block = data.get("run", {}) if isinstance(data, dict) else {}
        self.backtest_run_id = str(run_block.get("run_id", "") or data.get("run_id", "") or "")
        analysis = ScenarioResult(name="backtests_analysis_diagnostics", ok=True, summary="skipped_no_run_id")
        conditions = ScenarioResult(name="backtests_conditions_run", ok=True, summary="skipped_no_snapshot")
        if self.backtest_run_id and self._backtest_run_is_analysis_ready(run):
            analysis = await self._await_async_scenario(
                "backtests_analysis",
                {"action": "start", "analysis": "diagnostics", "run_id": self.backtest_run_id, "wait": "started"},
                "backtests_analysis_diagnostics",
            )
        elif self.backtest_run_id:
            analysis = ScenarioResult(
                name="backtests_analysis_diagnostics",
                ok=True,
                tool="backtests_analysis",
                summary="skipped_run_not_persisted_yet",
            )
        conditions = await self._await_async_scenario(
            "backtests_conditions",
            {
                "action": "run",
                "snapshot_id": self.snapshot_id,
                "version": str(self.snapshot_version),
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
                "signal_id": "feature-long",
                "start_at": "2026-01-01T00:00:00+00:00",
                "end_at": "2026-01-07T00:00:00+00:00",
                "bins": "3",
                "wait": "started",
            },
            "backtests_conditions_run",
        )
        return self._merge_results("backtest_cycle", [detail, plan_result, run, analysis, conditions])

    async def _signal_binding_cycle(self) -> ScenarioResult:
        detail = await self._tool_scenario("signal_api_binding_inspect", {"view": "detail"}, "signal_binding_detail")
        summary = await self._tool_scenario("signal_api_binding_inspect", {"view": "summary"}, "signal_binding_summary")
        return self._merge_results("signal_binding_cycle", [detail, summary])

    async def _features_cleanup_preview(self) -> ScenarioResult:
        if not self.published_custom_feature:
            return ScenarioResult(
                name="features_cleanup_preview",
                ok=True,
                tool="features_cleanup",
                summary="skipped_no_published_custom_feature",
            )
        return await self._tool_scenario(
            "features_cleanup",
            {"action": "preview", "scope": "features_only", "cf_names": [self.published_custom_feature], "dry_run": "true"},
            "features_cleanup_preview",
        )

    async def _tool_scenario(self, tool_name: str, arguments: dict[str, Any], scenario_name: str) -> ScenarioResult:
        envelope, data = await self._call_and_parse(tool_name, arguments, scenario_name=scenario_name, expect_ok=True)
        return self._scenario_from_envelope(scenario_name, tool_name, envelope, data)

    @staticmethod
    def _scenario_from_envelope(scenario_name: str, tool_name: str, envelope: ToolResultEnvelope, data: dict[str, Any]) -> ScenarioResult:
        return ScenarioResult(
            name=scenario_name,
            ok=envelope.ok,
            tool=tool_name,
            summary=envelope.summary,
            response_status=envelope.response_status,
            duration_ms=envelope.duration_ms,
            warning_count=len(envelope.warnings),
            warnings=list(envelope.warnings),
            metadata={"data": data, "key_facts": dict(envelope.key_facts)},
            error="" if envelope.ok else envelope.error_class or envelope.summary,
        )

    async def _await_async_scenario(self, tool_name: str, arguments: dict[str, Any], scenario_name: str, *, max_polls: int = 40) -> ScenarioResult:
        envelope, data = await self._call_and_parse(tool_name, arguments, scenario_name=scenario_name, expect_ok=True)
        current = ScenarioResult(
            name=scenario_name,
            ok=envelope.ok,
            tool=tool_name,
            summary=envelope.summary,
            response_status=envelope.response_status,
            duration_ms=envelope.duration_ms,
            warning_count=len(envelope.warnings),
            warnings=list(envelope.warnings),
            metadata={"data": data, "key_facts": dict(envelope.key_facts)},
            error="" if envelope.ok else envelope.error_class or envelope.summary,
        )
        resume_tool = envelope.resume_tool or tool_name
        resume_arguments = dict(envelope.resume_arguments)
        if current.response_status.lower() not in _RUNNING_STATUSES or not resume_arguments:
            return current
        for _ in range(max_polls):
            await asyncio.sleep(1.0)
            envelope, data = await self._call_and_parse(resume_tool, resume_arguments, scenario_name=f"{scenario_name}_resume", expect_ok=True)
            current = ScenarioResult(
                name=scenario_name,
                ok=envelope.ok,
                tool=resume_tool,
                summary=envelope.summary,
                response_status=envelope.response_status,
                duration_ms=current.duration_ms + envelope.duration_ms,
                warning_count=len(set(current.warnings + envelope.warnings)),
                warnings=list(dict.fromkeys(current.warnings + envelope.warnings)),
                metadata={"data": data, "key_facts": dict(envelope.key_facts)},
                error="" if envelope.ok else envelope.error_class or envelope.summary,
            )
            resume_arguments = dict(envelope.resume_arguments or resume_arguments)
            if current.response_status.lower() not in _RUNNING_STATUSES:
                break
        return current

    async def _follow_up_job_status(self, *, tool_name: str, result: ScenarioResult, scenario_name: str) -> ScenarioResult:
        data = result.metadata.get("data", {}) if isinstance(result.metadata, dict) else {}
        if not isinstance(data, dict):
            return ScenarioResult(name=scenario_name, ok=True, tool=tool_name, summary="skipped_no_job_id")
        job = data.get("job", {})
        if not isinstance(job, dict):
            return ScenarioResult(name=scenario_name, ok=True, tool=tool_name, summary="skipped_no_job_id")
        job_id = str(job.get("job_id", "") or "")
        if not job_id:
            return ScenarioResult(name=scenario_name, ok=True, tool=tool_name, summary="skipped_no_job_id")
        return await self._tool_scenario(tool_name, {"action": "status", "job_id": job_id}, scenario_name)

    async def _call_and_parse(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        scenario_name: str,
        expect_ok: bool,
    ) -> tuple[ToolResultEnvelope, dict[str, Any]]:
        try:
            envelope = await self.broker.call_tool(
                tool_name=tool_name,
                arguments=arguments,
                plan_id="live_coverage",
                slice_id=scenario_name,
            )
        except Exception as exc:
            return (
                ToolResultEnvelope(
                    call_id=f"error_{scenario_name}",
                    tool=tool_name,
                    ok=False,
                    retryable=False,
                    duration_ms=0,
                    summary=str(exc),
                    error_class=exc.__class__.__name__,
                ),
                {},
            )
        data = self._load_data(envelope)
        if expect_ok and not envelope.ok:
            return envelope, data
        return envelope, data

    @staticmethod
    def _load_data(envelope: ToolResultEnvelope) -> dict[str, Any]:
        if not envelope.raw_result_ref:
            return {}
        try:
            payload = json.loads(Path(envelope.raw_result_ref).read_text(encoding="utf-8"))
        except Exception:
            return {}
        data = payload.get("structuredContent", {}).get("data", {})
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _backtest_run_is_analysis_ready(result: ScenarioResult) -> bool:
        data = result.metadata.get("data", {}) if isinstance(result.metadata, dict) else {}
        if not isinstance(data, dict):
            return False
        run = data.get("run", {})
        if not isinstance(run, dict):
            return False
        status = str(run.get("status", "") or "").strip().lower()
        if status not in {"completed", "failed", "cancelled", "stopped"}:
            return False
        if run.get("summary_is_final") is False:
            return False
        return True

    @staticmethod
    def _extract_custom_feature_names(data: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for key in ("features", "items", "custom_features"):
            values = data.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                for field in ("name", "feature_name"):
                    name = str(item.get(field, "") or "").strip()
                    if name.startswith("cf_"):
                        names.append(name)
                        break
        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if name not in seen:
                deduped.append(name)
                seen.add(name)
        return deduped

    @staticmethod
    def _merge_results(name: str, parts: list[ScenarioResult]) -> ScenarioResult:
        return ScenarioResult(
            name=name,
            ok=all(item.ok for item in parts),
            tool="+".join(item.tool for item in parts if item.tool),
            summary=" | ".join(item.summary for item in parts if item.summary)[:800],
            response_status=";".join(item.response_status for item in parts if item.response_status),
            duration_ms=sum(item.duration_ms for item in parts),
            warning_count=sum(item.warning_count for item in parts),
            warnings=list(dict.fromkeys(warning for item in parts for warning in item.warnings)),
            metadata={"parts": [asdict(item) for item in parts]},
            error=" | ".join(item.error for item in parts if item.error)[:800],
        )


def build_live_coverage_summary(*, run_id: str, cycles: list[CoverageCycleResult]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "generated_at": int(time()),
        "cycle_count": len(cycles),
        "all_ok": all(item.ok for item in cycles),
        "cycles": [asdict(item) for item in cycles],
    }


__all__ = ["CoverageCycleResult", "LiveCoverageRunner", "ScenarioResult", "build_live_coverage_summary"]
