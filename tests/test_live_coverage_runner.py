from app.live_coverage.runner import CoverageCycleResult, LiveCoverageRunner, ScenarioResult, build_live_coverage_summary


def test_build_live_coverage_summary_aggregates_cycles() -> None:
    summary = build_live_coverage_summary(
        run_id="coverage_1",
        cycles=[
            CoverageCycleResult(cycle_index=1, ok=True, scenarios=[ScenarioResult(name="a", ok=True)], duration_ms=10),
            CoverageCycleResult(cycle_index=2, ok=False, scenarios=[ScenarioResult(name="b", ok=False, error="failed")], duration_ms=20),
        ],
    )

    assert summary["run_id"] == "coverage_1"
    assert summary["cycle_count"] == 2
    assert summary["all_ok"] is False
    assert summary["cycles"][1]["scenarios"][0]["error"] == "failed"


def test_extract_custom_feature_names_dedupes_and_filters() -> None:
    names = LiveCoverageRunner._extract_custom_feature_names(
        {
            "features": [
                {"name": "cf_alpha"},
                {"name": "not_custom"},
                {"feature_name": "cf_beta"},
                {"name": "cf_alpha"},
            ]
        }
    )

    assert names == ["cf_alpha", "cf_beta"]


def test_features_cleanup_preview_skips_without_published_feature() -> None:
    runner = LiveCoverageRunner.__new__(LiveCoverageRunner)
    runner.published_custom_feature = ""

    import asyncio

    result = asyncio.run(LiveCoverageRunner._features_cleanup_preview(runner))

    assert result.ok is True
    assert result.summary == "skipped_no_published_custom_feature"


def test_backtest_run_is_analysis_ready_requires_final_completed_run() -> None:
    ready = ScenarioResult(
        name="backtests_runs_start",
        ok=True,
        metadata={"data": {"run": {"status": "completed", "summary_is_final": True}}},
    )
    not_ready = ScenarioResult(
        name="backtests_runs_start",
        ok=True,
        metadata={"data": {"run": {"status": "publishing_result", "summary_is_final": False}}},
    )

    assert LiveCoverageRunner._backtest_run_is_analysis_ready(ready) is True
    assert LiveCoverageRunner._backtest_run_is_analysis_ready(not_ready) is False


def test_select_experiment_json_artifact_prefers_result_json_then_fallback() -> None:
    result = ScenarioResult(
        name="experiments_read_list",
        ok=True,
        metadata={
            "data": {
                "artifacts": [
                    {"relative_path": "manifest.json", "mime_type": "application/json"},
                    {"relative_path": "stdout.log", "mime_type": None},
                    {"relative_path": "result.json", "mime_type": "application/json"},
                ]
            }
        },
    )
    fallback = ScenarioResult(
        name="experiments_read_list",
        ok=True,
        metadata={
            "data": {
                "artifacts": [
                    {"relative_path": "manifest.json", "mime_type": "application/json"},
                    {"relative_path": "script.py", "mime_type": "text/x-python"},
                ]
            }
        },
    )

    assert LiveCoverageRunner._select_experiment_json_artifact(result) == "result.json"
    assert LiveCoverageRunner._select_experiment_json_artifact(fallback) == "manifest.json"


def test_scenario_plan_calls_system_workspace_only_in_first_cycle() -> None:
    runner = LiveCoverageRunner.__new__(LiveCoverageRunner)
    calls: list[tuple[str, str, dict]] = []

    async def fake_tool_scenario(tool: str, args: dict, name: str):
        calls.append((tool, name, args))
        return ScenarioResult(name=name, ok=True, tool=tool)

    async def fake_features_cleanup_preview():
        return ScenarioResult(name="features_cleanup_preview", ok=True, tool="features_cleanup")

    async def fake_research_cycle(cycle_index: int):
        return ScenarioResult(name=f"research_cycle_{cycle_index}", ok=True)

    async def fake_model_cycle(cycle_index: int):
        return ScenarioResult(name=f"model_cycle_{cycle_index}", ok=True)

    async def fake_experiments_cycle(cycle_index: int):
        return ScenarioResult(name=f"experiments_cycle_{cycle_index}", ok=True)

    async def fake_backtests_research_cycle(cycle_index: int):
        return ScenarioResult(name=f"backtests_research_cycle_{cycle_index}", ok=True)

    async def fake_backtest_cycle():
        return ScenarioResult(name="backtest_cycle", ok=True)

    async def fake_signal_binding_cycle():
        return ScenarioResult(name="signal_binding_cycle", ok=True)

    runner._tool_scenario = fake_tool_scenario
    runner._features_cleanup_preview = fake_features_cleanup_preview
    runner._research_cycle = fake_research_cycle
    runner._model_cycle = fake_model_cycle
    runner._experiments_cycle = fake_experiments_cycle
    runner._backtests_research_cycle = fake_backtests_research_cycle
    runner._backtest_cycle = fake_backtest_cycle
    runner._signal_binding_cycle = fake_signal_binding_cycle
    runner.dataset_ids = {"1h": "BTCUSDT_1h", "5m": "BTCUSDT_5m"}

    import asyncio

    for scenario in LiveCoverageRunner._scenario_plan(runner, 1):
        asyncio.run(scenario())
    first_calls = list(calls)
    calls.clear()
    for scenario in LiveCoverageRunner._scenario_plan(runner, 2):
        asyncio.run(scenario())
    second_calls = list(calls)

    assert any(name == "system_workspace_summary" for _, name, _ in first_calls)
    assert not any(name == "system_workspace_summary" for _, name, _ in second_calls)
    assert not any(tool == "signal_api_binding_apply" for tool, _, _ in first_calls + second_calls)
