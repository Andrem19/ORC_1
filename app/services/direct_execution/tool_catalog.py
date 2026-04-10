"""
Static dev_space1 tool catalog used by direct execution.
"""

from __future__ import annotations

DEFAULT_DEV_SPACE1_TOOLS: tuple[str, ...] = (
    "backtests_analysis",
    "backtests_conditions",
    "backtests_plan",
    "backtests_run_report",
    "backtests_runs",
    "backtests_strategy",
    "backtests_strategy_validate",
    "backtests_studies",
    "backtests_study_materialize",
    "backtests_walkforward",
    "datasets",
    "datasets_preview",
    "datasets_sync",
    "events",
    "events_sync",
    "experiments_inspect",
    "experiments_read",
    "experiments_registry_inspect",
    "experiments_run",
    "features_analytics",
    "features_catalog",
    "features_cleanup",
    "features_custom",
    "features_dataset",
    "gold_collection",
    "incidents",
    "models_compare",
    "models_dataset",
    "models_registry",
    "models_to_feature",
    "models_train",
    "notify_send",
    "notify_status",
    "notify_worker",
    "research_map",
    "research_project",
    "research_record",
    "research_search",
    "signal_api_binding_apply",
    "signal_api_binding_inspect",
    "system_bootstrap",
    "system_health",
    "system_logs",
    "system_queue",
    "system_reset_space",
    "system_workspace",
)

EXPENSIVE_DIRECT_TOOLS: frozenset[str] = frozenset(
    {
        "backtests_runs",
        "backtests_studies",
        "backtests_walkforward",
        "backtests_analysis",
        "backtests_conditions",
        "features_custom",
        "features_dataset",
        "models_dataset",
        "models_train",
        "models_to_feature",
        "experiments_run",
        "datasets_sync",
        "events_sync",
    }
)


def direct_available_tools() -> set[str]:
    return set(DEFAULT_DEV_SPACE1_TOOLS)


__all__ = ["DEFAULT_DEV_SPACE1_TOOLS", "EXPENSIVE_DIRECT_TOOLS", "direct_available_tools"]
