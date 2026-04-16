from __future__ import annotations

from typing import Any

from app.services.mcp_catalog.normalizer import build_catalog_snapshot


def make_catalog_snapshot(tool_defs: list[dict[str, Any]] | None = None):
    return build_catalog_snapshot(
        tools=tool_defs or _default_tool_defs(),
        endpoint_url="http://127.0.0.1:8766/mcp",
        bootstrap_manifest={
            "tools": [
                {"name": "research_map", "hint": "research_map.inspect is the default read path for atlas state."},
                {"name": "experiments_read", "hint": "experiments_read reads one concrete job artifact, not a generic project listing."},
            ]
        },
    )


def _default_tool_defs() -> list[dict[str, Any]]:
    return [
        {
            "name": "research_project",
            "description": "List, inspect, create, open, delete, and maintain research projects.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["list", "create", "open", "delete"]},
                    "project_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "research_map",
            "description": "Inspect or advance the research atlas. project_id is REQUIRED for all actions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "define", "advance_hypothesis"]},
                    "project_id": {"type": "string", "description": "REQUIRED for ALL research_map actions."},
                },
                "required": ["project_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "research_record",
            "description": "Create, update, or inspect one semantic research record.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["create", "update", "status"]},
                    "kind": {"type": "string", "enum": ["hypothesis", "note", "milestone"]},
                    "operation_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "research_search",
            "description": "Search one research project. query is REQUIRED.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "query": {"type": "string", "description": "REQUIRED free-text query."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "datasets",
            "description": "Inspect the dataset catalogue and metadata.",
            "inputSchema": {
                "type": "object",
                "properties": {"view": {"type": "string", "enum": ["catalog", "detail", "summary"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "datasets_preview",
            "description": "Preview one dataset through rows or chart samples.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string"},
                    "view": {"type": "string", "enum": ["rows", "chart"]},
                },
                "required": ["dataset_id", "view"],
                "additionalProperties": False,
            },
        },
        {
            "name": "datasets_sync",
            "description": "Add or sync one tracked instrument and optionally wait for completion.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "sync"]},
                    "symbol": {"type": "string"},
                    "wait": {"type": "string", "enum": ["none", "started", "completed"]},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "features_catalog",
            "description": "Inspect the managed feature catalog. timeframe is policy-locked when scope=timeframe.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": ["available", "timeframe"]},
                    "timeframe": {"type": "string", "description": "policy-locked timeframe token."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "features_dataset",
            "description": "Inspect, build, refresh, or remove managed feature datasets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "build", "refresh", "remove"]},
                    "view": {"type": "string", "enum": ["list", "summary", "columns"]},
                    "dataset_id": {"type": "string"},
                    "symbol": {"type": "string"},
                    "timeframe": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "features_custom",
            "description": "Inspect, validate, publish, or delete one custom feature.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "validate", "publish", "delete"]},
                    "view": {"type": "string", "enum": ["contract", "list", "detail", "source"]},
                    "name": {"type": "string"},
                    "project_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "features_analytics",
            "description": "Inspect stored feature analytics and build profitability heatmaps.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["analytics", "heatmap", "render", "portability"]},
                    "feature_name": {"type": "string"},
                    "symbol": {"type": "string"},
                    "anchor_timeframe": {"type": "string"},
                    "bucket_count": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "events",
            "description": "Inspect local normalized event stores.",
            "inputSchema": {
                "type": "object",
                "properties": {"view": {"type": "string", "enum": ["catalog", "align_preview"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "events_sync",
            "description": "Sync normalized local event stores.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "family": {"type": "string", "enum": ["funding", "expiry", "all"]},
                    "scope": {"type": "string", "enum": ["incremental", "backfill"]},
                    "wait": {"type": "string", "enum": ["none", "started", "completed"]},
                },
                "required": ["family", "scope"],
                "additionalProperties": False,
            },
        },
        {
            "name": "models_dataset",
            "description": "Prepare and inspect training datasets.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "preview", "materialize", "status"]},
                    "dataset_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "models_train",
            "description": "Run the unified model training lifecycle.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "status", "cancel"]},
                    "dataset_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "models_registry",
            "description": "Inspect and manage model cards and artifacts.",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["inspect", "update_card", "promote_version"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "models_compare",
            "description": "Compare immutable versions of one model.",
            "inputSchema": {
                "type": "object",
                "properties": {"model_id": {"type": "string"}},
                "required": ["model_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "models_to_feature",
            "description": "Publish one immutable model version into a model-backed custom feature workflow.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "scaffold", "validate", "publish"]},
                    "model_id": {"type": "string"},
                },
                "required": ["action", "model_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_plan",
            "description": "Plan one backtest run with readiness checks.",
            "inputSchema": {
                "type": "object",
                "properties": {"snapshot_id": {"type": "string"}, "symbol": {"type": "string"}},
                "required": ["snapshot_id", "symbol"],
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_runs",
            "description": "Start, inspect, or manage backtest runs.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "inspect", "status", "cancel"]},
                    "run_id": {"type": "string"},
                    "snapshot_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_studies",
            "description": "Preview or start multi-variant backtest studies.",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["preview", "start", "status", "result"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_walkforward",
            "description": "Start or inspect walk-forward evaluations.",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["start", "status", "result", "cancel"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_conditions",
            "description": "Run condition stability analysis. anchor_timeframe is currently policy-locked to '1h'.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["run", "list", "status", "result", "cancel"]},
                    "snapshot_id": {"type": "string"},
                    "symbol": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_analysis",
            "description": "Start or inspect post-run backtests analysis.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "status", "result", "cancel"]},
                    "analysis": {"type": "string", "enum": ["diagnostics", "layer_compare"]},
                    "run_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_strategy",
            "description": "Inspect, create, or version strategy snapshots.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["inspect", "create", "save_version"]},
                    "snapshot_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "backtests_strategy_validate",
            "description": "Validate signal or exit logic before saving into a snapshot.",
            "inputSchema": {
                "type": "object",
                "properties": {"mode": {"type": "string", "enum": ["signal", "exit"]}},
                "required": ["mode"],
                "additionalProperties": False,
            },
        },
        {
            "name": "experiments_run",
            "description": "Describe or start one custom experiment job.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["describe", "start"]},
                    "task_summary": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "experiments_inspect",
            "description": "Inspect experiment jobs, status, results, logs, and artifacts.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "enum": ["list", "summary", "detail", "status", "result", "logs", "artifacts"]},
                    "job_id": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "experiments_read",
            "description": "List experiment artifacts or read one artifact. job_id is REQUIRED when reading text/json content.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "view": {"type": "string", "enum": ["list", "text", "json"]},
                    "job_id": {"type": "string"},
                    "artifact_path": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "experiments_registry_inspect",
            "description": "Inspect the experiments registry and candidate tasks.",
            "inputSchema": {
                "type": "object",
                "properties": {"view": {"type": "string", "enum": ["summary", "tasks", "candidates", "detail"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "notify_send",
            "description": "Send one operator notification.",
            "inputSchema": {
                "type": "object",
                "properties": {"kind": {"type": "string", "enum": ["auto", "text", "photo", "document"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "system_bootstrap",
            "description": "Load mandatory startup rules and agent workflow guidance.",
            "inputSchema": {
                "type": "object",
                "properties": {"view": {"type": "string", "enum": ["summary", "detail", "raw"]}},
                "additionalProperties": False,
            },
        },
        {
            "name": "incidents",
            "description": "Manage infrastructure incident lifecycle.",
            "inputSchema": {
                "type": "object",
                "properties": {"action": {"type": "string", "enum": ["capture", "inspect", "update"]}},
                "additionalProperties": False,
            },
        },
    ]
