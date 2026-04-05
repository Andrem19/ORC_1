"""
Research context for MCP dev_space1 integration.

Provides:
- Static MCP tool catalog for prompt injection
- Research rules governing trading research methodology
- Research context formatter (reads from state/research_context.json)
- MCP worker instructions for task prompts
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.research_context")

# ---------------------------------------------------------------------------
# MCP Tool Catalog — static description of all dev_space1 tools
# ---------------------------------------------------------------------------

MCP_TOOL_CATALOG = """## MCP dev_space1 Tool Catalog

Pipeline: Data -> Features -> Models -> Strategy -> Backtest -> Analysis -> Live

### Data Layer
- datasets(view='catalog'|'instruments'|'summary') — list available market datasets
- datasets_sync(action='add'|'sync', symbol=, timeframes=) — register/sync instruments
- datasets_preview(view='rows'|'chart') — preview dataset content

### Feature Layer
- features_catalog(scope='available') — list all 48 feature columns (22 builtin + 26 custom cf_*)
- features_dataset(action='inspect'|'build'|'refresh') — build managed feature datasets
- features_custom(action='inspect'|'validate'|'publish') — author and publish custom cf_* features
- features_analytics(action='heatmap'|'portability'|'render') — analyze feature profitability

### Model Layer
- models_dataset(action='contract'|'preview'|'materialize') — prepare training datasets
- models_train(action='start') — train CatBoost/LightGBm models
- models_registry(action='inspect'|'compare') — manage model cards and versions
- models_to_feature(action='scaffold'|'validate'|'publish') — project model into cf_* feature
- models_compare(model_id=) — compare model versions by primary metric

### Strategy Layer
- backtests_strategy(action='inspect'|'create'|'save_version'|'clone'|'validate') — manage strategy snapshots
- backtests_strategy_validate(mode='signal'|'exit') — validate signal/exit logic before saving

### Backtest Layer
- backtests_plan(snapshot_id=, symbol=) — preflight readiness check (read-only)
- backtests_runs(action='start'|'inspect') — execute single backtest runs
- backtests_studies(action='preview'|'start') — multi-variant batch research
- backtests_walkforward(action='start') — rolling out-of-sample walk-forward evaluation
- backtests_conditions(action='run') — condition stability analysis across time bins

### Analysis Layer
- backtests_analysis(action='start', analysis='diagnostics'|'layer_compare'|'signal_ownership'|'robustness_bundle') — post-run analysis

### Research Memory
- research_project(action='list'|'create'|'open') — manage research projects
- research_map(action='inspect'|'define'|'record_attempt'|'advance_hypothesis') — research atlas
- research_record(action='create'|'update') — record hypotheses, results, decisions
- research_search(query=) — search research memory

### Events & Experiments
- events(view='catalog'|'align_preview') — funding/expiry event stores
- events_sync(family=, scope=) — sync event data
- experiments_run(action='describe'|'start') — run ad-hoc Python experiments

### Live Deployment
- signal_api_binding_apply(snapshot_id=, version=, symbol=) — deploy strategy to live signals

### Key Constraints
- Timeframes POLICY-LOCKED: anchor_timeframe='1h', execution_timeframe='5m' — no other values accepted
- Classifier columns (cl_*) auto-excluded from bulk builds unless force=True
- features_dataset(action='build') for new custom features; action='refresh' skips existing cl_*
- Predictive model-backed features cannot extend past 2024-12-31 (OOS leakage guard)
"""

# ---------------------------------------------------------------------------
# Research Rules — methodology for trading research
# ---------------------------------------------------------------------------

RESEARCH_RULES = """## Research Methodology Rules

1. Seek new orthogonal features, not cosmetic rewrites of existing rules
2. Test standalone first, then in best research shell, then with best base strategy
3. Check cannibalization: what the new layer adds, removes, and whether net integration improves
4. Judge improvement by SIMULTANEOUS progress in PnL, trade count, AND drawdown quality — all three
5. Reject ideas that fail full-history, window-stability, or integration-with-base checks
6. When one layer is exhausted, change hypothesis class: new features, gating, routing, missing-regime search
7. Do not assume a strong standalone signal integrates well — always verify integration
8. Use ownership analysis to separate real profit-carrying families from noise
9. Do not break load-bearing families without strong evidence
10. Exit signals are as important as entry signals — develop both
11. Record every experiment while fresh, including negative and equal-outcome results
12. Low overlap is useful but not sufficient; only integrated net improvement matters
13. Find weak quarters first, then search for features that specifically repair those windows
14. Repeated additive tweaks on the same short layer usually signal a research dead end
"""

# ---------------------------------------------------------------------------
# MCP Worker Instructions — injected into worker prompts for MCP tasks
# ---------------------------------------------------------------------------

MCP_WORKER_INSTRUCTIONS = """## MCP dev_space1 Tool Calling Guide

You have access to MCP dev_space1 tools. Call them directly as structured tool invocations.

### Rules
1. Always specify symbol='BTCUSDT', anchor_timeframe='1h', execution_timeframe='5m' unless told otherwise
2. For backtests: call backtests_plan first to validate readiness, then backtests_runs(action='start')
3. For models: use models_dataset -> models_train -> models_registry workflow
4. For features: use features_custom(action='validate') then features_custom(action='publish')
5. For async operations: poll with action='status' or action='inspect' and job_id
6. Report key metrics: Sharpe, return%, trade count, max drawdown, win rate, profit factor
7. GOLD COLLECTION: When your task produces a breakthrough (new best snapshot, profitable new feature, milestone result), call gold_collection(action='add', entity_type='snapshot'|'feature', entity_id='...', reason='...') to preserve it

### Result Format
Return your results in the standard JSON schema:
{
  "status": "success|error|partial",
  "summary": "What you found/did with key metrics",
  "artifacts": ["snapshot_ids", "run_ids", "model_ids", "feature_names"],
  "next_hint": "Suggested follow-up action",
  "confidence": 0.0-1.0,
  "error": "error message if any"
}
"""

# MCP keywords for auto-detecting MCP-related tasks
MCP_TASK_KEYWORDS = frozenset({
    "snapshot", "backtest", "feature", "model", "strategy",
    "mcp", "dataset", "signal", "catboost", "lightgbm",
    "walk-forward", "walkforward", "analysis", "heatmap",
    "classifier", "train", "predict", "btcusdt", "regime",
})


def is_mcp_task(task_description: str) -> bool:
    """Check if a task description relates to MCP dev_space1 work."""
    desc_lower = task_description.lower()
    return any(kw in desc_lower for kw in MCP_TASK_KEYWORDS)


def load_research_context(state_dir: str) -> dict[str, Any] | None:
    """Load cached MCP state from state/research_context.json."""
    path = Path(state_dir) / "research_context.json"
    if not path.exists():
        logger.warning("No research context file at %s", path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load research context: %s", e)
        return None


def format_research_context_for_planner(context: dict[str, Any]) -> str:
    """Format MCP state into concise prompt text for the planner."""
    sections: list[str] = []

    sections.append("## MCP Research State")
    updated = context.get("updated_at", "unknown")
    sections.append(f"(last updated: {updated})")
    sections.append("")

    # Baseline
    baseline = context.get("baseline", {})
    if baseline:
        sections.append("### Baseline (comparison anchor)")
        sections.append(
            f"{baseline.get('snapshot_id', '?')}@{baseline.get('version', '?')}: "
            f"Sharpe {baseline.get('sharpe', '?')}, "
            f"return {baseline.get('return_pct', '?')}%, "
            f"{baseline.get('trades', '?')} trades, "
            f"max DD {baseline.get('max_dd_pct', '?')}%, "
            f"win rate {baseline.get('win_rate', '?')}%"
        )
        sections.append("")

    # Strategy snapshots
    snapshots = context.get("snapshots", [])
    if snapshots:
        sections.append(f"### Strategy Snapshots ({len(snapshots)} total)")
        for s in snapshots:
            status_tag = "PROFITABLE" if s.get("profitable") else "negative"
            if s.get("worst"):
                status_tag = "WORST"
            sections.append(
                f"- {s['id']}@{s.get('version', '?')}: "
                f"return {s.get('return_pct', '?')}%, "
                f"{s.get('trades', '?')} trades, "
                f"Sharpe {s.get('sharpe', '?')}, "
                f"max DD {s.get('max_dd_pct', '?')}% "
                f"({status_tag})"
            )
        sections.append("")

    # Models
    models = context.get("models", [])
    if models:
        sections.append(f"### Models ({len(models)} registered)")
        for m in models:
            sections.append(
                f"- {m['id']}: {m.get('task_type', '?')}, "
                f"{'deployed' if m.get('deployable') else 'offline-only'}"
            )
        sections.append("")

    # Features
    features = context.get("features", {})
    if features:
        builtin = features.get("builtin_count", 0)
        custom = features.get("custom_count", 0)
        sections.append(f"### Available Features ({builtin + custom} total)")
        sections.append(f"{builtin} builtin: {', '.join(features.get('builtin_sample', []))}")
        sections.append(f"{custom} custom (cf_*): {', '.join(features.get('custom_sample', []))}")
        sections.append("")

    # Datasets
    datasets = context.get("datasets", {})
    if datasets:
        sections.append("### Datasets")
        for sym, tfs in datasets.items():
            sections.append(f"- {sym}: {', '.join(tfs)}")
        sections.append("")

    # Research projects
    projects = context.get("projects", [])
    if projects:
        sections.append(f"### Research Projects ({len(projects)} total)")
        for p in projects:
            sections.append(f"- {p}")
        sections.append("")

    # What hasn't been done
    gaps = context.get("gaps", [])
    if gaps:
        sections.append("### What Has NOT Been Done Yet")
        for g in gaps:
            sections.append(f"- {g}")
        sections.append("")

    return "\n".join(sections)
