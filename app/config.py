"""Configuration for the direct orchestrator runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from app.models import WorkerConfig


DEFAULT_PLANNER_PROMPT_TEMPLATE = """You are writing a markdown research plan, not executing research.
Use only the facts in this prompt. Do not use tools, do not inspect the workspace, do not gather more context, and do not spawn subagents.
Do not name MCP tools, facade names, actions, views, or exact API calls in the plan. Describe the work in plain operational language only.

## Status and Frame
Baseline is fixed: $baseline_snapshot_id@$baseline_version. Do not retune the baseline or recycle old tuned families.
symbol: $symbol, anchor_timeframe: $anchor_timeframe, execution_timeframe: $execution_timeframe
Wave context: $wave_context

## Goal
$goal

## Baseline
$baseline_snapshot_id@$baseline_version: Sharpe 1.06, +13% return, 498 trades, max DD 2.4%
This baseline is the fixed reference system for integration checks, not an object for retuning.

## Research Principles
- Search for genuinely new signals or missing-regime layers, not cosmetic baseline variations.
- Require standalone value, stability, integration quality, and cannibalization control.
- Prefer ideas that add new trades, new regimes, or new information layers.
- Do not spend the cycle retuning hours, days, thresholds, or execution details of the baseline.

## dev_space1 Capabilities
Workers available: $worker_count. Each markdown plan is executed by one worker, but the broader orchestration may run up to 3 plans in parallel.
The worker has access to MCP and will choose the exact tools and call sequence. Your job is to define intent, order, constraints, success criteria, and what evidence must be collected.
Constraints: timeframes stay locked to 1h/5m. Do not tell the worker to rediscover already-known basics like symbol or baseline ids unless that check is itself the purpose of the ETAP.

$research_context_section$previous_wave_summary_section$previous_findings_section$anti_patterns_section$research_history_section$previous_plan_excerpt_section## Output Format
Write a SINGLE research plan as plain markdown for Plan v$plan_version.
Rules:
1. Start with '# Plan v$plan_version'
2. Include sections for Status and Frame, Goal, Baseline, Research Principles, and dev_space1 Capabilities before the ETAPs
3. Write exactly 3 ETAP sections with '## ETAP N: Name' headers
4. Each ETAP must have: one-sentence goal, 4-6 numbered action steps, completion criteria
5. Action steps must be plain-language research instructions, not tool calls or API syntax
6. NEVER use placeholders like <...> or '...' — use concrete values
7. Assume symbol/timeframes are already known; do not spend steps rediscovering them
8. End each ETAP with a compact markdown results table template
9. Do NOT use code fences, pseudocode, or multi-line snippets anywhere
10. Keep the entire plan short: target 1400-2400 characters total
11. Do NOT embed raw Python, formulas, or API snippets in any step; describe custom feature logic in plain English
12. Prefer cheap validation steps first; avoid multi-hour builds or heavy studies in the initial plan
"""


@dataclass
class NotificationConfig:
    """Telegram notification settings."""
    enabled: bool = False
    min_interval_seconds: int = 30  # rate limit between messages
    translate_to_russian: bool = False
    translation_provider: str = "lmstudio"  # "lmstudio" | "qwen_cli" | "claude_cli" | "off"
    translation_fallback_1: str = ""  # "qwen_cli" | "claude_cli" | "lmstudio" | ""
    translation_fallback_2: str = ""  # "qwen_cli" | "claude_cli" | "lmstudio" | ""
    batch_enabled: bool = True  # batch worker-result notifications
    batch_debounce_seconds: float = 5.0  # wait before sending batch


@dataclass
class AdapterConfig:
    name: str
    cli_path: str = ""
    extra_flags: list[str] = field(default_factory=list)
    exclude_tools: list[str] = field(default_factory=list)
    model: str = ""
    mode: str = "default"
    use_bare: bool = False
    no_session_persistence: bool = False
    capture_stderr_live: bool = False
    # LM Studio / HTTP API settings
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096
    effort: str = ""  # e.g. "low", "medium", "high" — passed as --effort to CLI
    allow_tool_use: bool = False


@dataclass
class LMStudioConfig:
    """LM Studio shared connection settings."""
    base_url: str = "http://localhost:1234"
    model: str = ""  # empty = use currently loaded model
    reasoning_effort: str = "none"  # "none" = no thinking, "low"/"medium"/"high" = thinking


@dataclass
class MinimaxConfig:
    """MiniMax OpenAI-compatible API settings."""
    base_url: str = "https://api.minimax.io"
    model: str = "MiniMax-M2.7"
    api_key_env_var: str = "MINIMAX_API"  # env var name holding the API key
    temperature: float = 1.0  # MiniMax recommended default
    max_tokens: int = 8192


@dataclass
class SequenceReportConfig:
    """Adapter and LLM settings for sequence completion narrative generation."""
    provider: str = "claude_cli"  # claude_cli | lmstudio
    model: str = "opus"
    timeout_seconds: int = 120
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5


@dataclass
class DirectExecutionConfig:
    """Direct model execution settings."""
    provider: str = "lmstudio"  # lmstudio | minimax | qwen_cli | claude_cli
    fallback_1: str = ""  # "qwen_cli" | "claude_cli" | "lmstudio" | "off" | ""
    fallback_2: str = ""  # "qwen_cli" | "claude_cli" | "lmstudio" | "off" | ""
    timeout_seconds: int = 1200
    max_attempts_per_slice: int = 2
    max_tool_calls_per_slice: int = 24
    max_expensive_tool_calls_per_slice: int = 6
    mcp_endpoint_url: str = "http://127.0.0.1:8766/mcp"
    mcp_auth_mode: str = "none"  # none | bearer
    mcp_token_env_var: str = "DEV_SPACE1_MCP_BEARER_TOKEN"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 60.0
    first_action_timeout_seconds: int = 75
    stalled_action_timeout_seconds: int = 60
    retry_budget: int = 1
    incident_on_unclear_contract: bool = True
    contract_guardrails_enabled: bool = True
    parse_repair_attempts: int = 1
    qwen_tool_registry_preflight: bool = True
    qwen_preflight_timeout_seconds: int = 60
    qwen_primary_preflight_enabled: bool = True
    qwen_primary_preflight_max_attempts: int = 3
    qwen_primary_preflight_retry_delay_seconds: float = 2.0
    fallback_skip_repair_on_infra_signal: bool = True
    repair_tool_call_budget: int = 3
    primary_retry_budget: int = 1
    safe_exclude_tools: list[str] = field(default_factory=lambda: [
        "run_shell_command",
        "read_file",
        "write_file",
        "edit",
        "grep_search",
        "glob",
        "list_directory",
        "web_fetch",
        "system_reset_space",
    ])
    # LMStudio tool-call reliability settings
    lmstudio_zero_tool_retries: int = 2  # nudge retries before fallback
    lmstudio_first_turn_tool_choice: str = "required"  # "required" | "auto"
    max_blocked_retries: int = 2  # how many times to retry a blocked-checkpoint slice before abort cascade


@dataclass
class OrchestratorConfig:
    # --- Goal ---
    goal: str = "No goal specified"

    operator_directives: str = ""
    worker_system_prompt: str = (
        "You are a direct execution model with access to approved dev_space1 tools. "
        "Complete the assigned slice and return one terminal structured JSON result only."
    )

    # --- Workers ---
    workers: list[WorkerConfig] = field(default_factory=lambda: [
        WorkerConfig(worker_id="qwen-1", role="executor", system_prompt=""),
    ])

    # --- Timing / Limits ---
    poll_interval_seconds: int = 300  # 5 minutes default
    max_errors_total: int = 20
    max_empty_cycles: int = 12

    # --- Adapters ---
    planner_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(
        name="claude_planner_cli",
        cli_path="claude",
        model="opus",
        mode="batch_text",
        use_bare=True,
        no_session_persistence=True,
        capture_stderr_live=True,
    ))
    worker_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(
        name="qwen_worker_cli",
        cli_path="qwen",
    ))
    # --- State ---
    state_dir: str = "state"
    state_file: str = "orchestrator_state.json"
    execution_state_file: str = "execution_state_v2.json"
    startup_mode: str = "resume"  # "resume" | "reset" | "reset_all"
    current_run_id: str = ""

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "orchestrator.log"

    # --- Plan source / raw-plan conversion ---
    plan_source: str = "planner"  # planner | compiled_raw
    start_from: str = ""  # e.g. "v2" — skip compiled plans before plan_v2
    raw_plan_dir: str = "raw_plans"
    compiled_plan_dir: str = "compiled_plans"
    compiled_queue_skip_failures: bool = True
    # When True, transient infrastructure failures (LM Studio crashes, HTTP
    # timeouts, gate rejections) do NOT cause the rest of a compiled sequence
    # to be skipped — next batches still run and flow through the fallback
    # chain. Only explicit semantic aborts skip remaining batches.
    infra_failure_never_skip_batches: bool = True
    converter_use_llm: bool = True

    # --- Console display ---
    rich_console: bool = True
    console_log_level: str = "INFO"
    console_truncate_length: int = 300

    # --- Result handling ---
    detect_duplicate_results: bool = True

    # --- Research (MCP integration) ---
    research_config: dict[str, Any] = field(default_factory=dict)
    planner_decision_timeout_seconds: int = 180
    planner_decision_retry_attempts: int = 3
    worker_decision_timeout_seconds: int = 120
    worker_decision_retry_attempts: int = 2
    worker_timeout_policy_by_tag: dict[str, int] = field(default_factory=lambda: {
        "feature_contract": 180,
        "postmortem": 180,
    })
    decision_retry_backoff_seconds: float = 0.25
    decision_cycle_sleep_seconds: float = 1.0

    # --- Notifications ---
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # --- LM Studio (shared connection + translation) ---
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    minimax: MinimaxConfig = field(default_factory=MinimaxConfig)
    direct_execution: DirectExecutionConfig = field(default_factory=DirectExecutionConfig)
    sequence_report: SequenceReportConfig = field(default_factory=SequenceReportConfig)

    plan_dir: str = "plans"
    max_concurrent_plan_tasks: int = 3
    max_plans_per_run: int = 1
    max_consecutive_failed_plans: int = 1
    planner_prompt_template: str = DEFAULT_PLANNER_PROMPT_TEMPLATE

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / self.state_file

    @property
    def execution_state_path(self) -> Path:
        return Path(self.state_dir) / self.execution_state_file

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def load_config_from_dict(data: dict[str, Any]) -> OrchestratorConfig:
    """Create an OrchestratorConfig from a dictionary, with defaults for missing keys."""
    cfg = OrchestratorConfig()

    simple_fields = {
        "goal", "operator_directives", "worker_system_prompt",
        "poll_interval_seconds", "max_errors_total", "max_empty_cycles",
        "state_dir", "state_file", "execution_state_file",
        "log_level", "log_dir", "log_file",
        "plan_source", "start_from", "raw_plan_dir", "compiled_plan_dir",
        "compiled_queue_skip_failures", "infra_failure_never_skip_batches",
        "converter_use_llm",
        "rich_console", "console_log_level", "console_truncate_length",
        "detect_duplicate_results",
        "research_config",
        "planner_decision_timeout_seconds", "planner_decision_retry_attempts",
        "worker_decision_timeout_seconds", "worker_decision_retry_attempts",
        "worker_timeout_policy_by_tag",
        "decision_retry_backoff_seconds",
        "decision_cycle_sleep_seconds",
        "plan_dir",
        "max_concurrent_plan_tasks",
        "max_plans_per_run", "max_consecutive_failed_plans",
        "planner_prompt_template", "startup_mode", "current_run_id",
    }
    for key in simple_fields:
        if key in data:
            setattr(cfg, key, data[key])

    if "workers" in data:
        cfg.workers = [WorkerConfig(**w) for w in data["workers"]]

    def _filter_dataclass_kwargs(cls: type[Any], payload: dict[str, Any]) -> dict[str, Any]:
        allowed = {item.name for item in fields(cls)}
        return {key: value for key, value in dict(payload).items() if key in allowed}

    if "planner_adapter" in data:
        cfg.planner_adapter = AdapterConfig(**_filter_dataclass_kwargs(AdapterConfig, data["planner_adapter"]))

    if "worker_adapter" in data:
        cfg.worker_adapter = AdapterConfig(**_filter_dataclass_kwargs(AdapterConfig, data["worker_adapter"]))

    if "notifications" in data:
        cfg.notifications = NotificationConfig(**_filter_dataclass_kwargs(NotificationConfig, data["notifications"]))

    if "direct_execution" in data:
        cfg.direct_execution = DirectExecutionConfig(**_filter_dataclass_kwargs(DirectExecutionConfig, data["direct_execution"]))

    # LM Studio: parse shared connection config
    if "lmstudio" in data:
        lm_data = dict(data["lmstudio"])

        cfg.lmstudio = LMStudioConfig(
            **{k: v for k, v in lm_data.items()
               if k in ("base_url", "model", "reasoning_effort")},
        )

    # MiniMax: parse API connection config
    if "minimax" in data:
        mm_data = dict(data["minimax"])
        cfg.minimax = MinimaxConfig(
            **{k: v for k, v in mm_data.items()
               if k in ("base_url", "model", "api_key_env_var", "temperature", "max_tokens")},
        )

    if "sequence_report" in data:
        cfg.sequence_report = SequenceReportConfig(**_filter_dataclass_kwargs(SequenceReportConfig, data["sequence_report"]))

    if cfg.startup_mode not in ("resume", "reset", "reset_all"):
        raise ValueError(
            f"Invalid startup_mode '{cfg.startup_mode}'. "
            f"Must be one of: 'resume', 'reset', 'reset_all'"
        )
    if cfg.plan_source not in {"planner", "compiled_raw"}:
        raise ValueError(
            f"Invalid plan_source '{cfg.plan_source}'. "
            "Must be one of: 'planner', 'compiled_raw'"
        )

    return cfg
