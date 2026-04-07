"""
Configuration for the orchestrator.

All parameters are defined here or passed as function arguments.
No argparse usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models import WorkerConfig


@dataclass
class NotificationConfig:
    """Telegram notification settings."""
    enabled: bool = False
    min_interval_seconds: int = 30  # rate limit between messages
    translate_to_russian: bool = False
    translation_model_dir: str = "models/opus-mt-en-ru"


@dataclass
class AdapterConfig:
    name: str
    cli_path: str = ""
    extra_flags: list[str] = field(default_factory=list)
    timeout_seconds: int = 120
    model: str = ""
    mode: str = "default"
    use_bare: bool = False
    no_session_persistence: bool = False
    soft_timeout_seconds: int = 300
    hard_timeout_seconds: int = 900
    no_first_byte_seconds: int = 180
    capture_stderr_live: bool = False
    # LM Studio / HTTP API settings
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.7
    max_tokens: int = 4096


@dataclass
class McpReviewConfig:
    """Configuration for MCP problem review and tracking."""
    enabled: bool = True
    review_every_n_cycles: int = 10
    review_after_n_errors: int = 3
    fixes_dir: str = "fixes"
    planner_timeout: int = 180
    max_problems_in_context: int = 10


@dataclass
class OrchestratorConfig:
    # --- Goal ---
    goal: str = "No goal specified"

    # --- System prompts ---
    planner_system_prompt: str = (
        "You are the planner. Analyze the current state and decide the next action. "
        "Return your decision as a JSON object following the required schema."
    )
    worker_system_prompt: str = (
        "You are a worker agent. Complete the assigned task and return a structured JSON result."
    )

    # --- Workers ---
    workers: list[WorkerConfig] = field(default_factory=lambda: [
        WorkerConfig(worker_id="qwen-1", role="executor", system_prompt=""),
    ])

    # --- Timing ---
    poll_interval_seconds: int = 300  # 5 minutes default
    planner_timeout_seconds: int = 180
    worker_timeout_seconds: int = 300
    worker_restart_delay_seconds: int = 10

    # --- Limits ---
    max_errors_total: int = 20
    max_empty_cycles: int = 12
    max_task_attempts: int = 3
    max_worker_timeout_count: int = 3

    # --- Adapters ---
    planner_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(
        name="claude_planner_cli",
        cli_path="claude",
        model="opus",
        mode="batch_text",
        use_bare=True,
        no_session_persistence=True,
        soft_timeout_seconds=300,
        hard_timeout_seconds=900,
        no_first_byte_seconds=180,
        capture_stderr_live=True,
    ))
    worker_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(
        name="qwen_worker_cli",
        cli_path="qwen-code",
    ))

    # --- State ---
    state_dir: str = "state"
    state_file: str = "orchestrator_state.json"
    startup_mode: str = "resume"  # "resume" | "reset" | "reset_all"

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_file: str = "orchestrator.log"

    # --- Console display ---
    rich_console: bool = True
    console_truncate_length: int = 300

    # --- Feature flags ---
    detect_duplicate_results: bool = True
    require_structured_output: bool = True

    # --- Research (MCP integration) ---
    research_config: dict[str, Any] = field(default_factory=dict)

    # --- Notifications ---
    notifications: NotificationConfig = field(default_factory=NotificationConfig)

    # --- MCP problem review ---
    mcp_review: McpReviewConfig = field(default_factory=McpReviewConfig)

    # --- Plan mode ---
    plan_mode: bool = False
    plan_dir: str = "plans"
    max_concurrent_plan_tasks: int = 2
    plan_task_timeout_seconds: int = 600
    max_mcp_failures: int = 5
    mcp_health_check_interval_cycles: int = 5
    silent_worker_warn_seconds: int = 300
    max_plan_attempts: int = 3

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / self.state_file

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def load_config_from_dict(data: dict[str, Any]) -> OrchestratorConfig:
    """Create an OrchestratorConfig from a dictionary, with defaults for missing keys."""
    cfg = OrchestratorConfig()

    simple_fields = {
        "goal", "planner_system_prompt", "worker_system_prompt",
        "poll_interval_seconds", "planner_timeout_seconds",
        "worker_timeout_seconds", "worker_restart_delay_seconds",
        "max_errors_total", "max_empty_cycles", "max_task_attempts",
        "max_worker_timeout_count", "state_dir", "state_file",
        "log_level", "log_dir", "log_file",
        "rich_console", "console_truncate_length",
        "detect_duplicate_results", "require_structured_output",
        "research_config",
        "plan_mode", "plan_dir", "max_concurrent_plan_tasks",
        "plan_task_timeout_seconds", "max_mcp_failures",
        "silent_worker_warn_seconds", "max_plan_attempts",
        "startup_mode",
    }
    for key in simple_fields:
        if key in data:
            setattr(cfg, key, data[key])

    if "workers" in data:
        cfg.workers = [WorkerConfig(**w) for w in data["workers"]]

    if "planner_adapter" in data:
        cfg.planner_adapter = AdapterConfig(**data["planner_adapter"])

    if "worker_adapter" in data:
        cfg.worker_adapter = AdapterConfig(**data["worker_adapter"])

    if "notifications" in data:
        cfg.notifications = NotificationConfig(**data["notifications"])

    if "mcp_review" in data:
        cfg.mcp_review = McpReviewConfig(**data["mcp_review"])

    if cfg.startup_mode not in ("resume", "reset", "reset_all"):
        raise ValueError(
            f"Invalid startup_mode '{cfg.startup_mode}'. "
            f"Must be one of: 'resume', 'reset', 'reset_all'"
        )

    return cfg
