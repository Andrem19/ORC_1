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
    translation_backend: str = "opus"  # "opus" (Helsinki-NLP) or "lmstudio"
    translation_model_dir: str = "models/opus-mt-en-ru"
    translation_model_name: str = "Helsinki-NLP/opus-mt-en-ru"
    batch_enabled: bool = True  # batch worker-result notifications
    batch_debounce_seconds: float = 5.0  # wait before sending batch


@dataclass
class AdapterConfig:
    name: str
    cli_path: str = ""
    extra_flags: list[str] = field(default_factory=list)
    timeout_seconds: int = 1800
    model: str = ""
    mode: str = "default"
    use_bare: bool = False
    no_session_persistence: bool = False
    soft_timeout_seconds: int = 3600
    hard_timeout_seconds: int = 7200
    no_first_byte_seconds: int = 900
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
class ReportCompressorConfig:
    """LM Studio config for LLM-based report compression.
    base_url, model, reasoning_effort are injected from LMStudioConfig at load time.
    """
    enabled: bool = False
    base_url: str = "http://localhost:1234"
    model: str = ""  # injected from lmstudio shared
    reasoning_effort: str = ""  # injected from lmstudio shared
    max_tokens: int = 300
    timeout_seconds: int = 30


@dataclass
class LMStudioTranslationConfig:
    """LM Studio settings for EN→RU translation (per-feature overrides)."""
    max_tokens: int = 2048
    timeout_seconds: int = 60


@dataclass
class LMStudioAssistantConfig:
    """LM Studio settings for log analysis & execution prediction."""
    analysis_interval_cycles: int = 50
    max_log_lines: int = 200
    max_tokens: int = 4096  # thinking models: reasoning ~500 + JSON content ~1000
    timeout_seconds: int = 120


@dataclass
class LMStudioConfig:
    """LM Studio shared connection + feature-specific settings."""
    enabled: bool = False
    base_url: str = "http://localhost:1234"
    model: str = ""  # empty = use currently loaded model
    reasoning_effort: str = "none"  # "none" = no thinking, "low"/"medium"/"high" = thinking
    report_compressor: ReportCompressorConfig = field(default_factory=ReportCompressorConfig)
    assistant: LMStudioAssistantConfig = field(default_factory=LMStudioAssistantConfig)
    translation: LMStudioTranslationConfig = field(default_factory=LMStudioTranslationConfig)

    # Legacy aliases — keep max_log_lines / analysis_interval_cycles / max_tokens / timeout_seconds
    # accessible directly for backward compatibility with code that reads config.lmstudio.*
    @property
    def analysis_interval_cycles(self) -> int:
        return self.assistant.analysis_interval_cycles

    @property
    def max_log_lines(self) -> int:
        return self.assistant.max_log_lines

    @property
    def max_tokens(self) -> int:
        return self.assistant.max_tokens

    @property
    def timeout_seconds(self) -> int:
        return self.assistant.timeout_seconds


@dataclass
class OrchestratorConfig:
    # --- Goal ---
    goal: str = "No goal specified"

    # --- System prompts ---
    planner_system_prompt: str = (
        "You are the planner. Analyze the current state and decide the next action. "
        "Return your decision as a JSON object following the required schema."
    )
    operator_directives: str = ""
    worker_system_prompt: str = (
        "You are a worker agent. Complete the assigned task and return a structured JSON result."
    )

    # --- Workers ---
    workers: list[WorkerConfig] = field(default_factory=lambda: [
        WorkerConfig(worker_id="qwen-1", role="executor", system_prompt=""),
    ])

    # --- Timing ---
    poll_interval_seconds: int = 300  # 5 minutes default
    planner_timeout_seconds: int = 1800
    worker_timeout_seconds: int = 1800
    worker_restart_delay_seconds: int = 10

    # --- Limits ---
    max_errors_total: int = 20
    max_empty_cycles: int = 12
    max_task_attempts: int = 3
    max_worker_timeout_count: int = 3
    drain_timeout_seconds: int = 600  # timeout for graceful drain mode

    # --- Adapters ---
    planner_adapter: AdapterConfig = field(default_factory=lambda: AdapterConfig(
        name="claude_planner_cli",
        cli_path="claude",
        model="opus",
        mode="batch_text",
        use_bare=True,
        no_session_persistence=True,
        soft_timeout_seconds=3600,
        hard_timeout_seconds=7200,
        no_first_byte_seconds=900,
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

    # --- LM Studio (shared connection + report compressor + assistant + translation) ---
    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)

    # --- Plan mode ---
    plan_mode: bool = False
    plan_dir: str = "plans"
    max_concurrent_plan_tasks: int = 2
    plan_task_timeout_seconds: int = 3600
    max_mcp_failures: int = 5
    mcp_health_check_interval_cycles: int = 5
    silent_worker_warn_seconds: int = 900
    max_plan_attempts: int = 3
    plan_stages_guidance: str = "Create 5 to 7 research stages depending on complexity. If investigation can be parallelized across 2-3 workers, count those parallel branches as a single time-stage."

    @property
    def state_path(self) -> Path:
        return Path(self.state_dir) / self.state_file

    @property
    def report_compressor(self) -> ReportCompressorConfig:
        """Backward-compatible alias: config.report_compressor → config.lmstudio.report_compressor."""
        return self.lmstudio.report_compressor

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


def load_config_from_dict(data: dict[str, Any]) -> OrchestratorConfig:
    """Create an OrchestratorConfig from a dictionary, with defaults for missing keys."""
    cfg = OrchestratorConfig()

    simple_fields = {
        "goal", "planner_system_prompt", "operator_directives", "worker_system_prompt",
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
        "plan_stages_guidance",
        "startup_mode", "drain_timeout_seconds",
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

    # LM Studio: parse nested structure, inject shared base_url/model into features
    if "lmstudio" in data:
        lm_data = dict(data["lmstudio"])  # shallow copy
        # Extract sub-sections before passing to LMStudioConfig constructor
        compressor_data = lm_data.pop("report_compressor", {})
        assistant_data = lm_data.pop("assistant", {})
        translation_data = lm_data.pop("translation", {})

        # Legacy: flat fields that now belong to assistant sub-config
        for flat_key in ("analysis_interval_cycles", "max_log_lines",
                         "max_tokens", "timeout_seconds"):
            if flat_key in lm_data and flat_key not in assistant_data:
                assistant_data[flat_key] = lm_data.pop(flat_key)

        # Also handle legacy flat fields for report_compressor
        if "max_tokens" in lm_data and "max_tokens" not in compressor_data:
            compressor_data.setdefault("max_tokens", lm_data.get("max_tokens"))

        cfg.lmstudio = LMStudioConfig(
            **{k: v for k, v in lm_data.items()
               if k in ("enabled", "base_url", "model", "reasoning_effort")},
            report_compressor=ReportCompressorConfig(
                **compressor_data,
                base_url=lm_data.get("base_url", "http://localhost:1234"),
                model=lm_data.get("model", ""),
                reasoning_effort=lm_data.get("reasoning_effort", ""),
            ),
            assistant=LMStudioAssistantConfig(**assistant_data),
            translation=LMStudioTranslationConfig(**translation_data),
        )

    # Legacy: standalone [report_compressor] section (pre-reorganization)
    if "report_compressor" in data:
        rc = data["report_compressor"]
        cfg.lmstudio.report_compressor = ReportCompressorConfig(
            enabled=rc.get("enabled", cfg.lmstudio.report_compressor.enabled),
            base_url=rc.get("base_url", cfg.lmstudio.base_url),
            model=rc.get("model", cfg.lmstudio.model),
            max_tokens=rc.get("max_tokens", cfg.lmstudio.report_compressor.max_tokens),
            timeout_seconds=rc.get("timeout_seconds", cfg.lmstudio.report_compressor.timeout_seconds),
        )
        # If legacy report_compressor.enabled was True, enable lmstudio globally
        if rc.get("enabled"):
            cfg.lmstudio.enabled = True

    if cfg.startup_mode not in ("resume", "reset", "reset_all"):
        raise ValueError(
            f"Invalid startup_mode '{cfg.startup_mode}'. "
            f"Must be one of: 'resume', 'reset', 'reset_all'"
        )

    return cfg
