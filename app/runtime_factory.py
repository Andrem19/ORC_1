"""
Factory helpers for planner/worker adapters used by runtime and converter.
"""

from __future__ import annotations

from app.adapters.base import BaseAdapter
from app.config import OrchestratorConfig


def create_planner_adapter(config: OrchestratorConfig) -> BaseAdapter:
    from app.adapters.claude_planner_cli import ClaudePlannerCli

    return ClaudePlannerCli(
        cli_path=config.planner_adapter.cli_path,
        model=config.planner_adapter.model,
        extra_flags=config.planner_adapter.extra_flags,
        mode=config.planner_adapter.mode,
        use_bare=config.planner_adapter.use_bare,
        no_session_persistence=config.planner_adapter.no_session_persistence,
        capture_stderr_live=config.planner_adapter.capture_stderr_live,
        effort=config.planner_adapter.effort,
    )


def create_sequence_report_adapter(config: OrchestratorConfig) -> BaseAdapter:
    """Create an adapter for LLM sequence-report narrative generation."""
    sr = config.sequence_report
    provider = sr.provider.strip().lower()

    if provider == "claude_cli":
        from app.adapters.claude_planner_cli import ClaudePlannerCli
        return ClaudePlannerCli(
            cli_path="claude",
            model=sr.model,
            mode="batch_text",
            use_bare=True,
            no_session_persistence=True,
            capture_stderr_live=True,
        )

    if provider == "lmstudio":
        from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
        lm_cfg = config.lmstudio
        return LmStudioWorkerApi(
            base_url=lm_cfg.base_url or "http://localhost:1234",
            model=lm_cfg.model,
            api_key="lm-studio",
            temperature=0.4,
            max_tokens=2048,
            reasoning_effort="none",
        )

    raise ValueError(
        f"Unknown sequence_report provider: {sr.provider!r}. "
        "Supported: claude_cli, lmstudio"
    )


def create_worker_adapter(config: OrchestratorConfig) -> BaseAdapter:
    provider = config.direct_execution.provider.strip().lower()
    safe_exclude = list(config.direct_execution.safe_exclude_tools)

    if provider == "qwen_cli":
        from app.adapters.qwen_worker_cli import QwenWorkerCli
        return QwenWorkerCli(
            cli_path="qwen",
            allow_tool_use=True,
            exclude_tools=safe_exclude,
        )

    if provider == "claude_cli":
        from app.adapters.claude_worker_cli import ClaudeWorkerCli
        return ClaudeWorkerCli(
            allow_tool_use=True,
            exclude_tools=safe_exclude,
            mcp_servers=_build_mcp_servers_dict(config),
        )

    if provider == "minimax":
        return _create_minimax_adapter(config=config)

    # Default: lmstudio — use [worker_adapter] config section
    adapter = config.worker_adapter
    return _create_worker_adapter(config=config, adapter_config=adapter)


def _create_worker_adapter(*, config: OrchestratorConfig, adapter_config) -> BaseAdapter:
    from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
    from app.adapters.qwen_worker_cli import QwenWorkerCli

    wa = adapter_config
    if wa.name == "lmstudio_worker_api":
        lm_cfg = config.lmstudio
        return LmStudioWorkerApi(
            base_url=wa.base_url or lm_cfg.base_url or "http://localhost:1234",
            model=wa.model or lm_cfg.model,
            api_key=wa.api_key or "lm-studio",
            temperature=wa.temperature,
            max_tokens=wa.max_tokens,
            reasoning_effort=lm_cfg.reasoning_effort,
        )
    return QwenWorkerCli(
        cli_path=wa.cli_path,
        extra_flags=wa.extra_flags,
        exclude_tools=wa.exclude_tools,
        allow_tool_use=wa.allow_tool_use,
    )


def _create_minimax_adapter(*, config: OrchestratorConfig) -> BaseAdapter:
    """Create an LmStudioWorkerApi configured for MiniMax's OpenAI-compatible API.

    MiniMax supports the ``tools`` parameter in chat completions, so the
    entire LmStudioToolLoop + DirectMcpClient stack is reused unchanged.
    """
    import os

    from app.adapters.lmstudio_worker_api import LmStudioWorkerApi

    mm = config.minimax
    api_key = os.environ.get(mm.api_key_env_var, "")
    return LmStudioWorkerApi(
        base_url=mm.base_url,
        model=mm.model,
        api_key=api_key,
        temperature=mm.temperature,
        max_tokens=mm.max_tokens,
        reasoning_effort="none",  # MiniMax doesn't use reasoning_effort
    )


def create_fallback_adapter(provider_name: str, config: OrchestratorConfig) -> BaseAdapter | None:
    """Create a fallback worker adapter by provider name.

    Returns None when *provider_name* is empty/whitespace/disabled.
    Raises ValueError for unknown provider names.
    """
    if not provider_name or not provider_name.strip():
        return None

    name = provider_name.strip().lower()
    safe_exclude = list(config.direct_execution.safe_exclude_tools)

    # Explicitly disabled fallback
    if name in {"off", "none", "disabled", "false", "no"}:
        return None

    if name == "qwen_cli":
        from app.adapters.qwen_worker_cli import QwenWorkerCli

        return QwenWorkerCli(
            cli_path="qwen",
            allow_tool_use=True,
            exclude_tools=safe_exclude,
        )

    if name == "claude_cli":
        from app.adapters.claude_worker_cli import ClaudeWorkerCli

        return ClaudeWorkerCli(
            allow_tool_use=True,
            exclude_tools=safe_exclude,
            mcp_servers=_build_mcp_servers_dict(config),
        )

    if name == "lmstudio":
        from app.adapters.lmstudio_worker_api import LmStudioWorkerApi

        lm_cfg = config.lmstudio
        return LmStudioWorkerApi(
            base_url=lm_cfg.base_url or "http://localhost:1234",
            model=lm_cfg.model,
            api_key="lm-studio",
            temperature=config.worker_adapter.temperature,
            max_tokens=config.worker_adapter.max_tokens,
            reasoning_effort=lm_cfg.reasoning_effort,
        )

    if name == "minimax":
        return _create_minimax_adapter(config=config)

    raise ValueError(f"Unknown fallback provider: {name!r}. Supported: qwen_cli, claude_cli, lmstudio, minimax")


def _build_mcp_servers_dict(config: OrchestratorConfig) -> dict[str, dict[str, str]]:
    """Build MCP servers configuration dict from DirectExecutionConfig.

    Returns empty dict if MCP is disabled or auth mode is invalid.
    """
    de = config.direct_execution
    if not de.mcp_endpoint_url:
        return {}

    headers = {}
    if de.mcp_auth_mode.lower() == "bearer" and de.mcp_token_env_var:
        import os
        token = os.environ.get(de.mcp_token_env_var, "")
        if token:
            headers = {"Authorization": f"Bearer {token}"}

    return {
        "dev_space1": {
            "type": "http",
            "url": de.mcp_endpoint_url,
            "headers": headers,
        },
    }


def get_allowed_mcp_tools(allowed_tools: list[str] | None = None) -> list[str]:
    """Extract MCP tool names from allowed_tools list, prefixed with mcp__dev_space1__.

    Filters for tools that start with dev_space1_ or backtests_ and adds
    the mcp__dev_space1__ prefix expected by Claude CLI.
    """
    if not allowed_tools:
        return []
    mcp_prefixes = ("dev_space1_", "backtests_", "datasets_", "models_",
                    "features_", "events_", "experiments_", "system_",
                    "research_", "notify_", "gold_")
    result = []
    for tool in allowed_tools:
        tool_str = str(tool).strip()
        if any(tool_str.startswith(prefix) for prefix in mcp_prefixes):
            result.append(f"mcp__dev_space1__{tool_str}")
    return sorted(set(result))
