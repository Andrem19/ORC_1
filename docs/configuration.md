# Configuration

All parameters are defined in `app/config.py` and can be overridden via `config.toml` or function arguments. No argparse.

## Environment

This project uses **conda environment `env6`** (Python 3.10.18).

```bash
# Conda binary (full path — conda may not be on PATH)
CONDA=/home/jupiter/miniconda3/bin/conda

# Run the orchestrator
$CONDA run -n env6 python main.py

# Install packages
$CONDA run -n env6 pip install <package>

# Run tests (parallel, 4 workers by default)
$CONDA run -n env6 pytest

# Run tests sequentially (for debugging)
$CONDA run -n env6 pytest -n 0

# Custom worker count
$CONDA run -n env6 pytest -n 2
```

Tests run in parallel using **pytest-xdist** (4 worker processes by default, configured in `pyproject.toml`).

**Do NOT** use bare `python` or `pip` — they resolve to the system Python, not the project environment.

## Configuration File

Configuration is loaded from `config.toml` in the project root:

```json
{
  "goal": "Your high-level objective",

  "planner_system_prompt": "Custom system prompt for the planner",
  "worker_system_prompt": "Custom system prompt for workers",

  "workers": [
    {"worker_id": "qwen-1", "role": "executor", "system_prompt": ""},
    {"worker_id": "qwen-2", "role": "reviewer", "system_prompt": ""}
  ],

  "poll_interval_seconds": 300,
  "planner_timeout_seconds": 180,
  "worker_timeout_seconds": 300,

  "max_errors_total": 20,
  "max_empty_cycles": 12,
  "max_task_attempts": 3,
  "max_worker_timeout_count": 3,

  "planner_adapter": {
    "name": "claude_planner_cli",
    "cli_path": "claude",
    "model": "opus",
    "extra_flags": []
  },

  "worker_adapter": {
    "name": "qwen_worker_cli",
    "cli_path": "qwen-code",
    "extra_flags": []
  },

  "state_dir": "state",
  "state_file": "orchestrator_state.json",

  "log_level": "INFO",
  "log_dir": "logs",
  "log_file": "orchestrator.log"
}
```

## Parameter Reference

### Goal & Prompts

| Parameter | Default | Description |
|---|---|---|
| `goal` | "No goal specified" | The high-level objective |
| `planner_system_prompt` | Built-in default | System instructions for the planner model |
| `worker_system_prompt` | Built-in default | System instructions for worker models |

### Workers

| Parameter | Default | Description |
|---|---|---|
| `workers` | One qwen-1 executor | List of worker configs with id, role, and system prompt |

### Timing

| Parameter | Default | Description |
|---|---|---|
| `poll_interval_seconds` | 300 | Seconds between orchestrator cycles |
| `planner_timeout_seconds` | 180 | Max wait for planner CLI response |
| `worker_timeout_seconds` | 300 | Max wait for worker CLI response |
| `worker_restart_delay_seconds` | 10 | Delay before restarting a worker |

### Limits

| Parameter | Default | Description |
|---|---|---|
| `max_errors_total` | 20 | Stop after this many total errors |
| `max_empty_cycles` | 12 | Stop after this many cycles with no progress |
| `max_task_attempts` | 3 | Max retries for a single task |
| `max_worker_timeout_count` | 3 | Max timeouts before force-stopping a task |

### Adapters

| Parameter | Default | Description |
|---|---|---|
| `planner_adapter.cli_path` | "claude" | Path to Claude CLI |
| `planner_adapter.model` | "opus" | Claude model to use |
| `planner_adapter.extra_flags` | [] | Additional CLI flags |
| `worker_adapter.cli_path` | "qwen-code" | Path to Qwen CLI (used when `name` is not `lmstudio_worker_api`) |
| `worker_adapter.extra_flags` | [] | Additional CLI flags |

### LM Studio Adapter

To use LM Studio as the worker instead of Qwen CLI, set the `worker_adapter` section:

```json
{
  "worker_adapter": {
    "name": "lmstudio_worker_api",
    "base_url": "http://localhost:1234",
    "model": "qwen3-4b",
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

| Parameter | Default | Description |
|---|---|---|
| `worker_adapter.name` | "qwen_worker_cli" | Set to `lmstudio_worker_api` to use LM Studio |
| `worker_adapter.base_url` | "http://localhost:1234" | LM Studio server URL |
| `worker_adapter.model` | "" | Model identifier (empty = use currently loaded model) |
| `worker_adapter.api_key` | "lm-studio" | API key (LM Studio doesn't require auth by default) |
| `worker_adapter.temperature` | 0.7 | Sampling temperature |
| `worker_adapter.max_tokens` | 4096 | Max tokens in response |

Prerequisites:
1. Install and run [LM Studio](https://lmstudio.ai/)
2. Load a model in LM Studio
3. Start the local server (Developer tab → Start Server, default port 1234)

### State & Logging

| Parameter | Default | Description |
|---|---|---|
| `state_dir` | "state" | Directory for state files |
| `state_file` | "orchestrator_state.json" | State filename |
| `log_level` | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `log_dir` | "logs" | Log directory |
| `log_file` | "orchestrator.log" | Log filename |

### Feature Flags

| Parameter | Default | Description |
|---|---|---|
| `detect_duplicate_results` | true | Skip duplicate worker results |
| `require_structured_output` | true | Enforce JSON output parsing |

## Cost Optimization

To reduce token usage:

1. **Increase `poll_interval_seconds`** — fewer cycles = fewer planner calls
2. **Set `max_empty_cycles` lower** — stop sooner when nothing's happening
3. **Tune worker `system_prompt`** — shorter prompts = fewer input tokens
4. **Set `max_errors_total` conservatively** — stop early on repeated failures

The orchestrator already avoids calling the planner when nothing changed. Increasing the poll interval compounds this saving.
