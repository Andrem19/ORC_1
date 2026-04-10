# Configuration

The runtime reads `config.toml` and builds one canonical markdown planner-worker orchestrator.

## Environment

Use only the project conda env `env6`. Do not create, activate, or use `.venv`, `venv`, bare `python`, or bare `pip` inside this repository.

```bash
CONDA=/home/jupiter/miniconda3/bin/conda
$CONDA activate env6
$CONDA run -n env6 python main.py
$CONDA run -n env6 pytest -q
```

For terminals opened in VS Code, the workspace is configured to start directly in `conda env6`.

## Core Runtime Settings

Key fields in `config.toml`:

| Parameter | Description |
| --- | --- |
| `goal` | High-level research objective |
| `operator_directives` | Highest-priority planner instructions |
| `planner_prompt_template` | Visible markdown plan template |
| `worker_system_prompt` | Shared worker prompt prefix |
| `workers` | Logical worker slots used for wave dispatch |
| `plan_dir` | Directory for plans, reports, waves, and planner diagnostics |
| `max_concurrent_plan_tasks` | Upper bound for wave size before the barrier |
| `plan_task_timeout_seconds` | Timeout for one worker plan execution |
| `poll_interval_seconds` | Idle poll interval |
| `max_errors_total` | Stop after too many total errors |
| `max_empty_cycles` | Stop after too many cycles with no progress |

The effective wave width is:

```text
min(3, len(workers), max_concurrent_plan_tasks)
```

## Planner Adapter

Example:

```toml
[planner_adapter]
name = "claude_planner_cli"
cli_path = "claude"
model = "opus"
mode = "stream_json"
```

The planner must return one markdown plan per call.

## Worker Adapter

CLI example:

```toml
[worker_adapter]
name = "qwen_worker_cli"
cli_path = "qwen-code"
```

LM Studio example:

```toml
[worker_adapter]
name = "lmstudio_worker_api"
base_url = "http://localhost:1234"
model = "qwen/qwen3.5-9b"
```

## Notifications

Telegram notifications are optional observer behavior. They must not block startup.

If `notifications.translation_backend = "lmstudio"`, the shared `[lmstudio]` connection and `[lmstudio.translation]` limits are used only for translation.

## Removed Legacy Fields

Legacy dual-runtime configuration flags and old planner-output controls are no longer part of the runtime architecture.
