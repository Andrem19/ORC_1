# CLI Agent Orchestrator

A resilient, token-efficient orchestrator that manages a planner model (Claude) and worker agents (Qwen) through an external polling loop.

## What It Does

The orchestrator breaks a high-level goal into subtasks, assigns them to worker agents, monitors progress, and adapts the plan — all while minimizing model calls.

**Key principle:** Models are invoked only when state actually changes. Between checks, the orchestrator sleeps.

## Quick Start

### Run the demo (no real CLI needed)

```bash
python -m examples.demo_run
```

### Run tests

```bash
python -m pytest tests/ -v
```

### Run with real CLIs

1. Create `config.json` in the project root:

```json
{
  "goal": "Build a REST API with authentication",
  "poll_interval_seconds": 60,
  "planner_adapter": {"cli_path": "claude", "model": "opus"},
  "worker_adapter": {"cli_path": "qwen-code"},
  "workers": [
    {"worker_id": "qwen-1", "role": "executor", "system_prompt": ""}
  ]
}
```

2. Run:

```bash
python -m app.main
```

## Project Structure

```
app/
  models.py              # Data models, enums, state
  config.py              # Configuration
  orchestrator.py         # Main loop
  scheduler.py           # Timing and planner-call decisions
  prompts.py             # Prompt construction
  result_parser.py       # JSON extraction and validation
  state_store.py         # Persistent state (JSON files)
  logging_setup.py       # Logging configuration
  main.py                # Entry point

  adapters/
    base.py              # Abstract adapter interface
    claude_planner_cli.py  # Claude CLI adapter
    qwen_worker_cli.py     # Qwen CLI adapter
    lmstudio_worker_api.py # LM Studio HTTP API adapter
    fake_planner.py       # Fake planner for testing
    fake_worker.py        # Fake worker for testing

  services/
    planner_service.py   # Planner interaction logic
    worker_service.py    # Worker interaction logic
    task_supervisor.py   # Stop/retry/reassign decisions
    memory_service.py    # Short-term memory management

tests/                   # 53 tests covering all scenarios
examples/                # Demo configuration and run script
docs/                    # Documentation
```

## Architecture

See [architecture.md](architecture.md) for details.

**Three roles:**
- **Orchestrator** — external Python loop, holds state, calls models only when needed
- **Planner (Claude)** — analyzes state, decides next action, returns structured JSON
- **Workers (Qwen or LM Studio)** — execute atomic subtasks, return structured JSON results

## Key Features

- **Token-efficient:** Planner is called only on state changes, not every cycle
- **Resilient:** Survives subprocess crashes, invalid JSON, timeouts, and process restarts
- **Extensible:** Swap adapters without changing core logic
- **Testable:** Full test suite with fake adapters, no real CLIs needed
- **Stateful:** JSON-based state persistence survives orchestrator restarts
