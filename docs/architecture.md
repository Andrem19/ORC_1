# Architecture

## Overview

The orchestrator follows a **supervisor pattern** where an external Python process manages the lifecycle of model interactions, rather than keeping models in an endless chat loop.

```
┌──────────────────────────────────────────────────┐
│                  Orchestrator                     │
│  (Python process — holds state, drives loop)     │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Scheduler │  │  State   │  │    Memory    │   │
│  │          │  │  Store   │  │   Service    │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
│                                                  │
│  ┌─────────────────┐  ┌──────────────────────┐  │
│  │ Planner Service  │  │  Worker Service      │  │
│  │  (Claude CLI)    │  │  (Qwen / LM Studio)  │  │
│  └────────┬────────┘  └──────────┬───────────┘  │
│           │                      │               │
└───────────┼──────────────────────┼───────────────┘
            │                      │
    ┌───────▼──────┐    ┌──────────▼──────┐
    │ Claude Code   │    │ Qwen Code CLI   │
    │ CLI (planner) │    │ or LM Studio    │
    └───────────────┘    │ (worker agent)  │
                         └─────────────────┘
```

## Roles

### Orchestrator (Python process)
- Owns the main loop
- Stores all state externally (JSON files)
- Decides WHEN to call models
- Handles errors, timeouts, retries
- Survives restarts via state persistence

### Planner (Claude)
- Called ONLY when state changes or a decision is needed
- Receives: goal, concise memory, current state summary, new results
- Returns: structured JSON with one of 6 decisions
- Never waits or polls — pure decision-making

### Workers (Qwen / LM Studio)
- Receive atomic, well-scoped tasks
- Return structured JSON results
- No architectural decisions
- No access to global strategy
- Can use Qwen Code CLI (subprocess) or LM Studio (HTTP API)

## Data Flow

```
1. Orchestrator loads config + state
       │
2. Collect results from active workers
       │
3. Scheduler: should we call planner?
       │
   ┌───YES──────────────────────────┐
   │                                 │
   │  4. Build concise prompt        │
   │     (goal + memory + state      │
   │      + new results)             │
   │                                 │
   │  5. Call planner → JSON output  │
   │                                 │
   │  6. Parse decision              │
   │     launch_worker / wait /      │
   │     retry / stop / reassign /   │
   │     finish                      │
   │                                 │
   │  7. Execute decision            │
   │                                 │
   └─────────────────────────────────┘
       │
       NO (no state change)
       │
       Increment empty_cycles
       │
8. Check stop conditions
   (max errors / max empty / finish)
       │
9. Save state → sleep → repeat
```

## Task Lifecycle

```
PENDING → RUNNING → WAITING_RESULT → COMPLETED
                                      or FAILED
                                      or TIMED_OUT
                                      or STALLED
                                      or CANCELLED
```

- **PENDING** — created, not assigned
- **RUNNING** — assigned to a worker
- **WAITING_RESULT** — worker executing, awaiting output
- **COMPLETED** — successfully finished
- **FAILED** — worker returned error, awaiting planner decision
- **TIMED_OUT** — worker exceeded time limit
- **STALLED** — task was active during orchestrator restart
- **CANCELLED** — explicitly cancelled by planner or system

## Why This Architecture

### External loop instead of in-context orchestration
Models are expensive and stateless. Keeping them in a live chat wastes tokens on waiting. The external loop:
- Sleeps between checks (zero token cost)
- Only invokes models when there's something to decide
- Maintains state outside the model context

### JSON-based structured communication
Free-text responses are unreliable for automation. Both planner and workers must return JSON matching a defined schema. This makes parsing deterministic and errors catchable.

### Adapter pattern for CLI independence
Each CLI (Claude, Qwen, or any future tool) is wrapped in an adapter implementing `BaseAdapter`. The core orchestrator never imports CLI-specific code.

### File-based state for simplicity
JSON files are human-readable, debuggable, and require no external dependencies. Atomic writes via temp+rename prevent corruption. For a single-process orchestrator, this is more reliable than SQLite.

## Extensibility Points

| What to change | Where |
|---|---|
| Replace planner model | New adapter implementing `BaseAdapter` |
| Replace worker model | New adapter implementing `BaseAdapter` |
| Use LM Studio as worker | Set `worker_adapter.name` to `lmstudio_worker_api` in config |
| Add MCP integration | New adapter wrapping MCP protocol |
| Add task queue | Replace `Scheduler` with event-driven version |
| Add more workers | Add entries to `config.workers` |
| Different worker roles | Use `WorkerConfig.role` to route tasks |
| Change state storage | Implement new `StateStore`-compatible class |
| Add monitoring | Subscribe to `OrchestratorEvent` log entries |
