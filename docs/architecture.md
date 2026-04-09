# Architecture

## Overview

The repository now has one orchestration architecture only:

- one Python supervisor loop
- one markdown planner contract
- one worker JSON report contract
- one wave barrier model for parallel execution

```text
Orchestrator
  -> PlannerService
     -> Claude Code CLI
  -> PlanOrchestratorService
     -> build wave
     -> dispatch plans to workers
     -> finalize wave summaries
  -> WorkerService
     -> Qwen Code CLI or LM Studio
  -> StateStore / PlanStore
```

## Planner Contract

The planner returns one plain markdown plan per call.

Required shape:

- `# Plan vN`
- baseline/status framing
- research principles
- capability/tool section
- 3-5 `## ETAP N: ...` sections

The planner never returns JSON decisions and does not control worker lifecycle directly.

## Worker Contract

Each worker executes one markdown plan and returns one JSON report.

The report captures:

- execution status
- what was requested
- what was done
- results table
- key metrics
- artifacts
- verdict
- confidence
- error
- MCP problems

## Wave Execution Model

1. Start a wave.
2. Ask the planner for slot 1.
3. Dispatch slot 1 to an idle worker.
4. Continue requesting plans until the wave reaches `min(3, len(workers), max_concurrent_plan_tasks)`.
5. Wait until all plans in the wave are terminal.
6. Build a wave summary.
7. Feed that summary into the next planner call.

## Persistence

Runtime state is split into:

- `state/` for orchestrator process state
- `plans/` for plan markdown, metadata, reports, wave summaries, and planner diagnostics

Canonical artifacts:

- `plan_vN.md`
- `plan_vN_meta.json`
- `reports/plan_vN_report.json`
- `waves/wave_N.json`
- `planner_runs/*.json`

## Removed Legacy Concepts

These are intentionally gone:

- planner-side control enums and command routing
- auxiliary memory and task-supervision layers
- legacy planner-output extraction helpers
- dual-runtime execution branches
