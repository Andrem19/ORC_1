# Execution Flow

## Canonical Runtime

The orchestrator now has one execution model only:

1. Start one planner request.
2. Planner returns one markdown plan.
3. Dispatch that plan to one worker.
4. Keep asking the planner for more plans until the current wave reaches `min(3, len(workers), max_concurrent_plan_tasks)`.
5. Wait until every dispatched plan in the wave reaches a terminal state.
6. Aggregate the wave results and ask the planner for the next wave.

There is no legacy dual-runtime planner branch anymore.

## Startup

1. Load `config.toml`
2. Build planner and worker adapters
3. Fail fast if either adapter is unavailable
4. Load persisted state and mark stale in-flight tasks as stalled
5. Restore saved plans, reports, planner runs, and wave summaries from `plan_dir`
6. Enter the polling loop

## Per-Cycle Loop

```text
while True:
    cycle += 1
    collect worker results
    process plan reports
    if planner is running:
        read streamed planner output
        classify thinking_only / tool_use_detected / visible_text_started
        retry compact -> strict when needed
    dispatch pending plans to idle workers
    if current wave is complete:
        finalize wave summary
    if planner idle and current wave still filling:
        request next plan slot
    save state
    sleep
```

## Planner Contract

The planner returns plain markdown only.

Required shape:

- Starts with `# Plan vN`
- Includes Status and Frame, Goal, Baseline, Research Principles, and dev_space1 Capabilities
- Includes 3-5 `## ETAP N: ...` sections
- Each ETAP contains concrete MCP tool steps, completion criteria, and a result-table template

## Worker Contract

Each worker executes one markdown plan and returns one JSON report with:

- `status`
- `what_was_requested`
- `what_was_done`
- `results_table`
- `key_metrics`
- `artifacts`
- `verdict`
- `confidence`
- `error`
- `mcp_problems`

## Persistence

Canonical artifacts inside `plan_dir`:

- `plan_vN.md`
- `plan_vN_meta.json`
- `reports/plan_vN_report.json`
- `waves/wave_N.json`
- `planner_runs/*.json`
- `cumulative_findings.txt`

## Restart Recovery

On restart:

1. State is loaded from disk.
2. Tasks left in `RUNNING` or `WAITING_RESULT` are marked stalled.
3. Persisted plan metadata is restored.
4. Plans left in `running` are reopened as `pending` so they can be dispatched again.
5. The current wave is resumed or reconstructed from saved plan metadata.
