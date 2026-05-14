# Operations Guide

## Diagnosing a Hung Runtime

Start with:

```bash
tail -100 logs/orchestrator.log
```

Look for:

- planner startup and first visible text timing
- `thinking_only`
- `tool_use_detected`
- worker launch failures
- wave open/close/finalize messages

## Key Artifact Paths

Check the canonical plan directory:

```bash
ls plans
ls plans/reports
ls plans/waves
ls plans/planner_runs
```

Important files:

- `plan_vN.md`
- `plan_vN_meta.json`
- `reports/plan_vN_report.json`
- `waves/wave_N.json`
- `planner_runs/*.json`

## Planner Failures

If the planner is slow or silent:

1. Inspect `plans/planner_runs/*.json`
2. Check whether the run was classified as `thinking_only`, `tool_use_detected`, or `invalid_output`
3. Verify `planner_adapter.cli_path` manually
4. Verify the planner backend/model mapping from the startup log

## Worker Failures

If workers never start:

1. Verify `worker_adapter` availability at startup
2. Test the worker CLI or LM Studio endpoint manually
3. Inspect plan metadata for `launch_failed` and `launch_error`

The orchestrator now fails fast on missing planner or worker adapters before entering the loop.

## Safe Restart

Normal restart:

```bash
python main.py
```

The runtime restores state, reopens incomplete plans, and resumes the current wave.

## Reset Behavior

`startup_mode = "reset"`:

- archives current state and plan artifacts
- clears active state and plan files
- keeps research context

`startup_mode = "reset_all"`:

- archives current state and plan artifacts
- clears state, plans, and research context
