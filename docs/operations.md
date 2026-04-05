# Operations Guide

## Diagnosing a Hung Orchestrator

If the orchestrator appears stuck and not making progress:

### 1. Check the logs

```bash
tail -50 logs/orchestrator.log
```

Look for:
- Repeated `[no_change]` entries — the planner is not being called because nothing changed
- Repeated `[sleeping]` entries with no `[planner_called]` in between — normal idle behavior
- `[planner_error]` — the planner CLI failed
- `[worker_failed]` — a worker keeps failing

### 2. Check the state file

```bash
cat state/orchestrator_state.json | python -m json.tool | head -30
```

Key fields to inspect:
- `status` — should be `running` during active work, `finished` when done
- `current_cycle` — should increment over time
- `empty_cycles` — if this equals `max_empty_cycles`, the orchestrator stopped due to no progress
- `total_errors` — if this equals `max_errors_total`, the orchestrator stopped due to too many errors
- `tasks[].status` — look for tasks stuck in `running` or `waiting_result`

### 3. Check for alive processes

```bash
ps aux | grep -E "claude|qwen-code" | grep -v grep
```

If worker subprocess processes are alive but the orchestrator logs show no new results, the worker may be hung. The orchestrator will eventually time it out based on `worker_timeout_seconds`.

## CLI Errors

### Planner CLI returns non-zero exit code

The `ClaudePlannerCli` adapter catches subprocess errors and returns a fallback `PlannerOutput` with `decision=wait`. This prevents a single CLI crash from stopping the entire system. The error is logged and `total_errors` is incremented.

If errors persist:
1. Verify the CLI path is correct in config (`planner_adapter.cli_path`)
2. Check if the CLI works manually: `claude --print "test"`
3. Check stderr in the logs for specific error messages
4. Increase `planner_timeout_seconds` if the model is slow to respond

### Worker CLI returns non-zero exit code

The `QwenWorkerCli` adapter similarly catches errors and returns a `TaskResult` with `status=error`. The task is marked as failed and the planner decides whether to retry.

If workers consistently fail:
1. Verify the CLI path (`worker_adapter.cli_path`)
2. Test manually: `qwen-code --print "test"`
3. Check if the task description is too complex — break it into smaller pieces
4. Increase `worker_timeout_seconds`

## Invalid Model Output

### Planner returns non-JSON or malformed JSON

The `result_parser` handles several cases:
- **Markdown-wrapped JSON** (`` ```json ... ``` ``) — extracted automatically
- **JSON mixed with text** — first JSON object found is used
- **Completely invalid output** — defaults to `decision=wait` so the system doesn't crash
- **Unknown decision string** — defaults to `wait`

If the planner consistently returns unparseable output:
1. Check `planner_system_prompt` — ensure it instructs JSON-only output
2. Set `require_structured_output=true` in config (enforced by default)
3. Manually inspect the planner's raw output from the logs (enable DEBUG level)

### Worker returns non-JSON

Workers follow the same fallback pattern. Invalid output produces a `TaskResult` with `status=error`. The orchestrator increments `total_errors` and the planner decides the next action.

## Safe Restart Procedure

The orchestrator is designed for safe restarts:

### Normal restart

```bash
# Send SIGTERM to the orchestrator process
kill <PID>

# Or press Ctrl+C if running in foreground

# Wait for current cycle to complete (state is saved every cycle)

# Restart
python -m app.main
```

### What happens during restart

1. Orchestrator loads `state/orchestrator_state.json`
2. Any tasks that were `running` or `waiting_result` → marked `stalled`
3. Memory entries are preserved
4. Error counts and cycle counts are preserved
5. The planner sees stalled tasks and can retry, reassign, or finish

### Forced restart (if process is unresponsive)

```bash
kill -9 <PID>

# State from the last completed cycle is on disk
# Restart normally
python -m app.main
```

State is saved atomically (temp file + rename), so even a `kill -9` cannot corrupt the state file. At worst, you lose the current in-progress cycle.

## State Cleanup

### Fresh start (delete all state)

```bash
rm -rf state/
```

The orchestrator creates a new empty state on next run.

### Reset specific tasks

Edit `state/orchestrator_state.json` directly:
- Set `status` to `"idle"` to restart from scratch
- Set specific tasks' `status` to `"cancelled"` to skip them
- Set `total_errors` to `0` to reset error counters
- Set `empty_cycles` to `0` to reset the no-progress counter

### Clean logs

```bash
rm -rf logs/*
```

Logs are recreated automatically.

## Tuning for Production

### Reduce token costs
- Increase `poll_interval_seconds` (default 300 = 5 min)
- Lower `max_empty_cycles` (default 12) to stop sooner on no-progress
- Lower `max_errors_total` (default 20) to fail fast on errors

### Improve resilience
- Increase `max_task_attempts` (default 3) for retry-hungry tasks
- Increase `planner_timeout_seconds` for complex planning decisions
- Increase `worker_timeout_seconds` for long-running subtasks

### Monitor actively
- Set `log_level=DEBUG` during development
- Watch `empty_cycles` — if it consistently reaches `max_empty_cycles`, the planner may need better prompts or the goal may be unachievable
- Watch `total_errors` — a steady increase suggests systemic issues (bad CLI config, unachievable tasks)

## Using LM Studio as a Worker

LM Studio provides a local OpenAI-compatible API for running models like Qwen, Llama, or DeepSeek locally.

### Prerequisites

1. Install [LM Studio](https://lmstudio.ai/)
2. Download a model (e.g., `qwen3-4b`, `llama-3-8b`)
3. Start the local server: Developer tab → Start Server (default `localhost:1234`)

### Configuration

```json
{
  "goal": "Your objective",
  "worker_adapter": {
    "name": "lmstudio_worker_api",
    "base_url": "http://localhost:1234",
    "model": "qwen3-4b",
    "temperature": 0.7,
    "max_tokens": 4096
  }
}
```

### Verifying LM Studio is running

```bash
curl http://localhost:1234/v1/models
```

Should return a JSON list of loaded models. If this fails, LM Studio is not serving.

### Troubleshooting

- **Connection refused** — LM Studio server not started. Open LM Studio, go to Developer tab, start server.
- **Model not found** — Ensure a model is loaded in LM Studio before starting the orchestrator. The `model` field in config must match the model identifier shown in LM Studio.
- **Timeout** — Increase `worker_timeout_seconds` in config. Local inference on CPU can be slow for large models.
- **Empty response** — Check LM Studio console for errors. The model may have crashed or OOM'd.
