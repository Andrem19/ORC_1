# Execution Flow

## Startup

1. Orchestrator initializes with config
2. Attempts to load saved state from `state/state.json`
3. If state exists and has running tasks → marks them as STALLED (subprocesses died)
4. Saves updated state
5. Enters main loop

## Main Loop

```
while True:
    cycle++
    
    1. Collect results
       - For each WAITING_RESULT task, call worker adapter
       - Parse output, check for duplicates
       - Handle result (complete/fail task)
    
    2. Decide: call planner?
       - YES if: first cycle, new results, pending tasks with no active workers,
         or all tasks settled (need next move)
       - NO if: active workers still running, nothing changed
    
    3. If calling planner:
       a. Build concise prompt (goal + memory + state + new results)
       b. Call planner adapter (Claude CLI)
       c. Parse JSON response into PlannerOutput
       d. Execute decision:
          - launch_worker: create task, set WAITING_RESULT
          - wait: do nothing
          - retry_worker: find failed task, reset and reassign
          - stop_worker: cancel tasks for specified worker
          - reassign_task: move task to different worker
          - finish: exit loop
       e. If WAIT decision: increment empty_cycles
    
    4. If NOT calling planner:
       - Increment empty_cycles
    
    5. Check stop conditions:
       - total_errors >= max_errors_total → STOP
       - empty_cycles >= max_empty_cycles → STOP
       - planner decided FINISH → STOP
    
    6. Save state to disk
    
    7. Sleep for poll_interval_seconds
```

## When Planner Is Called

| Condition | Why |
|---|---|
| First cycle (cycle 0) | Need initial plan |
| New results arrived | Need to decide what's next |
| Pending tasks, no active workers | Need to assign work |
| All tasks done/failed | Need next move or finish |

## When Planner Is NOT Called

| Condition | Why |
|---|---|
| Active workers still running | Nothing to decide yet |
| No new results, no pending tasks | Waiting for in-flight work |

## Planner Decision Outcomes

### `launch_worker`
Creates a new Task, assigns it to a worker, sets status to WAITING_RESULT. Next cycle, the worker service will execute this task.

### `wait`
Does nothing. Increments empty_cycles. Useful when workers are expected to produce results soon.

### `retry_worker`
Finds the most recently failed task, resets it (increments attempt counter), and sets status to WAITING_RESULT with optionally updated instructions.

### `stop_worker`
Cancels all active tasks for the specified worker. Used when a worker is stuck or misbehaving.

### `reassign_task`
Moves an active/stalled task to a different worker with optionally updated instructions.

### `finish`
Exits the main loop. The orchestrator saves final state and returns the stop reason.

## Sleep Behavior

- Default: `poll_interval_seconds` (300s = 5 minutes)
- During sleep, no model calls are made — zero token cost
- After wake, orchestrator checks state before deciding whether to call planner

## State Saving

- State is saved after every cycle
- Uses atomic writes (temp file + rename) to prevent corruption
- State includes: all tasks, results, memory, error counts, cycle count, planner decisions

## Recovery After Restart

When the orchestrator process restarts:

1. Load state from disk
2. Any task that was RUNNING or WAITING_RESULT → marked STALLED
3. STALLED tasks are treated like FAILED tasks by the planner
4. Planner can decide to retry, reassign, or finish
5. No duplicate tasks are created
6. Memory is preserved from previous run
