# Testing

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_orchestrator_happy_path.py -v

# Run with short tracebacks
python -m pytest tests/ -v --tb=short
```

All tests use **fake adapters** (`FakePlanner`, `FakeWorker`) — no real CLI calls are made. Tests run in under 1 second.

## Test Files

### `test_config.py` (6 tests)
Configuration loading and validation:
- Default config creation with sensible values
- Config from dictionary with overrides
- Missing keys fall back to defaults
- Adapter settings are preserved
- State path construction
- Round-trip serialization (`to_dict()`)

### `test_state_store.py` (7 tests)
State persistence via JSON files:
- Save and load empty state
- Save and load state with tasks and results
- Load from nonexistent file returns `None`
- Load from corrupt file returns `None`
- `clear()` removes the state file
- Memory entries survive save/load
- Stop reason survives save/load

### `test_result_parser.py` (16 tests)
Parsing and validating model output:
- Valid planner JSON parsed correctly
- Markdown-fenced JSON (`` ```json ``` ``) extracted
- JSON mixed with surrounding text extracted
- Invalid JSON falls back to `wait` decision
- Unknown decision strings handled
- Partial/incomplete JSON handled
- Valid worker JSON parsed correctly
- Worker error responses parsed
- Worker output with no JSON handled
- Invalid worker status normalized
- Duplicate result detection (same summary)
- Non-duplicate detection (different summary)
- `None` previous result treated as non-duplicate
- Useless result detection (empty output)
- Useless result detection (error with no message)
- Useful result not flagged as useless

### `test_scheduler.py` (9 tests)
Orchestrator scheduling decisions:
- Call planner on first cycle
- Call planner when new results arrive
- Call planner when tasks are pending but no workers active
- Do NOT call planner when workers are still running
- Stop when max errors reached
- Stop when max empty cycles reached
- Stop when planner decided `finish`
- Do NOT stop under normal conditions
- Sleep interval uses configured value
- Call planner when all tasks have settled

### `test_orchestrator_happy_path.py` (3 tests)
End-to-end happy path scenarios:
- Single task: planner launches worker → worker succeeds → planner finishes
- Multiple tasks: planner issues sequential subtasks, all succeed
- Planner receives new results and acts on them

### `test_orchestrator_no_progress.py` (2 tests)
No-progress detection:
- Orchestrator stops after reaching `max_empty_cycles`
- Empty cycle counter resets when a new task is launched

### `test_orchestrator_retry.py` (2 tests)
Worker error retry:
- Worker error triggers retry (planner decides `retry_worker`)
- Task fails permanently after max retry attempts exhausted

### `test_orchestrator_restart_recovery.py` (3 tests)
State recovery after orchestrator restart:
- State persists across restart (loaded from JSON)
- Running tasks marked as `stalled` after restart (subprocesses died)
- Memory entries preserved across restart

### `test_orchestrator_invalid_output.py` (3 tests)
Handling invalid planner output:
- Invalid JSON defaults to `wait` decision
- Empty output handled gracefully
- Failing planner adapter handled without crash

## Critical Test Scenarios

These are the most important paths to verify after any code change:

1. **Happy path** (`test_orchestrator_happy_path.py`) — the core workflow works
2. **Restart recovery** (`test_orchestrator_restart_recovery.py`) — state survives crashes
3. **Invalid output** (`test_orchestrator_invalid_output.py`) — system is resilient to bad model output
4. **No progress stop** (`test_orchestrator_no_progress.py`) — the system stops itself when stuck

## Test Architecture

Tests use dependency injection to pass fake adapters:

```python
planner = FakePlanner(responses=[...], delay=0.01)
worker = FakeWorker(responses=[...], delay=0.01)

orch = Orchestrator(
    config=config,
    state_store=StateStore(tmp_path),
    planner_adapter=planner,
    worker_adapter=worker,
)
```

This allows full control over planner and worker behavior without real CLI calls.
