# Testing

## Run Tests

```bash
python -m pytest -q
```

Use focused runs while iterating:

```bash
python -m pytest -q tests/test_plan_orchestrator.py
python -m pytest -q tests/test_main_startup.py
```

## What Is Covered

The fast unit suite focuses on the surviving runtime only:

- config/bootstrap validation
- planner stream rendering and watchdog classification
- plan store round-trips
- state store round-trips
- wave filling and barrier behavior
- planner retry and fail-fast behavior
- worker dispatch failure accounting
- reset/archive behavior

## Test Style

- Use fake adapters or mocks only
- Avoid real CLI calls
- Keep filesystem usage isolated to temporary directories
- Prefer narrow unit coverage over broad slow integration tests
