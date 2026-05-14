# ORC_1

Python orchestration runtime for planner/worker research loops.

The project coordinates a planner model, multiple worker adapters, structured
execution reports, and persisted wave summaries. It was built for long-running
research workflows where each wave can dispatch up to three worker plans in
parallel and then feed the consolidated result back into the next planner call.

## Highlights

- Planner and worker adapter boundaries for Claude CLI, Qwen CLI, fake workers,
  and LM Studio-compatible APIs.
- Structured plan compilation, execution parsing, report identity, and state
  persistence.
- Runtime console and logging helpers for observing long-running loops.
- Unit tests around orchestration guardrails, parsing, direct execution, and
  fallback behavior.
- Documentation for architecture, configuration, execution flow, operations, and
  testing in `docs/`.

## Quickstart

This repository is designed for a local Python environment. The original runtime
used `conda env6`, but any equivalent Python environment with the dependencies
installed should work.

```bash
pip install -r requirements.txt
cp config.example.toml config.toml
python -m pytest
python -m examples.demo_run
```

For the real runtime:

```bash
python main.py
```

## Runtime Files

Runtime state, generated plans, compiled plans, logs, and local agent scratch
files are intentionally not tracked. Keep machine-specific settings in
`config.toml`; use `config.example.toml` as the public template.

## Documentation

- `docs/README.md` - project overview
- `docs/architecture.md` - orchestration architecture
- `docs/configuration.md` - runtime configuration
- `docs/execution_flow.md` - planner/worker execution flow
- `docs/operations.md` - operational notes
- `docs/testing.md` - test guidance
