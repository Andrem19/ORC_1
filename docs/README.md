# CLI Agent Orchestrator

## Environment

This repository is pinned to `conda env6`.

- Open the project terminal only with `conda env6`.
- Do not create or use `.venv` or any other local virtualenv in this repository.
- Run commands as `conda run -n env6 ...` or from an already activated `env6` shell.

This project runs one canonical research loop:

- Claude Code CLI acts as the planner and returns one markdown plan.
- Qwen Code CLI or LM Studio acts as the worker and returns one structured JSON report.
- The orchestrator fills waves of up to 3 parallel plans, waits for the wave to resolve, summarizes results, and asks the planner for the next wave.

## Entry Points

Run the real runtime:

```bash
conda run -n env6 python main.py
```

Run the fake demo:

```bash
conda run -n env6 python -m examples.demo_run
```

## Main Packages

```text
app/
  orchestrator.py
  config.py
  models.py
  scheduler.py
  plan_models.py
  plan_store.py
  plan_prompts.py
  worker_report_parser.py
  planner_stream.py
  state_store.py

  services/
    planner_service.py
    worker_service.py
    notification_service.py
    plan_orchestrator/
      _core.py
      _plan_io.py
      _dispatch.py
```

## Contracts

- Planner output: plain markdown plan
- Worker output: structured JSON plan report
- Persisted artifacts: plan markdown, plan metadata, worker reports, wave summaries, planner diagnostics

There is no legacy alternate planner runtime anymore.
