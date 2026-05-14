"""
Demo run — shows the full orchestrator lifecycle with fake adapters.

This demonstrates:
1. Starting with a goal
2. Planner returning markdown plans
3. Workers executing those plans
4. Wave-based planning and result collection
5. Final completion

Run: python -m examples.demo_run
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import OrchestratorConfig
from app.logging_setup import setup_logging
from app.orchestrator import Orchestrator
from app.state_store import StateStore


def main():
    logger = setup_logging(log_level="DEBUG", log_dir="logs", log_file="demo.log")

    config = OrchestratorConfig(
        goal="Build a simple REST API with user authentication",
        poll_interval_seconds=0,  # No real delay for demo
        max_empty_cycles=10,
    )

    # Scripted planner responses simulating a realistic wave workflow
    planner_responses = [
        """# Plan v1

## Status and Frame
Baseline is fixed for the demo.

## Goal
Build the Flask project skeleton and authentication base.

## Baseline
Treat the current repo state as fixed input.

## Research Principles
- Execute concrete implementation work only.

## dev_space1 Capabilities
Workers available: 1.

## ETAP 1: Scaffold
Goal: create the initial Flask project structure.
1. Create `app.py`, `requirements.txt`, and `config.py`.
Completion criteria: project files exist.
| artifact | status |
| --- | --- |
| scaffold | pending |

## ETAP 2: User Model
Goal: implement the user model with password hashing.
1. Add a SQLAlchemy user model with bcrypt helpers.
Completion criteria: model and password helpers exist.
| artifact | status |
| --- | --- |
| model | pending |

## ETAP 3: Auth Routes
Goal: implement auth endpoints.
1. Add register, login, and logout routes.
Completion criteria: all routes return JSON.
| artifact | status |
| --- | --- |
| auth | pending |
""",
        """# Plan v2

## Status and Frame
Continue from the scaffold and auth base.

## Goal
Add validation and error handling.

## Baseline
Do not rewrite the existing structure.

## Research Principles
- Extend the system cleanly without churn.

## dev_space1 Capabilities
Workers available: 1.

## ETAP 1: Validation
Goal: validate auth inputs.
1. Add schemas for incoming auth payloads.
Completion criteria: invalid requests are rejected cleanly.
| artifact | status |
| --- | --- |
| validation | pending |

## ETAP 2: Error Handling
Goal: normalize error responses.
1. Add JSON error handlers for common failures.
Completion criteria: 400/401/404/500 handlers exist.
| artifact | status |
| --- | --- |
| errors | pending |

## ETAP 3: Summary
Goal: capture final integration state.
1. Return a structured worker report with artifacts.
Completion criteria: report includes produced files and status.
| artifact | status |
| --- | --- |
| summary | pending |
""",
    ]

    # Scripted worker responses
    worker_responses = [
        {
            "status": "success",
            "what_was_requested": "Build the Flask scaffold and auth base",
            "what_was_done": "Created Flask project files, implemented the user model, and added auth routes returning JSON.",
            "results_table": [{"artifact": "app.py", "status": "created"}],
            "key_metrics": {"files_created": 4},
            "artifacts": ["app.py", "requirements.txt", "config.py", "routes/auth.py"],
            "verdict": "PROMOTE",
            "confidence": 0.95,
            "error": "",
        },
        {
            "status": "success",
            "what_was_requested": "Add validation and error handling",
            "what_was_done": "Added validation schemas and JSON error handlers for common HTTP failures.",
            "results_table": [{"artifact": "schemas.py", "status": "created"}],
            "key_metrics": {"handlers_added": 4},
            "artifacts": ["schemas.py", "errors.py"],
            "verdict": "PROMOTE",
            "confidence": 0.85,
            "error": "",
        },
    ]

    planner = FakePlanner(responses=planner_responses, delay=0.01)
    worker = FakeWorker(responses=worker_responses, delay=0.01)

    tmp = Path(tempfile.mkdtemp())
    store = StateStore(tmp / "demo_state.json")

    orch = Orchestrator(
        config=config,
        state_store=store,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    logger.info("=" * 60)
    logger.info("DEMO: Orchestrator starting")
    logger.info("Goal: %s", config.goal)
    logger.info("=" * 60)

    reason = orch.run()

    logger.info("=" * 60)
    logger.info("DEMO: Orchestrator finished — %s", reason.value)
    logger.info("Cycles: %d", orch.state.current_cycle)
    logger.info("Completed tasks: %d", len(orch.state.completed_tasks()))
    logger.info("Total results: %d", len(orch.state.results))
    logger.info("=" * 60)

    # Print task summary
    print("\n--- Task Summary ---")
    for task in orch.state.tasks:
        print(f"  [{task.status.value}] {task.description[:80]}")

    print("\n--- Results ---")
    for result in orch.state.results:
        print(f"  [{result.status}] {result.summary[:80]}")

    print(f"\nStop reason: {reason.value}")
    print(f"Total cycles: {orch.state.current_cycle}")


if __name__ == "__main__":
    main()
