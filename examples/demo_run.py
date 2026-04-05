"""
Demo run — shows the full orchestrator lifecycle with fake adapters.

This demonstrates:
1. Starting with a goal
2. Planner breaking work into subtasks
3. Workers executing subtasks
4. Planner reviewing results and issuing next tasks
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

    # Scripted planner responses simulating a realistic workflow
    planner_responses = [
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Create Flask project structure: app.py, requirements.txt, config.py",
            "reason": "Starting with project scaffolding",
            "check_after_seconds": 0,
        },
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Implement User model with SQLAlchemy and password hashing",
            "reason": "Project structure ready, now implementing the data model",
            "memory_update": "Project structure created with Flask, SQLAlchemy",
            "check_after_seconds": 0,
        },
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Implement auth routes: POST /register, POST /login, POST /logout",
            "reason": "User model exists, adding authentication endpoints",
            "memory_update": "User model with bcrypt password hashing done",
            "check_after_seconds": 0,
        },
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Add input validation and error handlers",
            "reason": "Auth routes working, adding safety layer",
            "memory_update": "Auth routes with JWT tokens implemented",
            "check_after_seconds": 0,
        },
        {
            "decision": "finish",
            "reason": "All components built and verified",
            "should_finish": True,
            "final_summary": "REST API built successfully: Flask app with User model, auth routes, JWT tokens, and input validation.",
        },
    ]

    # Scripted worker responses
    worker_responses = [
        {
            "status": "success",
            "summary": "Created Flask project: app.py with factory pattern, requirements.txt (Flask, flask-sqlalchemy, flask-bcrypt, pyjwt), config.py with env-based settings",
            "artifacts": ["app.py", "requirements.txt", "config.py"],
            "confidence": 0.95,
        },
        {
            "status": "success",
            "summary": "Implemented User model with SQLAlchemy: id, username, email, password_hash fields. Added bcrypt hashing via set_password/check_password methods.",
            "artifacts": ["models.py"],
            "confidence": 0.9,
        },
        {
            "status": "success",
            "summary": "Implemented /register (creates user, returns 201), /login (validates credentials, returns JWT), /logout (blacklists token). All routes return JSON.",
            "artifacts": ["routes/auth.py"],
            "confidence": 0.88,
        },
        {
            "status": "success",
            "summary": "Added marshmallow schemas for input validation on all auth routes. Added 400/401/404/500 error handlers returning JSON responses.",
            "artifacts": ["schemas.py", "errors.py"],
            "confidence": 0.85,
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
    logger.info("Memory entries: %d", len(orch.state.memory))
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
