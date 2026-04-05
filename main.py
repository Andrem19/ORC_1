"""
Entry point for the orchestrator.

Usage:
    python main.py              # Production mode (reads config.json)
    python main.py --demo       # Demo mode with fake adapters
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

# Load .env file if present (secrets like API keys, tokens)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

from app.config import OrchestratorConfig, load_config_from_dict
from app.logging_setup import setup_logging
from app.orchestrator import Orchestrator
from app.services.notification_service import NotificationService
from app.state_store import StateStore

logger = logging.getLogger("orchestrator")


def create_real_orchestrator(config: OrchestratorConfig) -> Orchestrator:
    """Create an orchestrator with real CLI adapters."""
    from app.adapters.claude_planner_cli import ClaudePlannerCli
    from app.adapters.qwen_worker_cli import QwenWorkerCli
    from app.adapters.lmstudio_worker_api import LmStudioWorkerApi

    planner_adapter = ClaudePlannerCli(
        cli_path=config.planner_adapter.cli_path,
        model=config.planner_adapter.model,
        extra_flags=config.planner_adapter.extra_flags,
    )

    # Select worker adapter by name
    wa = config.worker_adapter
    if wa.name == "lmstudio_worker_api":
        worker_adapter = LmStudioWorkerApi(
            base_url=wa.base_url or "http://localhost:1234",
            model=wa.model,
            api_key=wa.api_key or "lm-studio",
            temperature=wa.temperature,
            max_tokens=wa.max_tokens,
        )
    else:
        worker_adapter = QwenWorkerCli(
            cli_path=wa.cli_path,
            extra_flags=wa.extra_flags,
        )

    state_store = StateStore(config.state_path)
    notification_service = NotificationService(config.notifications)

    return Orchestrator(
        config=config,
        state_store=state_store,
        planner_adapter=planner_adapter,
        worker_adapter=worker_adapter,
        notification_service=notification_service,
    )


def run_demo() -> None:
    """Run with fake adapters for demonstration."""
    from app.adapters.fake_planner import FakePlanner
    from app.adapters.fake_worker import FakeWorker

    config = OrchestratorConfig(
        goal="Build a simple REST API with user authentication",
        poll_interval_seconds=1,
        max_empty_cycles=5,
    )

    planner_responses = [
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Create the project structure for a REST API with Flask.",
            "reason": "Starting with project scaffolding",
            "check_after_seconds": 1,
        },
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Add user model and authentication routes.",
            "reason": "Structure ready, implementing auth",
            "memory_update": "Project structure created",
            "check_after_seconds": 1,
        },
        {
            "decision": "launch_worker",
            "target_worker_id": "qwen-1",
            "task_instruction": "Add input validation and error handling.",
            "reason": "Auth done, adding safety",
            "memory_update": "Auth routes implemented",
            "check_after_seconds": 1,
        },
        {
            "decision": "finish",
            "reason": "All subtasks completed",
            "should_finish": True,
            "final_summary": "REST API with auth built successfully.",
        },
    ]

    worker_responses = [
        {"status": "success", "summary": "Created project structure", "artifacts": ["app.py"], "confidence": 0.95},
        {"status": "success", "summary": "Implemented User model and auth routes", "artifacts": ["app.py"], "confidence": 0.9},
        {"status": "success", "summary": "Added validation and error handlers", "artifacts": ["app.py"], "confidence": 0.85},
    ]

    planner = FakePlanner(responses=planner_responses, delay=0.01)
    worker = FakeWorker(responses=worker_responses, delay=0.01)

    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    state_store = StateStore(state_dir / "demo_state.json")

    orch = Orchestrator(
        config=config,
        state_store=state_store,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    logger.info("=== DEMO START ===")
    reason = orch.run()
    logger.info("=== DEMO END: %s ===", reason.value)


def main() -> None:
    """Main entry point."""
    if "--demo" in sys.argv:
        setup_logging(log_level="DEBUG", log_dir="logs", log_file="demo.log")
        run_demo()
        return

    # Production mode
    setup_logging(log_level="INFO")
    logger.info("Orchestrator starting...")

    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = load_config_from_dict(config_data)
        logger.info("Config loaded from config.json")
    else:
        config = OrchestratorConfig()
        logger.info("Using default config (no config.json found)")

    orch = create_real_orchestrator(config)

    # Load research context if MCP integration is configured
    if config.research_config:
        orch.load_research_context()

    reason = orch.run()
    logger.info("Orchestrator stopped: %s", reason.value)


if __name__ == "__main__":
    main()
