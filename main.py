"""
Entry point for the orchestrator.

Usage:
    python main.py              # Production mode (reads config.toml)
    python main.py --demo       # Demo mode with fake adapters
"""

from __future__ import annotations

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
        mode=config.planner_adapter.mode,
        use_bare=config.planner_adapter.use_bare,
        no_session_persistence=config.planner_adapter.no_session_persistence,
        capture_stderr_live=config.planner_adapter.capture_stderr_live,
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

    # Check adapter availability before starting
    planner_ok = planner_adapter.is_available()
    worker_ok = worker_adapter.is_available()
    if not planner_ok:
        logger.error(
            "Planner adapter '%s' is NOT available (cli_path='%s')",
            planner_adapter.name(), config.planner_adapter.cli_path,
        )
    else:
        logger.info("Planner adapter '%s' available (cli_path='%s')", planner_adapter.name(), config.planner_adapter.cli_path)
        runtime = planner_adapter.runtime_summary()
        logger.info(
            "Planner runtime: mode=%s bare=%s no_session_persistence=%s resolved_model=%s",
            runtime.get("mode"),
            runtime.get("use_bare"),
            runtime.get("no_session_persistence"),
            runtime.get("resolved_model"),
        )
        if runtime.get("has_custom_backend") or runtime.get("has_model_remap"):
            logger.warning(
                "Planner is using custom backend/model mapping: base_url=%s configured_model=%s resolved_model=%s",
                runtime.get("custom_base_url") or "<default>",
                runtime.get("configured_model"),
                runtime.get("resolved_model"),
            )

    if not worker_ok:
        logger.error(
            "Worker adapter '%s' is NOT available (cli_path='%s')",
            worker_adapter.name(), config.worker_adapter.cli_path or config.worker_adapter.base_url,
        )
    else:
        logger.info("Worker adapter '%s' available (cli_path='%s')", worker_adapter.name(), config.worker_adapter.cli_path or config.worker_adapter.base_url)

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
        setup_logging(log_level="DEBUG", log_dir="logs", log_file="demo.log", rich_console=True)
        # Start Rich progress display for demo
        if sys.stdout.isatty():
            from app.rich_handler import ProgressManager
            ProgressManager.get().start()
        try:
            run_demo()
        finally:
            if sys.stdout.isatty():
                from app.rich_handler import ProgressManager
                ProgressManager.get().stop()
        return

    # Production mode: initial logging before config is loaded
    setup_logging(log_level="INFO")
    logger.info("Orchestrator starting...")

    # Load config
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config.toml"
    if config_path.exists():
        with open(config_path, "rb") as f:
            config_data = tomllib.load(f)
        config = load_config_from_dict(config_data)
        logger.info("Config loaded from config.toml")
    else:
        config = OrchestratorConfig()
        logger.info("Using default config (no config.toml found)")

    # Reconfigure logging with config values (log_level, log_dir, log_file)
    setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir,
        log_file=config.log_file,
        rich_console=config.rich_console,
        truncate_length=config.console_truncate_length,
    )
    logger.info("Logging configured: level=%s, dir=%s, file=%s", config.log_level, config.log_dir, config.log_file)
    resolved_log_path = getattr(logger, "orchestrator_log_path", "")
    if resolved_log_path:
        logger.info("Log file path: %s", resolved_log_path)
        if not Path(resolved_log_path).exists():
            logger.error("Configured log file was not created: %s", resolved_log_path)

    # Start Rich progress display if enabled
    if config.rich_console and sys.stdout.isatty():
        from app.rich_handler import ProgressManager
        ProgressManager.get().start()

    # Acquire PID lock — prevent concurrent instances
    from app.pid_lock import PidLock

    pid_lock = PidLock(Path(config.state_dir) / "orchestrator.pid")
    if not pid_lock.acquire():
        logger.critical("Aborting: another orchestrator instance holds the lock")
        sys.exit(1)

    try:
        # Handle startup_mode: reset state before creating orchestrator
        if config.startup_mode in ("reset", "reset_all"):
            from app.reset_manager import ResetManager
            from app.plan_store import PlanStore
            from app.state_store import StateStore

            logger.info("startup_mode=%s — performing reset", config.startup_mode)
            plan_store = PlanStore(config.plan_dir) if config.plan_mode else None
            state_store = StateStore(config.state_path)
            ResetManager(config.state_dir, state_store, plan_store).perform_reset(config.startup_mode)
            config.startup_mode = "resume"  # in-memory revert to prevent double-reset
            logger.info("Reset done — orchestrator will start from clean state")

        orch = create_real_orchestrator(config)

        # Ensure plan directory exists for plan mode
        if config.plan_mode:
            Path(config.plan_dir).mkdir(parents=True, exist_ok=True)
            logger.info("Plan mode enabled, plans directory: %s", config.plan_dir)

        # Restore state from previous run if available
        orch.load_state()

        # Load translation model at startup if enabled
        if config.notifications.translate_to_russian:
            logger.info("Loading translation model...")
            try:
                orch.notification_service.init_translation()
            except RuntimeError as e:
                logger.critical("Translation model failed to load: %s", e)
                sys.exit(1)

        # Load research context if MCP integration is configured
        if config.research_config:
            orch.load_research_context()

        # Graceful shutdown on SIGTERM / SIGINT (3 phases)
        import signal as _signal

        _stop_phase = 0  # 0=running, 1=drain, 2=immediate stop, 3+=KeyboardInterrupt

        def _signal_handler(signum: int, _frame: Any) -> None:
            nonlocal _stop_phase
            sig_name = _signal.Signals(signum).name
            if _stop_phase == 0:
                # First press — enter drain mode (let running tasks finish)
                logger.warning(
                    "Received %s, entering drain mode (waiting for running tasks to finish)...",
                    sig_name,
                )
                logger.warning("Press Ctrl+C again to force immediate stop.")
                orch.request_drain()
                from app.scheduler import Scheduler
                Scheduler._wake.set()
                _stop_phase = 1
            elif _stop_phase == 1:
                # Second press — force immediate stop
                logger.warning("Received second %s, forcing immediate shutdown!", sig_name)
                orch.request_stop()
                from app.scheduler import Scheduler
                Scheduler._wake.set()
                _stop_phase = 2
            else:
                # Third+ press — hard interrupt
                logger.warning("Received third %s, raising KeyboardInterrupt!", sig_name)
                raise KeyboardInterrupt

        _signal.signal(_signal.SIGTERM, _signal_handler)
        # SIGINT: first press = graceful, second press = hard stop
        _signal.signal(_signal.SIGINT, _signal_handler)

        try:
            reason = orch.run()
            logger.info("Orchestrator stopped: %s", reason.value)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user — shutting down immediately")
            orch._finish(StopReason.NO_PROGRESS, "Interrupted by user (Ctrl+C)")
    finally:
        # Stop Rich progress display
        if config.rich_console and sys.stdout.isatty():
            from app.rich_handler import ProgressManager
            ProgressManager.get().stop()
        pid_lock.release()
        logging.shutdown()


if __name__ == "__main__":
    main()
