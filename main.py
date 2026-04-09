"""
Entry point for the orchestrator.

Usage:
    python main.py              # Production mode (reads config.toml)
    python main.py --demo       # Demo mode with fake adapters
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

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
from app.execution_store import ExecutionStateStore
from app.logging_setup import setup_logging
from app.logging_runtime import clear_log_root
from app.models import StopReason
from app.orchestrator import Orchestrator
from app.runtime_factory import create_planner_adapter, create_worker_adapter
from app.run_context import ensure_current_run, generate_run_id, read_current_run_id
from app.services.notification_service import NotificationService

logger = logging.getLogger("orchestrator")


def build_live_validation_config(config: OrchestratorConfig) -> OrchestratorConfig:
    """Return an isolated config profile for safe live validation runs."""
    live = load_config_from_dict(config.to_dict())
    live.state_dir = str(Path(config.state_dir) / "live_validation")
    live.plan_dir = str(Path(config.plan_dir) / "live_validation")
    live.log_file = "live_validation.log"
    live.startup_mode = "reset_all"
    live.rich_console = False
    live.console_log_level = "WARNING"
    live.poll_interval_seconds = min(int(config.poll_interval_seconds or 10), 10)
    return live


def _is_vscode_terminal() -> bool:
    term_program = str(os.environ.get("TERM_PROGRAM", "")).lower()
    if term_program == "vscode":
        return True
    return any(
        key in os.environ
        for key in ("VSCODE_PID", "VSCODE_IPC_HOOK_CLI", "VSCODE_GIT_IPC_HANDLE")
    )


def _apply_terminal_safety_defaults(config: OrchestratorConfig, *, live_validate: bool, detach: bool) -> tuple[OrchestratorConfig, list[str]]:
    notes: list[str] = []
    if _is_vscode_terminal() and not detach:
        if int(config.console_truncate_length or 300) > 160:
            config.console_truncate_length = 160
            notes.append("reduced_console_truncate_length")
    if live_validate and not detach and str(config.console_log_level or "INFO").upper() != "WARNING":
        config.console_log_level = "WARNING"
        notes.append("raised_console_log_level_for_live_validation")
    return config, notes


def _should_force_detach(config: OrchestratorConfig, *, live_validate: bool, detach: bool) -> bool:
    del config, live_validate, detach
    return False


def _resolve_run_id(config: OrchestratorConfig) -> str:
    if config.current_run_id:
        return config.current_run_id
    if config.startup_mode == "resume":
        existing = read_current_run_id(config.state_dir) or read_current_run_id(config.plan_dir) or read_current_run_id(config.log_dir)
        if existing:
            config.current_run_id = existing
            return existing
    run_id = generate_run_id()
    config.current_run_id = run_id
    for root in (config.state_dir, config.plan_dir, config.log_dir):
        ensure_current_run(root, run_id)
    return run_id


def _reset_logging_before_reconfigure(log_dir: str) -> None:
    root_logger = logging.getLogger("orchestrator")
    for handler in list(root_logger.handlers):
        try:
            handler.flush()
            handler.close()
        finally:
            root_logger.removeHandler(handler)
    logging.shutdown()
    clear_log_root(log_dir)


def _stop_running_orchestrator(config: OrchestratorConfig) -> int:
    pid_path = Path(config.state_dir) / "orchestrator.pid"
    if not pid_path.exists():
        print(f"No running orchestrator PID file found at {pid_path}")
        return 0
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        print(f"Could not parse PID file: {pid_path}")
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"No live process for PID {pid}; stale PID file was left at {pid_path}")
        return 0
    except OSError as exc:
        print(f"Failed to stop PID {pid}: {exc}")
        return 1

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"Stopped orchestrator PID {pid}")
            return 0
        except OSError:
            print(f"Stopped orchestrator PID {pid}")
            return 0
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        print(f"Stopped orchestrator PID {pid}")
        return 0
    except OSError as exc:
        print(f"Failed to force-stop PID {pid}: {exc}")
        return 1
    print(f"Force-stopped orchestrator PID {pid}")
    return 0


def _print_orchestrator_status(config: OrchestratorConfig) -> int:
    pid_path = Path(config.state_dir) / "orchestrator.pid"
    state_path = Path(config.execution_state_path)
    log_path = Path(config.log_dir) / config.log_file

    pid_text = ""
    pid: int | None = None
    if pid_path.exists():
        pid_text = pid_path.read_text(encoding="utf-8").strip()
        try:
            pid = int(pid_text)
        except ValueError:
            pid = None

    running = False
    if pid is not None:
        try:
            os.kill(pid, 0)
            running = True
        except ProcessLookupError:
            running = False
        except OSError:
            running = False

    profile = "live_validation" if str(config.state_dir).endswith("live_validation") else "default"
    print(f"profile: {profile}")
    print(f"running: {'yes' if running else 'no'}")
    print(f"pid_file: {pid_path}")
    print(f"pid: {pid if pid is not None else (pid_text or 'none')}")
    print(f"state_path: {state_path}")
    print(f"log_path: {log_path}")
    return 0


def _validate_runtime_startup(orch: Orchestrator) -> str:
    """Return a startup error when required planner/worker adapters are unavailable."""
    planner_adapter = getattr(orch, "planner_adapter", None)
    worker_adapter = getattr(orch, "worker_adapter", None)
    planner_available = bool(planner_adapter and planner_adapter.is_available())
    worker_available = bool(worker_adapter and worker_adapter.is_available())
    if orch.config.plan_source == "compiled_raw":
        return "" if worker_available else "Plannerless runtime requires available worker CLI before startup: worker CLI."
    if planner_available and worker_available:
        return ""
    missing: list[str] = []
    if not planner_available:
        missing.append("planner CLI")
    if not worker_available:
        missing.append("worker CLI")
    return f"Planner-worker runtime requires available adapters before startup: {', '.join(missing)}."


def _log_plan_source_startup(config: OrchestratorConfig) -> None:
    """Log the selected runtime plan source and obvious mismatches."""
    logger.info(
        "Runtime plan source: %s (raw_plan_dir=%s, compiled_plan_dir=%s)",
        config.plan_source,
        config.raw_plan_dir,
        config.compiled_plan_dir,
    )
    compiled_root = Path(config.compiled_plan_dir)
    raw_root = Path(config.raw_plan_dir)
    compiled_manifests = list(compiled_root.glob("*/manifest.json")) if compiled_root.exists() else []
    raw_files = list(raw_root.glob("*.md")) if raw_root.exists() else []
    if config.plan_source == "planner" and compiled_manifests:
        logger.warning(
            "Compiled plan artifacts exist (%d manifests), but plan_source=planner so runtime will invoke the planner CLI.",
            len(compiled_manifests),
        )
    if config.plan_source == "compiled_raw":
        if not raw_files:
            logger.warning("plan_source=compiled_raw but no raw plan files were found in %s", raw_root)
        if not compiled_manifests:
            logger.warning(
                "plan_source=compiled_raw but no compiled manifests were found in %s. Run `python converter.py` first.",
                compiled_root,
            )


def create_real_orchestrator(config: OrchestratorConfig) -> Orchestrator:
    """Create an orchestrator with real CLI adapters."""
    planner_adapter = create_planner_adapter(config)
    worker_adapter = create_worker_adapter(config)

    # Check adapter availability before starting
    planner_ok = planner_adapter.is_available()
    worker_ok = worker_adapter.is_available()
    if config.plan_source == "planner" and not planner_ok:
        logger.error(
            "Planner adapter '%s' is NOT available (cli_path='%s')",
            planner_adapter.name(), config.planner_adapter.cli_path,
        )
    elif planner_ok:
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
    else:
        logger.warning(
            "Planner adapter '%s' is unavailable, but plan_source=%s so startup may still continue.",
            planner_adapter.name(),
            config.plan_source,
        )

    if not worker_ok:
        logger.error(
            "Worker adapter '%s' is NOT available (cli_path='%s')",
            worker_adapter.name(), config.worker_adapter.cli_path or config.worker_adapter.base_url,
        )
    else:
        logger.info("Worker adapter '%s' available (cli_path='%s')", worker_adapter.name(), config.worker_adapter.cli_path or config.worker_adapter.base_url)

    notification_service = NotificationService(
        config.notifications, lmstudio_config=config.lmstudio,
    )

    console_controller = None
    if config.rich_console:
        from app.rich_handler import ProgressManager

        console_controller = ProgressManager.get().controller

    return Orchestrator(
        config=config,
        planner_adapter=planner_adapter,
        worker_adapter=worker_adapter,
        notification_service=notification_service,
        console_controller=console_controller,
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
            "plan_id": "plan_demo",
            "goal": "Build a REST API",
            "baseline_ref": {
                "snapshot_id": "active-signal-v1",
                "version": 1,
                "symbol": "BTCUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
            },
            "global_constraints": ["produce one bounded demo slice"],
            "slices": [
                {
                    "slice_id": "slice_demo",
                    "title": "Demo final report",
                    "hypothesis": "the fake worker can close a slice",
                    "objective": "emit one final report action",
                    "success_criteria": ["one final report returned"],
                    "allowed_tools": ["system_health"],
                    "evidence_requirements": ["demo completion summary"],
                    "policy_tags": ["demo"],
                    "max_turns": 1,
                    "max_tool_calls": 0,
                    "max_expensive_calls": 0,
                    "parallel_slot": 1,
                }
            ],
        }
    ]

    worker_responses = [
        {
            "type": "final_report",
            "summary": "Created project structure, auth routes, and validation handlers.",
            "facts": {"files": 1},
            "artifacts": ["app.py"],
            "key_metrics": {"files": 1},
            "verdict": "PROMOTE",
            "confidence": 0.95,
            "reportable_issues": [],
        },
    ]

    planner = FakePlanner(responses=planner_responses, delay=0.01)
    worker = FakeWorker(responses=worker_responses, delay=0.01)

    state_dir = Path("state")
    state_dir.mkdir(exist_ok=True)
    orch = Orchestrator(
        config=config,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    logger.info("=== DEMO START ===")
    reason = orch.run()
    logger.info("=== DEMO END: %s ===", reason.value)


def main() -> None:
    """Main entry point."""
    live_validate = "--live-validate" in sys.argv
    detach = "--detach" in sys.argv
    stop = "--stop" in sys.argv
    status = "--status" in sys.argv
    if "--demo" in sys.argv:
        setup_logging(log_level="DEBUG", log_dir="logs", log_file="demo.log", rich_console=True)
        # Start Rich progress display for demo
        if True:
            from app.rich_handler import ProgressManager

            ProgressManager.get().start()
        try:
            run_demo()
        finally:
            if True:
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

    if live_validate:
        config = build_live_validation_config(config)
    if stop:
        raise SystemExit(_stop_running_orchestrator(config))
    if status:
        raise SystemExit(_print_orchestrator_status(config))
    config, safety_notes = _apply_terminal_safety_defaults(config, live_validate=live_validate, detach=detach)

    _reset_logging_before_reconfigure(config.log_dir)
    run_id = _resolve_run_id(config)

    # Reconfigure logging with config values (log_level, log_dir, log_file)
    setup_logging(
        log_level=config.log_level,
        log_dir=config.log_dir,
        log_file=config.log_file,
        rich_console=config.rich_console,
        console_log_level=config.console_log_level,
        truncate_length=config.console_truncate_length,
        run_id=run_id,
    )
    logger.info("Logging configured: level=%s, dir=%s, file=%s, run_id=%s", config.log_level, config.log_dir, config.log_file, run_id)
    if safety_notes:
        logger.warning("Terminal safety mode applied: %s", ", ".join(safety_notes))
    resolved_log_path = getattr(logger, "orchestrator_log_path", "")
    if resolved_log_path:
        logger.info("Log file path: %s", resolved_log_path)
        if not Path(resolved_log_path).exists():
            logger.error("Configured log file was not created: %s", resolved_log_path)
    if detach:
        logger.warning("--detach is ignored; orchestrator now always runs in the foreground.")
    _log_plan_source_startup(config)

    # Start Rich progress display if enabled
    if config.rich_console:
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
            from app.plan_store import PlanStore
            from app.reset_manager import ResetManager
            from app.state_store import StateStore

            logger.info("startup_mode=%s — performing reset", config.startup_mode)
            plan_store = PlanStore(config.plan_dir)
            state_store = StateStore(config.state_path, run_id=config.current_run_id)
            ResetManager(config.state_dir, state_store, plan_store).perform_reset(config.startup_mode)
            ExecutionStateStore(config.execution_state_path, run_id=config.current_run_id).clear()
            config.startup_mode = "resume"  # in-memory revert to prevent double-reset
            logger.info("Reset done — orchestrator will start from clean state")

        orch = create_real_orchestrator(config)
        if live_validate:
            logger.warning("Running live validation profile with isolated dirs.")

        startup_issue = _validate_runtime_startup(orch)
        if startup_issue:
            orch.state.status = "finished"
            orch.state.stop_reason = StopReason.SUBPROCESS_ERROR
            orch.save_state()
            logger.error(startup_issue)
            logger.info("Orchestrator stopped: %s", StopReason.SUBPROCESS_ERROR.value)
            return

        Path(config.plan_dir).mkdir(parents=True, exist_ok=True)
        logger.info("Plans directory: %s", config.plan_dir)

        # Restore state from previous run if available
        orch.load_state()

        # Load research context if MCP integration is configured
        if config.research_config:
            orch.load_research_context()

        # Graceful shutdown on SIGTERM / SIGINT (3 phases)
        import signal as _signal
        import time as _time

        _stop_phase = 0  # 0=running, 1=drain, 2=immediate stop, 3+=KeyboardInterrupt
        _last_signal_time = 0.0  # debounce: suppress logs if < 2s apart

        def _signal_handler(signum: int, _frame: Any) -> None:
            nonlocal _stop_phase, _last_signal_time
            sig_name = _signal.Signals(signum).name
            now = _time.monotonic()
            suppressed = (now - _last_signal_time) < 2.0
            _last_signal_time = now
            if _stop_phase == 0:
                # First press — enter drain mode (let running tasks finish)
                # Transition phase BEFORE calling into orchestrator to prevent
                # re-entry from queued signals during network I/O.
                _stop_phase = 1
                logger.warning(
                    "Received %s, entering drain mode (waiting for running tasks to finish)...",
                    sig_name,
                )
                logger.warning("Press Ctrl+C again to force immediate stop.")
                orch.request_drain()
            elif _stop_phase == 1:
                # Second press — force immediate stop
                _stop_phase = 2
                if not suppressed:
                    logger.warning(
                        "Received second %s, forcing immediate shutdown!", sig_name,
                    )
                orch.request_stop()
            else:
                # Third+ press — hard interrupt
                if not suppressed:
                    logger.warning(
                        "Received third %s, raising KeyboardInterrupt!", sig_name,
                    )
                raise KeyboardInterrupt

        _signal.signal(_signal.SIGTERM, _signal_handler)
        # SIGINT: first press = graceful, second press = hard stop
        _signal.signal(_signal.SIGINT, _signal_handler)

        _finish_called = False
        try:
            reason = orch.run()
            logger.info("Orchestrator stopped: %s", reason.value)
        except KeyboardInterrupt:
            if not _finish_called:
                _finish_called = True
                logger.warning("Interrupted by user — shutting down immediately")
                orch._finish(StopReason.NO_PROGRESS, "Interrupted by user (Ctrl+C)")
    finally:
        # Stop Rich progress display
        if config.rich_console:
            from app.rich_handler import ProgressManager
            ProgressManager.get().stop()
        pid_lock.release()
        logging.shutdown()


if __name__ == "__main__":
    main()
