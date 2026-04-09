from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.adapters.fake_planner import FakePlanner
from app.adapters.fake_worker import FakeWorker
from app.config import LMStudioConfig, OrchestratorConfig
from app.orchestrator import Orchestrator
from main import (
    _apply_terminal_safety_defaults,
    _log_plan_source_startup,
    _print_orchestrator_status,
    _reset_logging_before_reconfigure,
    _should_force_detach,
    _stop_running_orchestrator,
    _validate_runtime_startup,
    build_live_validation_config,
    create_real_orchestrator,
)


def _make_orch(*, planner_ok: bool = True, worker_ok: bool = True):
    planner_adapter = MagicMock()
    planner_adapter.is_available.return_value = planner_ok
    worker_adapter = MagicMock()
    worker_adapter.is_available.return_value = worker_ok
    return SimpleNamespace(
        config=SimpleNamespace(plan_source="planner"),
        planner_adapter=planner_adapter,
        worker_adapter=worker_adapter,
    )


def test_validate_runtime_startup_accepts_available_adapters():
    orch = _make_orch(planner_ok=True, worker_ok=True)

    assert _validate_runtime_startup(orch) == ""


def test_validate_runtime_startup_reports_missing_worker():
    orch = _make_orch(planner_ok=True, worker_ok=False)

    message = _validate_runtime_startup(orch)

    assert "worker CLI" in message
    assert "Planner-worker runtime requires available adapters before startup" in message


def test_validate_runtime_startup_accepts_missing_planner_in_compiled_raw_mode():
    orch = _make_orch(planner_ok=False, worker_ok=True)
    orch.config.plan_source = "compiled_raw"

    assert _validate_runtime_startup(orch) == ""


def test_log_plan_source_startup_warns_when_compiled_manifests_exist_but_planner_mode(tmp_path):
    cfg = OrchestratorConfig(
        goal="test",
        plan_source="planner",
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
    )
    (tmp_path / "raw_plans").mkdir()
    compiled_dir = tmp_path / "compiled_plans" / "plan_v1"
    compiled_dir.mkdir(parents=True)
    (compiled_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with patch("main.logger") as mock_logger:
        _log_plan_source_startup(cfg)

    info_messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
    warning_messages = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
    assert any("Runtime plan source: %s" in message for message in info_messages)
    assert any("runtime will invoke the planner CLI" in message for message in warning_messages)


def test_log_plan_source_startup_warns_when_compiled_raw_has_no_manifests(tmp_path):
    cfg = OrchestratorConfig(
        goal="test",
        plan_source="compiled_raw",
        raw_plan_dir=str(tmp_path / "raw_plans"),
        compiled_plan_dir=str(tmp_path / "compiled_plans"),
    )
    (tmp_path / "raw_plans").mkdir()
    (tmp_path / "raw_plans" / "plan_v1.md").write_text("# Plan", encoding="utf-8")

    with patch("main.logger") as mock_logger:
        _log_plan_source_startup(cfg)

    info_messages = [str(call.args[0]) for call in mock_logger.info.call_args_list]
    warning_messages = [str(call.args[0]) for call in mock_logger.warning.call_args_list]
    assert any("Runtime plan source: %s" in message for message in info_messages)
    assert any("Run `python converter.py` first." in message for message in warning_messages)


def test_orchestrator_initializes_without_legacy_plan_flags(tmp_path):
    cfg = OrchestratorConfig(
        goal="test",
        plan_dir=str(tmp_path / "plans"),
    )

    orch = Orchestrator(
        config=cfg,
        planner_adapter=FakePlanner(responses=[]),
        worker_adapter=FakeWorker(responses=[]),
    )

    assert orch._plan_service is None
    assert orch.planner_adapter is not None
    assert orch.worker_adapter is not None
    assert not hasattr(orch, "planner_service")
    assert not hasattr(orch, "worker_service")


def test_create_real_orchestrator_passes_shared_lmstudio_settings_to_worker():
    cfg = OrchestratorConfig(
        goal="test",
        lmstudio=LMStudioConfig(
            base_url="http://127.0.0.1:1234",
            model="qwen/qwen3.5-9b",
            reasoning_effort="none",
        ),
    )
    cfg.worker_adapter.name = "lmstudio_worker_api"
    cfg.worker_adapter.base_url = ""
    cfg.worker_adapter.model = ""

    planner_instance = MagicMock()
    planner_instance.is_available.return_value = True
    planner_instance.name.return_value = "claude_planner_cli"
    planner_instance.runtime_summary.return_value = {
        "mode": "stream-json",
        "use_bare": False,
        "no_session_persistence": True,
        "resolved_model": "glm-5.1",
        "has_custom_backend": False,
        "has_model_remap": False,
        "configured_model": "opus",
        "custom_base_url": "",
    }
    worker_instance = MagicMock()
    worker_instance.is_available.return_value = True
    worker_instance.name.return_value = "lmstudio_worker_api"

    with (
        patch("app.adapters.claude_planner_cli.ClaudePlannerCli", return_value=planner_instance),
        patch("app.adapters.lmstudio_worker_api.LmStudioWorkerApi", return_value=worker_instance) as lm_cls,
        patch("main.NotificationService"),
        patch("main.Orchestrator", return_value=MagicMock()),
    ):
        create_real_orchestrator(cfg)

    lm_cls.assert_called_once()
    kwargs = lm_cls.call_args.kwargs
    assert kwargs["base_url"] == "http://127.0.0.1:1234"
    assert kwargs["model"] == "qwen/qwen3.5-9b"
    assert kwargs["reasoning_effort"] == "none"


def test_build_live_validation_config_isolates_runtime_dirs(tmp_path):
    cfg = OrchestratorConfig(
        goal="test",
        state_dir=str(tmp_path / "state"),
        plan_dir=str(tmp_path / "plans"),
        poll_interval_seconds=120,
    )

    live = build_live_validation_config(cfg)

    assert live.state_dir.endswith("live_validation")
    assert live.plan_dir.endswith("live_validation")
    assert live.log_file == "live_validation.log"
    assert live.startup_mode == "reset_all"
    assert live.rich_console is False
    assert live.console_log_level == "WARNING"
    assert live.poll_interval_seconds == 10


def test_apply_terminal_safety_defaults_for_vscode(monkeypatch):
    cfg = OrchestratorConfig(goal="test", rich_console=True, console_log_level="INFO", console_truncate_length=300)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    updated, notes = _apply_terminal_safety_defaults(cfg, live_validate=False, detach=False)

    assert updated.rich_console is True
    assert updated.console_log_level == "INFO"
    assert updated.console_truncate_length == 160
    assert "reduced_console_truncate_length" in notes


def test_apply_terminal_safety_defaults_keeps_detached_run_unchanged(monkeypatch):
    cfg = OrchestratorConfig(goal="test", rich_console=True, console_log_level="INFO", console_truncate_length=300)
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    updated, notes = _apply_terminal_safety_defaults(cfg, live_validate=False, detach=True)

    assert updated.rich_console is True
    assert updated.console_log_level == "INFO"
    assert notes == []


def test_should_force_detach_in_vscode_for_multi_worker(monkeypatch):
    cfg = OrchestratorConfig(goal="test")
    cfg.workers = [cfg.workers[0], cfg.workers[0]]
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert _should_force_detach(cfg, live_validate=False, detach=False) is False


def test_should_force_detach_skips_single_worker_normal_run(monkeypatch):
    cfg = OrchestratorConfig(goal="test")
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert _should_force_detach(cfg, live_validate=False, detach=False) is False


def test_should_force_detach_skips_when_already_detached(monkeypatch):
    cfg = OrchestratorConfig(goal="test")
    cfg.workers = [cfg.workers[0], cfg.workers[0]]
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert _should_force_detach(cfg, live_validate=False, detach=True) is False


def test_should_force_detach_skips_for_detached_child(monkeypatch):
    cfg = OrchestratorConfig(goal="test")
    cfg.workers = [cfg.workers[0], cfg.workers[0]]
    monkeypatch.setenv("TERM_PROGRAM", "vscode")

    assert _should_force_detach(cfg, live_validate=False, detach=False) is False


def test_stop_running_orchestrator_returns_zero_when_no_pid(tmp_path, capsys):
    cfg = OrchestratorConfig(goal="test", state_dir=str(tmp_path / "state"))

    rc = _stop_running_orchestrator(cfg)

    assert rc == 0
    assert "No running orchestrator PID file found" in capsys.readouterr().out


def test_print_orchestrator_status_without_pid(tmp_path, capsys):
    cfg = OrchestratorConfig(
        goal="test",
        state_dir=str(tmp_path / "state"),
        log_dir=str(tmp_path / "logs"),
    )

    rc = _print_orchestrator_status(cfg)

    out = capsys.readouterr().out
    assert rc == 0
    assert "profile: default" in out
    assert "running: no" in out
    assert "pid: none" in out


def test_print_orchestrator_status_live_profile(tmp_path, capsys):
    cfg = OrchestratorConfig(
        goal="test",
        state_dir=str(tmp_path / "state" / "live_validation"),
        log_dir=str(tmp_path / "logs"),
        log_file="live_validation.log",
    )

    rc = _print_orchestrator_status(cfg)

    out = capsys.readouterr().out
    assert rc == 0
    assert "profile: live_validation" in out
    assert "log_path:" in out


def test_reset_logging_before_reconfigure_clears_existing_logs(tmp_path):
    log_dir = tmp_path / "logs"
    run_dir = log_dir / "runs" / "run-123"
    run_dir.mkdir(parents=True)
    (log_dir / "orchestrator.log").write_text("root", encoding="utf-8")
    (run_dir / "orchestrator.log").write_text("nested", encoding="utf-8")
    (run_dir / "orchestrator.events.jsonl").write_text("{}", encoding="utf-8")

    _reset_logging_before_reconfigure(str(log_dir))

    assert log_dir.exists()
    assert list(log_dir.iterdir()) == []
