from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import OrchestratorConfig, load_config_from_dict
from app.execution_store import ExecutionStateStore
from app.logging_setup import setup_logging
from app.reporting import PostRunReportBuilder
from app.runtime_factory import create_planner_adapter


def _load_config() -> OrchestratorConfig:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib

    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        return OrchestratorConfig()
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    return load_config_from_dict(payload)


async def _run(args: argparse.Namespace) -> int:
    config = _load_config()
    if args.run_id:
        config.current_run_id = args.run_id
    store = ExecutionStateStore(config.execution_state_path, run_id=config.current_run_id)
    state = store.load()
    if state is None:
        print("Execution state not found for the requested run.")
        return 1
    planner_adapter = None if args.skip_llm else create_planner_adapter(config)
    builder = PostRunReportBuilder(
        config=config,
        planner_adapter=planner_adapter,
        run_id=config.current_run_id,
        skip_llm=bool(args.skip_llm),
    )
    result, _run_report = await builder.build(state=state)
    print(
        f"Reports rebuilt: plan_reports={result['plan_reports']} "
        f"sequence_reports={result['sequence_reports']} run_report={result['run_report_json']}"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild canonical reports for a finished run.")
    parser.add_argument("--run-id", default="", help="Explicit run id to rebuild")
    parser.add_argument("--skip-llm", action="store_true", help="Skip Russian narrative generation")
    parser.add_argument("--json-only", action="store_true", help="Accepted for compatibility; markdown is still rendered from JSON")
    parser.add_argument("--rebuild-all", action="store_true", help="Accepted for compatibility; all report layers are rebuilt")
    args = parser.parse_args()
    setup_logging(log_level="INFO", log_dir="logs", log_file="reporter.log", rich_console=False)
    raise SystemExit(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
