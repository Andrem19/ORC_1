from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from app.compiled_plan_store import CompiledPlanStore
from app.config import OrchestratorConfig, load_config_from_dict
from app.logging_setup import setup_logging
from app.raw_plan_converter_service import RawPlanConverterService
from app.raw_plan_ordering import raw_plan_sort_key
from app.raw_plan_semantic_service import RawPlanSemanticService
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


async def _fetch_mcp_tool_catalog(config: OrchestratorConfig) -> list:
    from app.services.direct_execution.mcp_client import DirectMcpClient, DirectMcpConfig

    direct = getattr(config, "direct_execution", None)
    if not direct:
        return []
    endpoint = str(getattr(direct, "mcp_endpoint_url", "") or "").strip()
    if not endpoint:
        return []
    mcp_config = DirectMcpConfig(
        endpoint_url=endpoint,
        auth_mode=str(getattr(direct, "mcp_auth_mode", "none") or "none"),
        token_env_var=str(getattr(direct, "mcp_token_env_var", "DEV_SPACE1_MCP_BEARER_TOKEN") or ""),
        connect_timeout_seconds=float(getattr(direct, "connect_timeout_seconds", 10) or 10),
        read_timeout_seconds=float(getattr(direct, "read_timeout_seconds", 60) or 60),
        retry_budget=int(getattr(direct, "retry_budget", 1) or 1),
    )
    client = DirectMcpClient(mcp_config)
    try:
        tools = await client.list_tools()
        await client.close()
        print(f"MCP tool catalog fetched: {len(tools)} tools from {endpoint}")
        return tools
    except Exception as exc:
        print(f"Warning: could not fetch MCP tool catalog: {exc}", file=sys.stderr)
        try:
            await client.close()
        except Exception:
            pass
        return []


async def _run() -> int:
    config = _load_config()
    raw_dir = Path(config.raw_plan_dir)
    store = CompiledPlanStore(config.compiled_plan_dir)
    raw_files = sorted(raw_dir.glob("*.md"), key=raw_plan_sort_key)
    if not raw_files:
        print(f"No raw plans found in {raw_dir}")
        return 1

    semantic_service = None
    if config.converter_use_llm:
        planner_adapter = create_planner_adapter(config)
        if not planner_adapter.is_available():
            print(
                "Planner adapter is unavailable but converter_use_llm=true. "
                "Install/configure the planner adapter or disable converter_use_llm in config.toml.",
                file=sys.stderr,
            )
            return 2
        semantic_service = RawPlanSemanticService(
            adapter=planner_adapter,
            timeout_seconds=config.planner_decision_timeout_seconds,
            retry_attempts=config.planner_decision_retry_attempts,
            retry_backoff_seconds=config.decision_retry_backoff_seconds,
        )

    mcp_tool_catalog = await _fetch_mcp_tool_catalog(config)

    converter = RawPlanConverterService(
        semantic_service=semantic_service,
        use_llm=bool(config.converter_use_llm),
        mcp_tool_catalog=mcp_tool_catalog,
    )
    compiled = 0
    failed = 0
    print(f"Converting {len(raw_files)} raw plan(s) from {raw_dir} into {store.root_dir}")
    for raw_file in raw_files:
        print(f"[START] {raw_file.name}")
        try:
            sequence = await converter.convert_path(raw_file)
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {raw_file.name}: {exc}")
            continue
        manifest_path = store.save_sequence(sequence)
        if sequence.report.compile_status == "compiled":
            compiled += 1
            print(
                f"[OK] {raw_file.name}: {sequence.report.stage_count} stage(s), "
                f"{sequence.report.compiled_plan_count} plan batch(es) -> {manifest_path}"
            )
        else:
            failed += 1
            errors = ", ".join(sequence.report.errors) or "unknown_error"
            print(f"[FAIL] {raw_file.name}: {errors} -> {manifest_path}")
    print(
        f"Conversion complete: total={len(raw_files)} compiled={compiled} failed={failed} "
        f"output_dir={store.root_dir}"
    )
    return 0 if compiled else 1


def main() -> None:
    setup_logging(log_level="INFO", log_dir="logs", log_file="converter.log", rich_console=False)
    try:
        rc = asyncio.run(_run())
    except KeyboardInterrupt:
        print("Conversion interrupted by user.")
        raise SystemExit(130)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
