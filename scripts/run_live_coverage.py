#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from uuid import uuid4

import tomli

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config_from_dict
from app.live_coverage import LiveCoverageRunner, build_live_coverage_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic live MCP coverage through the broker.")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--config", default="config.toml")
    args = parser.parse_args()

    config = load_config_from_dict(tomli.load(open(args.config, "rb")))
    run_id = f"live_coverage_{uuid4().hex[:10]}"
    cycles = asyncio.run(LiveCoverageRunner(config=config, run_id=run_id).run_cycles(cycles=max(1, args.cycles)))
    summary = build_live_coverage_summary(run_id=run_id, cycles=cycles)
    output_dir = Path(config.plan_dir) / "runs" / run_id / "coverage"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, "summary_path": str(output_path), "all_ok": summary["all_ok"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
