"""
Demo: LM Studio worker adapter with a fake LM Studio server.

This demonstrates the LM Studio adapter working without a real
LM Studio instance, by using a mock HTTP server.

Run: python -m examples.demo_lmstudio
"""

import json
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
from app.adapters.fake_planner import FakePlanner
from app.config import OrchestratorConfig
from app.logging_setup import setup_logging
from app.orchestrator import Orchestrator
from app.state_store import StateStore


# --- Mock LM Studio server ---

class MockLmStudio(BaseHTTPRequestHandler):
    """Simulates LM Studio /v1/chat/completions."""

    _call_count = 0

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "mock-model", "object": "model"}]})
            self._respond(200, body)
        else:
            self._respond(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            MockLmStudio._call_count += 1
            n = MockLmStudio._call_count

            # Simulate different responses per call
            if n == 1:
                content = json.dumps({
                    "status": "success",
                    "what_was_requested": "Create project structure and core logic",
                    "what_was_done": "Created the initial project structure for the utility library.",
                    "results_table": [{"artifact": "main.py", "status": "created"}],
                    "key_metrics": {"files_created": 2},
                    "artifacts": ["main.py", "config.py"],
                    "verdict": "WATCHLIST",
                    "confidence": 0.9,
                    "error": "",
                    "mcp_problems": [],
                })
            elif n == 2:
                content = json.dumps({
                    "status": "success",
                    "what_was_requested": "Implement core logic",
                    "what_was_done": "Implemented the core utility logic with basic error handling.",
                    "results_table": [{"artifact": "core.py", "status": "created"}],
                    "key_metrics": {"modules_added": 1},
                    "artifacts": ["core.py"],
                    "verdict": "PROMOTE",
                    "confidence": 0.85,
                    "error": "",
                    "mcp_problems": [],
                })
            else:
                content = json.dumps({
                    "status": "success",
                    "what_was_requested": "Add tests and documentation",
                    "what_was_done": "Added tests and documented the delivered components in the final report.",
                    "results_table": [{"artifact": "test_main.py", "status": "created"}],
                    "key_metrics": {"tests_added": 1},
                    "artifacts": ["test_main.py"],
                    "verdict": "PROMOTE",
                    "confidence": 0.8,
                    "error": "",
                    "mcp_problems": [],
                })

            resp = json.dumps({
                "choices": [{
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
            })
            self._respond(200, resp)
        else:
            self._respond(404, '{"error":"not found"}')

    def _respond(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *a):
        pass


def main():
    logger = setup_logging(log_level="DEBUG", log_dir="logs", log_file="demo_lmstudio.log")

    # Start mock server
    server = HTTPServer(("127.0.0.1", 0), MockLmStudio)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base_url = f"http://127.0.0.1:{port}"
    logger.info("Mock LM Studio server on %s", base_url)

    # Configure orchestrator with LM Studio worker
    config = OrchestratorConfig(
        goal="Build a Python utility library",
        poll_interval_seconds=0,
        max_empty_cycles=10,
    )

    # LM Studio worker adapter pointing to mock server
    worker = LmStudioWorkerApi(
        base_url=base_url,
        model="mock-model",
        temperature=0.5,
        max_tokens=2048,
    )

    # Planner scripts the workflow
    planner = FakePlanner(responses=[
        """# Plan v1

## Status and Frame
Mock LM Studio demo baseline.

## Goal
Create the project skeleton and core logic.

## Baseline
Start from an empty utility library skeleton.

## Research Principles
- Keep the plan concrete and sequential.

## dev_space1 Capabilities
Workers available: 1.

## ETAP 1: Structure
Goal: create the project structure.
1. Create the initial library files.
Completion criteria: project skeleton exists.
| artifact | status |
| --- | --- |
| structure | pending |

## ETAP 2: Core Logic
Goal: implement the core module.
1. Add the main utility logic with error handling.
Completion criteria: core module exists.
| artifact | status |
| --- | --- |
| core | pending |

## ETAP 3: Summary
Goal: summarize the implementation.
1. Return a worker report with artifacts.
Completion criteria: structured report emitted.
| artifact | status |
| --- | --- |
| summary | pending |
""",
        """# Plan v2

## Status and Frame
Continue from the existing library implementation.

## Goal
Add tests and final validation.

## Baseline
Do not rewrite completed modules.

## Research Principles
- Extend coverage cleanly.

## dev_space1 Capabilities
Workers available: 1.

## ETAP 1: Tests
Goal: add automated coverage.
1. Create tests for the main utility flow.
Completion criteria: test file exists.
| artifact | status |
| --- | --- |
| tests | pending |

## ETAP 2: Documentation
Goal: summarize behavior.
1. Document the delivered components in the worker report.
Completion criteria: report includes tests and docs artifacts.
| artifact | status |
| --- | --- |
| docs | pending |

## ETAP 3: Summary
Goal: finalize output.
1. Return the structured worker report.
Completion criteria: verdict recorded.
| artifact | status |
| --- | --- |
| summary | pending |
""",
    ])

    tmp = Path(tempfile.mkdtemp())
    store = StateStore(tmp / "state.json")

    orch = Orchestrator(
        config=config,
        state_store=store,
        planner_adapter=planner,
        worker_adapter=worker,
    )

    logger.info("=== DEMO: LM Studio Worker ===")
    reason = orch.run()

    print("\n--- LM Studio Worker Demo Results ---")
    for task in orch.state.tasks:
        print(f"  [{task.status.value}] {task.description[:70]}")
    for r in orch.state.results:
        print(f"  [{r.status}] {r.summary[:70]}")
    print(f"\nStop reason: {reason.value}")
    print(f"LM Studio API calls: {MockLmStudio._call_count}")

    server.shutdown()


if __name__ == "__main__":
    main()
