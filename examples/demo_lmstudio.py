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
                    "summary": "Created project structure: main.py, config.py, utils.py",
                    "artifacts": ["main.py", "config.py"],
                    "confidence": 0.9,
                })
            elif n == 2:
                content = json.dumps({
                    "status": "success",
                    "summary": "Implemented core logic with error handling",
                    "artifacts": ["core.py"],
                    "confidence": 0.85,
                })
            else:
                content = json.dumps({
                    "status": "success",
                    "summary": "Added tests and documentation",
                    "artifacts": ["test_main.py"],
                    "confidence": 0.8,
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
        {"decision": "launch_worker", "target_worker_id": "lmstudio-1",
         "task_instruction": "Create project structure", "reason": "start",
         "check_after_seconds": 0},
        {"decision": "launch_worker", "target_worker_id": "lmstudio-1",
         "task_instruction": "Implement core logic", "reason": "structure done",
         "memory_update": "Project structure created",
         "check_after_seconds": 0},
        {"decision": "launch_worker", "target_worker_id": "lmstudio-1",
         "task_instruction": "Add tests", "reason": "core done",
         "memory_update": "Core logic implemented",
         "check_after_seconds": 0},
        {"decision": "finish", "should_finish": True,
         "final_summary": "Library built with LM Studio worker",
         "reason": "all done"},
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
