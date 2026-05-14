"""
Demo configuration example.

Shows how to set up a config for a real run.
"""

from app.config import OrchestratorConfig, AdapterConfig
from app.models import WorkerConfig

demo_config = OrchestratorConfig(
    goal="Build a REST API with user authentication using Flask",
    operator_directives=(
        "Highest priority: produce a concrete markdown plan immediately. "
        "Do not use tools or inspect the workspace while planning."
    ),
    planner_prompt_template="""You are writing a markdown execution plan.
Use only the facts in this prompt.

## Goal
$goal

## Output Format
Write plain markdown starting with '# Plan v$plan_version' and then 3-5 sections named '## ETAP N: ...'.
Each ETAP must contain numbered action steps, completion criteria, and a results table template.
""",

    worker_system_prompt=(
        "You are a Python developer. Complete the assigned task precisely. "
        "Return results as structured JSON."
    ),

    workers=[
        WorkerConfig(worker_id="qwen-1", role="executor", system_prompt="Focus on code implementation."),
        WorkerConfig(worker_id="qwen-2", role="tester", system_prompt="Focus on testing and validation."),
    ],

    poll_interval_seconds=60,
    planner_timeout_seconds=120,
    worker_timeout_seconds=180,

    max_errors_total=15,
    max_empty_cycles=8,
    max_task_attempts=3,

    planner_adapter=AdapterConfig(
        name="claude_planner_cli",
        cli_path="claude",
        model="opus",
    ),

    worker_adapter=AdapterConfig(
        name="qwen_worker_cli",
        cli_path="qwen",
    ),

    state_dir="state",
    log_level="INFO",
)


if __name__ == "__main__":
    import json
    print(json.dumps(demo_config.to_dict(), indent=2))
