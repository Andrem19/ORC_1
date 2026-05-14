"""Tests for zero-tool-call stall detection in DirectExecutionService."""
from __future__ import annotations

from unittest.mock import MagicMock

from app.execution_models import BaselineRef, ExecutionPlan, PlanSlice, WorkerAction, make_id
from app.services.direct_execution.service import DirectExecutionService
from tests.mcp_catalog_fixtures import make_catalog_snapshot


def _make_plan_with_slice(turn_count: int = 0, tool_call_count: int = 0) -> tuple[ExecutionPlan, PlanSlice]:
    plan = ExecutionPlan(
        plan_id="plan_1",
        goal="g",
        baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
        global_constraints=[],
        slices=[
            PlanSlice(
                slice_id="slice_1",
                title="t",
                hypothesis="h",
                objective="o",
                success_criteria=["done"],
                allowed_tools=["research_record"],
                evidence_requirements=["done"],
                policy_tags=["test"],
                max_turns=24,
                max_tool_calls=18,
                max_expensive_calls=0,
                parallel_slot=1,
            )
        ],
    )
    sl = plan.slices[0]
    sl.turn_count = turn_count
    sl.tool_call_count = tool_call_count
    return plan, sl


def _make_turn(status: str = "partial", summary: str = "partial done") -> object:
    class Turn:
        def __init__(self) -> None:
            self.turn_id = make_id("turn")
            self.action = WorkerAction(
                action_id=make_id("action"),
                action_type="checkpoint",
                status=status,
                summary=summary,
            )
            self.direct_attempt = MagicMock()
            self.direct_attempt.tool_call_count = 0

    return Turn()


def _make_service() -> DirectExecutionService:
    orch = MagicMock()
    orch.config = MagicMock()
    orch.config.workers = [MagicMock(worker_id="w1")]
    orch.config.max_concurrent_plan_tasks = 1
    orch.config.state_dir = "state"
    orch.config.current_run_id = "test_run"
    orch.config.plan_source = "compiled_raw"
    orch.config.compiled_plan_dir = "compiled_plans"
    orch.config.raw_plan_dir = "raw_plans"
    orch.config.compiled_queue_skip_failures = True
    orch.config.research_config = {}
    orch.config.worker_system_prompt = ""
    orch.config.direct_execution = MagicMock()
    orch.config.direct_execution.provider = "lmstudio"
    orch.config.direct_execution.mcp_endpoint_url = "http://localhost"
    orch.config.direct_execution.mcp_auth_mode = "none"
    orch.config.direct_execution.mcp_token_env_var = ""
    orch.config.direct_execution.connect_timeout_seconds = 10
    orch.config.direct_execution.read_timeout_seconds = 60
    orch.config.direct_execution.retry_budget = 1
    orch.config.direct_execution.max_tool_calls_per_slice = 24
    orch.config.direct_execution.max_expensive_tool_calls_per_slice = 6
    orch.config.direct_execution.timeout_seconds = 600
    orch.config.direct_execution.safe_exclude_tools = []
    orch.config.direct_execution.first_action_timeout_seconds = 45
    orch.config.direct_execution.stalled_action_timeout_seconds = 60
    orch.execution_state = MagicMock()
    orch.execution_state.plans = []
    orch.execution_store = MagicMock()
    orch.artifact_store = MagicMock()
    orch.notification_service = MagicMock()
    orch.process_registry = MagicMock()
    orch.mcp_catalog_snapshot = make_catalog_snapshot()
    return DirectExecutionService(orch)


def test_partial_checkpoint_not_blocked_on_first_attempt() -> None:
    """First attempt with 0 tool calls should NOT trigger stall detection."""
    svc = _make_service()
    plan, sl = _make_plan_with_slice(turn_count=1, tool_call_count=0)
    turn = _make_turn(status="partial", summary="thinking...")

    svc._apply_checkpoint(plan, sl, turn)

    assert sl.status == "checkpointed"
    assert sl.last_checkpoint_status == "partial"
    assert sl.last_error == ""


def test_partial_checkpoint_blocked_after_two_attempts_with_zero_tools() -> None:
    """After 2+ turns with 0 tool calls total, partial checkpoint should be overridden to blocked."""
    svc = _make_service()
    plan, sl = _make_plan_with_slice(turn_count=2, tool_call_count=0)
    turn = _make_turn(status="partial", summary="thinking again...")

    svc._apply_checkpoint(plan, sl, turn)

    assert sl.status == "checkpointed"
    assert sl.last_checkpoint_status == "blocked"
    assert sl.last_error == "direct_model_zero_tool_call_stall"
    assert "zero tool calls" in sl.last_summary.lower()


def test_partial_checkpoint_not_blocked_if_tools_were_used() -> None:
    """If tool_call_count > 0, stall detection should NOT trigger."""
    svc = _make_service()
    plan, sl = _make_plan_with_slice(turn_count=3, tool_call_count=5)
    turn = _make_turn(status="partial", summary="partial progress")

    svc._apply_checkpoint(plan, sl, turn)

    assert sl.status == "checkpointed"
    assert sl.last_checkpoint_status == "partial"
    assert sl.last_error == ""


def test_blocked_checkpoint_not_overridden() -> None:
    """A watchdog 'blocked' checkpoint should not be overridden by stall detection."""
    svc = _make_service()
    plan, sl = _make_plan_with_slice(turn_count=3, tool_call_count=0)
    turn = _make_turn(status="blocked", summary="model stalled")
    turn.action.reason_code = "direct_model_stalled_before_first_action"

    svc._apply_checkpoint(plan, sl, turn)

    assert sl.last_checkpoint_status == "blocked"
    assert sl.last_error == "direct_model_stalled_before_first_action"
    # Should NOT contain the stall override text
    assert "zero tool calls" not in sl.last_summary.lower()


def test_is_zero_tool_call_stall_detects_stall() -> None:
    _, sl = _make_plan_with_slice(turn_count=3, tool_call_count=0)
    sl.last_checkpoint_status = "partial"
    assert DirectExecutionService._is_zero_tool_call_stall(sl) is True


def test_is_zero_tool_call_stall_not_stall_with_tools() -> None:
    _, sl = _make_plan_with_slice(turn_count=3, tool_call_count=1)
    sl.last_checkpoint_status = "partial"
    assert DirectExecutionService._is_zero_tool_call_stall(sl) is False


def test_is_zero_tool_call_stall_not_stall_first_attempt() -> None:
    _, sl = _make_plan_with_slice(turn_count=1, tool_call_count=0)
    sl.last_checkpoint_status = "partial"
    assert DirectExecutionService._is_zero_tool_call_stall(sl) is False


def test_prompt_includes_must_use_tools_guidance() -> None:
    from app.services.direct_execution.prompt import build_direct_slice_prompt

    prompt = build_direct_slice_prompt(
        plan_id="p1",
        slice_payload={"slice_id": "s1", "title": "t", "hypothesis": "h", "objective": "o",
                        "success_criteria": [], "evidence_requirements": []},
        baseline_bootstrap={"baseline_snapshot_id": "active-signal-v1", "baseline_version": 1,
                            "symbol": "BTCUSDT", "anchor_timeframe": "1h", "execution_timeframe": "5m"},
        known_facts={},
        recent_turn_summaries=[],
        checkpoint_summary="",
        allowed_tools=["research_record"],
        max_tool_calls=8,
        max_expensive_tool_calls=2,
    )
    assert "MUST call at least one tool" in prompt
    assert "Do not return a checkpoint without using tools" in prompt
