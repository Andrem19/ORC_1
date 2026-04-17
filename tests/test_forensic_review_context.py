"""Tests for forensic-review / backtests-analysis context detection.

Regression tests for the fix that ensures slices with forensic/review/reconstruct
markers in their title/objective are recognized as backtests analysis context slices,
so the mixed-domain exploration guard prevents research_memory loops.
"""

from __future__ import annotations

from app.services.direct_execution.backtests_protocol import is_backtests_context


# --- Unit: is_backtests_context recognizes forensic/review markers ---


def test_forensic_in_title() -> None:
    """Slices with 'forensic' in the title are backtests context."""
    assert is_backtests_context(
        title="Полный forensic-review ветки A1",
        objective="Reconstruct the exact logic",
        allowed_tools={"backtests_strategy", "backtests_runs", "backtests_analysis", "research_memory"},
    )


def test_forensic_in_objective() -> None:
    assert is_backtests_context(
        title="Some slice",
        objective="Perform a forensic audit of v2 results",
        allowed_tools={"backtests_runs", "research_memory"},
    )


def test_reconstruct_marker() -> None:
    assert is_backtests_context(
        title="Reconstruct A1 logic",
        allowed_tools={"backtests_strategy", "research_memory"},
    )


def test_reproducible_marker() -> None:
    assert is_backtests_context(
        objective="Establish a reproducible baseline",
        allowed_tools={"backtests_runs", "research_memory"},
    )


def test_reproduce_marker() -> None:
    assert is_backtests_context(
        title="Reproduce v2 claimed results",
        allowed_tools={"backtests_analysis"},
    )


def test_review_marker() -> None:
    assert is_backtests_context(
        title="Review of backtest results",
        allowed_tools={"backtests_runs"},
    )


def test_audit_marker() -> None:
    assert is_backtests_context(
        title="Audit backtests artifacts",
        allowed_tools={"backtests_strategy", "backtests_runs"},
    )


def test_no_markers_no_backtests_tools() -> None:
    """Without markers and without backtests tools, should return False."""
    assert not is_backtests_context(
        title="Some research slice",
        objective="Explore data",
        allowed_tools={"research_memory", "research_project"},
    )


def test_no_markers_with_backtests_tools() -> None:
    """Without markers but with backtests tools, should still return False.

    The markers in the title/objective are still required even when
    backtests tools are present — the markers provide domain intent.
    """
    result = is_backtests_context(
        title="Generic slice",
        objective="Do something",
        allowed_tools={"backtests_runs", "research_memory"},
    )
    # This returns False because neither title nor objective has any marker
    assert not result


# --- Integration: compiled_plan_v3_stage_2 pattern recognition ---


def test_compiled_plan_v3_stage_2_is_backtests_context() -> None:
    """The exact slice that failed should now be recognized as backtests context."""
    assert is_backtests_context(
        runtime_profile="generic_mutation",
        title="Полный forensic-review ветки A1",
        objective=(
            "Reconstruct the exact logic, artifacts, and claimed results "
            "of the A1_high_iv_long_v1 branch from v2 to establish a "
            "reproducible baseline before any new testing."
        ),
        success_criteria=[
            "A1 logic is strictly and reproducibly documented in a v3 snapshot",
            "v2 claimed results are compared against v3 reproduced results",
            "Any drift or mismatch is explicitly recorded",
        ],
        policy_tags=["forensic", "reproducibility"],
        allowed_tools={"backtests_strategy", "backtests_runs", "backtests_analysis", "research_memory"},
    )


def test_compiled_plan_v3_stage_2_triggers_mixed_domain_guard() -> None:
    """Verify that the tool loop's _is_backtests_analysis_context_slice
    returns True for the forensic-review pattern."""
    from app.services.direct_execution.lmstudio_tool_loop import LmStudioToolLoop

    # Build a minimal tool loop with the failing slice's parameters
    from app.adapters.lmstudio_worker_api import LmStudioWorkerApi
    from unittest.mock import MagicMock

    adapter = MagicMock(spec=LmStudioWorkerApi)
    adapter.base_url = "https://api.minimax.io"
    adapter.api_key = "test"
    adapter.model = "test"
    adapter.temperature = 0.5
    adapter.max_tokens = 1024
    adapter.reasoning_effort = ""
    adapter.extra_body = None

    loop = LmStudioToolLoop(
        adapter=adapter,
        mcp_client=MagicMock(),
        incident_store=MagicMock(),
        allowed_tools={"backtests_strategy", "backtests_runs", "backtests_analysis", "research_memory"},
        slice_title="Полный forensic-review ветки A1",
        slice_objective="Reconstruct the exact logic of the A1 branch",
        success_criteria=["A1 logic is reproducibly documented"],
        policy_tags=["forensic", "reproducibility"],
        required_output_facts=[],
        runtime_profile="generic_mutation",
        finalization_mode="none",
        max_tool_calls=36,
        max_expensive_tool_calls=12,
        safe_exclude_tools=set(),
        first_action_timeout_seconds=75,
        stalled_action_timeout_seconds=60,
    )

    assert loop._is_backtests_analysis_context_slice(), (
        "Forensic-review slice should be recognized as backtests analysis context"
    )
    assert loop._is_backtests_slice(), (
        "Forensic-review slice should be recognized as backtests slice"
    )


def test_compiled_plan_v3_stage_2_triggers_prompt_guide() -> None:
    """Verify that the prompt builder includes backtests analysis protocol."""
    from app.services.direct_execution.prompt import _looks_like_backtests_analysis_slice

    assert _looks_like_backtests_analysis_slice(
        slice_payload={
            "title": "Полный forensic-review ветки A1",
            "objective": "Reconstruct the exact logic of the A1 branch",
            "success_criteria": ["A1 logic is reproducibly documented"],
            "policy_tags": ["forensic", "reproducibility"],
        },
        allowed_tools=["backtests_strategy", "backtests_runs", "backtests_analysis", "research_memory"],
    )
