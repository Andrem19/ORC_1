from __future__ import annotations

from app.services.direct_execution.prompt import build_direct_slice_prompt


def test_direct_prompt_includes_research_contract_hints() -> None:
    prompt = build_direct_slice_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Research shortlist",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["done"],
            "evidence_requirements": ["proof"],
        },
        baseline_bootstrap={
            "baseline_snapshot_id": "active-signal-v1",
            "baseline_version": 1,
            "symbol": "BTCUSDT",
            "anchor_timeframe": "1h",
            "execution_timeframe": "5m",
        },
        known_facts={"compiled_plan_v1_stage_1.atlas_dimensions": ["market_regime", "timeframe"]},
        recent_turn_summaries=[],
        checkpoint_summary="",
        allowed_tools=["research_record", "research_search", "research_map"],
        max_tool_calls=16,
        max_expensive_tool_calls=4,
        worker_system_prompt="",
    )

    assert "top-level atlas={statement, expected_outcome, falsification_criteria, coordinates}" in prompt
    assert "Do not put atlas inside record" in prompt
    assert "research_search requires a concrete non-empty query" in prompt
    assert "market_regime, timeframe" in prompt


def _base_prompt_kwargs(**overrides):  # type: ignore[no-untyped-def]
    kwargs = dict(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "s1", "title": "t", "hypothesis": "h", "objective": "o",
            "success_criteria": [], "evidence_requirements": [],
        },
        baseline_bootstrap={
            "baseline_snapshot_id": "active-signal-v1", "baseline_version": 1,
            "symbol": "BTCUSDT", "anchor_timeframe": "1h", "execution_timeframe": "5m",
        },
        known_facts={},
        recent_turn_summaries=[],
        checkpoint_summary="",
        max_tool_calls=8,
        max_expensive_tool_calls=2,
    )
    kwargs.update(overrides)
    return kwargs


def test_hint_research_map_project_id_included_when_tool_allowed() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["research_map"]))
    assert "research_map always requires project_id" in prompt


def test_hint_research_project_list_before_create() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["research_project"]))
    assert "research_project(action='list')" in prompt


def test_hint_experiments_read_job_id_required() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["experiments_read"]))
    assert "NOT a listing tool" in prompt
    assert "job_id" in prompt


def test_hint_experiments_inspect_list_first() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["experiments_inspect"]))
    assert "experiments_inspect(view='list')" in prompt


def test_hint_backtests_conditions_snapshot_id_required() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["backtests_conditions"]))
    assert "snapshot_id" in prompt
    assert "backtests_strategy" not in prompt


def test_hint_backtests_strategy_only_when_allowed() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(allowed_tools=["backtests_conditions", "backtests_strategy"])
    )
    assert "backtests_strategy" in prompt


def test_hint_research_project_create_returns_project_id() -> None:
    prompt = build_direct_slice_prompt(**_base_prompt_kwargs(allowed_tools=["research_project"]))
    assert "project_id in the ids field" in prompt
    assert "research_project(action='list')" in prompt


def test_known_facts_are_compacted_without_recursive_prefix_chains() -> None:
    prompt = build_direct_slice_prompt(
        **_base_prompt_kwargs(
            allowed_tools=["research_project"],
            slice_payload={
                "slice_id": "s2",
                "title": "t",
                "hypothesis": "h",
                "objective": "o",
                "success_criteria": [],
                "evidence_requirements": [],
                "depends_on": ["stage_6"],
            },
            known_facts={
                "stage_6.stage_5.stage_4.project_id": "proj_1",
                "stage_3.stage_2.stage_1.shortlist": ["momentum"],
            },
        )
    )
    assert "stage_6.project_id = proj_1" in prompt
    assert "stage_6.stage_5" not in prompt
    assert "stage_3.shortlist" in prompt
