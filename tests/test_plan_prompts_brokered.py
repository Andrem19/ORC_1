from __future__ import annotations

from app.plan_prompts import build_brokered_plan_creation_prompt, build_brokered_worker_prompt


def test_brokered_worker_prompt_forbids_mcp_prefixed_tool_names() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Catalog",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["features_catalog", "events", "research_record"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 0,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert "valid: features_catalog" in prompt
    assert "invalid: mcp__dev_space1__features_catalog" in prompt
    assert "Never add MCP namespace" in prompt
    assert "Do not treat the worker CLI session's own tool registry as relevant" in prompt


def test_brokered_plan_creation_prompt_includes_known_blockers() -> None:
    prompt = build_brokered_plan_creation_prompt(
        goal="find liquidation edge",
        baseline_bootstrap={},
        plan_version=1,
        worker_count=3,
        available_tools=["events", "features_catalog"],
        previous_state_summary="slice_a failed",
        previous_blockers=[
            "domain_capability_gap_liquidation_events_missing: no liquidation event family",
        ],
    )

    assert "Known blockers and capability gaps" in prompt
    assert "If previous blockers mention missing liquidation-family coverage" in prompt
    assert "do not plan liquidation_* feature engineering as already feasible" in prompt


def test_brokered_worker_prompt_includes_relevant_broker_constraints() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Search and sync",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["research_search", "events_sync"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 1,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert "Broker-enforced tool constraints:" in prompt
    assert 'For `research_search`, use `level="normal"`' in prompt
    assert "do not use the baseline snapshot id as `project_id`" in prompt
    assert "omit `project_id` unless a real research project id was explicitly confirmed" in prompt
    assert 'never use `wait="completed"`' in prompt
    assert 'For `events_sync`, always include both `family` and `scope`' in prompt
    assert "do not start another `events_sync` call with the same intent" in prompt


def test_brokered_worker_prompt_includes_features_catalog_alias_constraint() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Catalog",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["features_catalog"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 0,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert 'prefer `scope="available"`' in prompt


def test_brokered_worker_prompt_includes_datasets_preview_constraints() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Datasets",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["datasets", "datasets_preview"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 0,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert 'always include both `dataset_id` and `view`' in prompt
    assert 'Do not pass `symbol` or `timeframes` directly to `datasets_preview`' in prompt


def test_brokered_worker_prompt_includes_features_custom_contract_constraints() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Custom features",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["features_custom"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 1,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert 'never use `action="create"`' in prompt
    assert 'features_custom(action="inspect", view="contract")' in prompt


def test_brokered_worker_prompt_includes_features_analytics_selector_constraint() -> None:
    prompt = build_brokered_worker_prompt(
        plan_id="plan_1",
        slice_payload={
            "slice_id": "slice_1",
            "title": "Analytics",
            "hypothesis": "h",
            "objective": "o",
            "success_criteria": ["x"],
            "allowed_tools": ["features_analytics"],
            "max_turns": 3,
            "max_tool_calls": 2,
            "max_expensive_calls": 1,
        },
        baseline_bootstrap={},
        known_facts={},
        recent_turn_summaries=[],
        latest_tool_summary="",
        remaining_budget={},
        checkpoint_summary="",
        active_operation={},
    )

    assert "always specify one concrete feature" in prompt
    assert 'Do not call `features_analytics(action="heatmap"|"analytics"|"render"|"portability")` without a specific feature selector.' in prompt
