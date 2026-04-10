from __future__ import annotations

from app.services.direct_execution.research_handles import (
    ResearchHandleState,
    repair_atlas_coordinates,
    repair_experiment_handle,
    repair_project_handle,
    update_handle_state,
)
from app.services.direct_execution.tool_preflight import preflight_direct_tool_call


def test_update_handle_state_tracks_confirmed_project_id_from_create() -> None:
    state = ResearchHandleState()
    update_handle_state(
        tool_name="research_project",
        arguments={"action": "create", "project": {"name": "active-signal-v1"}},
        result_payload={
            "ok": True,
            "payload": {
                "structuredContent": {
                    "data": {
                        "project": {
                            "project_id": "active-signal-v1-486b2d99",
                            "name": "active-signal-v1",
                        }
                    }
                }
            },
        },
        state=state,
    )

    assert state.confirmed_project_id == "active-signal-v1-486b2d99"
    assert "active-signal-v1" in state.project_aliases


def test_repair_project_handle_rewrites_stale_alias_to_confirmed_id() -> None:
    state = ResearchHandleState(
        confirmed_project_id="active-signal-v1-486b2d99",
        project_aliases={"active-signal-v1", "active-signal-v1-486b2d99"},
    )

    repaired, notes = repair_project_handle(
        tool_name="research_map",
        arguments={"action": "inspect", "project_id": "active-signal-v1"},
        state=state,
    )

    assert repaired["project_id"] == "active-signal-v1-486b2d99"
    assert notes


def test_repair_atlas_coordinates_coerces_aliases_and_types() -> None:
    state = ResearchHandleState(
        atlas_dimensions={
            "execution": ["5m", "1h"],
            "symbol": ["BTCUSDT"],
            "timeframe": [1, 5, 15, 60],
        }
    )

    repaired, notes = repair_atlas_coordinates(
        tool_name="research_record",
        arguments={
            "action": "create",
            "kind": "hypothesis",
            "atlas": {
                "coordinates": {
                    "execution": 5,
                    "symbol": "BTCUSDT",
                    "anchor": "1h",
                }
            },
        },
        state=state,
    )

    assert repaired["atlas"]["coordinates"] == {
        "execution": "5m",
        "symbol": "BTCUSDT",
        "timeframe": 1,
    }
    assert notes


def test_update_handle_state_tracks_experiment_job_id_from_experiments_run() -> None:
    state = ResearchHandleState()
    update_handle_state(
        tool_name="experiments_run",
        arguments={"action": "start"},
        result_payload={
            "ok": True,
            "payload": {
                "structuredContent": {
                    "data": {
                        "job": {"job_id": "20260410-155728-d5680c26"},
                    }
                }
            },
        },
        state=state,
    )

    assert state.confirmed_experiment_job_id == "20260410-155728-d5680c26"
    assert "20260410-155728-d5680c26" in state.experiment_job_ids


def test_repair_experiment_handle_fills_placeholder_job_id() -> None:
    state = ResearchHandleState(
        confirmed_experiment_job_id="20260410-155728-d5680c26",
        experiment_job_ids={"20260410-155728-d5680c26"},
    )
    repaired, notes = repair_experiment_handle(
        tool_name="experiments_read",
        arguments={"view": "list", "job_id": "job_123"},
        state=state,
    )

    assert repaired["job_id"] == "20260410-155728-d5680c26"
    assert notes


def test_repair_atlas_coordinates_after_preflight_hoists_and_coerces_values() -> None:
    state = ResearchHandleState(
        atlas_dimensions={
            "anchor": [1],
            "execution": ["5m"],
            "symbol": ["BTCUSDT"],
        }
    )
    preflight = preflight_direct_tool_call(
        "research_record",
        {
            "action": "create",
            "kind": "hypothesis",
            "project_id": "proj_1",
            "record": {
                "title": "H1",
                "summary": "s",
                "atlas": {
                    "statement": "st",
                    "expected_outcome": "ok",
                    "falsification_criteria": "fail",
                    "coordinates": {"anchor": "1h", "execution": "5m", "symbol": "BTCUSDT"},
                },
            },
        },
    )
    repaired, notes = repair_atlas_coordinates(
        tool_name="research_record",
        arguments=preflight.arguments,
        state=state,
    )

    assert repaired["atlas"]["coordinates"]["anchor"] == 1
    assert notes


def test_repair_atlas_coordinates_selects_valid_value_from_list_candidate() -> None:
    state = ResearchHandleState(
        atlas_dimensions={
            "strategy_state": ["active", "testing"],
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h", "5m"],
        }
    )
    repaired, _ = repair_atlas_coordinates(
        tool_name="research_record",
        arguments={
            "action": "create",
            "kind": "hypothesis",
            "atlas": {
                "coordinates": {
                    "strategy_state": "active",
                    "symbol": "BTCUSDT",
                    "timeframe": ["5m", "1h"],
                }
            },
        },
        state=state,
    )

    assert repaired["atlas"]["coordinates"]["timeframe"] in {"1h", "5m"}


def test_repair_atlas_coordinates_falls_back_to_first_allowed_value() -> None:
    state = ResearchHandleState(
        atlas_dimensions={
            "strategy_state": ["active", "testing"],
            "symbol": ["BTCUSDT"],
            "timeframe": ["1h", "5m"],
        }
    )
    repaired, _ = repair_atlas_coordinates(
        tool_name="research_record",
        arguments={
            "action": "create",
            "kind": "hypothesis",
            "atlas": {
                "coordinates": {
                    "strategy_state": "active",
                    "symbol": "BTCUSDT",
                    "timeframe": "multi",
                }
            },
        },
        state=state,
    )

    assert repaired["atlas"]["coordinates"]["timeframe"] == "1h"
