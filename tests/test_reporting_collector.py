from __future__ import annotations

from app.execution_models import BaselineRef
from app.reporting.collector import _aggregate_plan_verdict, _rollup_key_metrics
from app.reporting.models import PlanBatchReport, SliceResultReport


def test_aggregate_plan_verdict_returns_failed_when_slice_failed() -> None:
    verdict = _aggregate_plan_verdict(
        slice_results=[
            SliceResultReport(
                slice_id="slice_1",
                title="Slice 1",
                status="failed",
                verdict="FAILED",
            )
        ],
        plan_status="completed",
    )

    assert verdict == "FAILED"


def test_rollup_key_metrics_preserves_conflicting_values_deterministically() -> None:
    reports = [
        PlanBatchReport(
            plan_id="plan_1",
            plan_source_kind="compiled_raw",
            baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
            slice_results=[
                SliceResultReport(
                    slice_id="slice_a",
                    title="Slice A",
                    status="completed",
                    key_metrics={"sharpe_ratio": 1.1, "trades": 40},
                )
            ],
        ),
        PlanBatchReport(
            plan_id="plan_2",
            plan_source_kind="compiled_raw",
            baseline_ref=BaselineRef(snapshot_id="active-signal-v1", version=1),
            slice_results=[
                SliceResultReport(
                    slice_id="slice_b",
                    title="Slice B",
                    status="completed",
                    key_metrics={"sharpe_ratio": 1.3, "trades": 40},
                )
            ],
        ),
    ]

    rollup = _rollup_key_metrics(reports)

    assert rollup["trades"] == 40
    assert rollup["sharpe_ratio"] == [
        {"plan_id": "plan_1", "slice_id": "slice_a", "value": 1.1},
        {"plan_id": "plan_2", "slice_id": "slice_b", "value": 1.3},
    ]
