from __future__ import annotations

from debuger import classify_fallback_signal


def test_classify_fallback_signal_distinguishes_attempt_from_failed_chain() -> None:
    attempt_line = (
        "2026-04-17 01:25:39 [INFO] orchestrator.direct.fallback - "
        "Attempting fallback #1 with provider 'claude_cli' for slice compiled_plan_v2_stage_5"
    )
    failed_line = (
        "2026-04-17 01:30:00 [WARNING] orchestrator.direct.fallback - "
        "Fallback #1 (claude_cli) also failed for slice compiled_plan_v2_stage_5: auto_salvage_stub_rejected"
    )

    failed_slice, attempted_slice = classify_fallback_signal(attempt_line)
    assert failed_slice is None
    assert attempted_slice == "compiled_plan_v2_stage_5"

    failed_slice, attempted_slice = classify_fallback_signal(failed_line)
    assert failed_slice == "compiled_plan_v2_stage_5"
    assert attempted_slice is None
