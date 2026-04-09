from __future__ import annotations

from pathlib import Path

from app.raw_plan_parser import parse_raw_plan_file


RAW_DIR = Path(__file__).resolve().parents[1] / "raw_plans"


def test_parse_known_raw_plans_extracts_stages_and_confidence() -> None:
    for name in ("plan_v1.md", "plan_v6.md", "plan_v12.md"):
        document = parse_raw_plan_file(RAW_DIR / name)
        assert document.source_file.endswith(name)
        assert document.source_hash
        assert document.title
        assert document.candidate_stages
        assert document.parse_confidence >= 0.45


def test_parse_raw_plan_extracts_baseline_hint_when_present() -> None:
    document = parse_raw_plan_file(RAW_DIR / "plan_v1.md")

    assert document.baseline_ref_hint["snapshot_id"] == "active-signal-v1"
    assert document.baseline_ref_hint["version"] == 1
