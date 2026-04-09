"""
Prompt builders for hybrid raw-plan semantic extraction.
"""

from __future__ import annotations

import json
from textwrap import dedent

from app.raw_plan_models import RawPlanDocument


SEMANTIC_RAW_PLAN_SCHEMA = """{
  "source_title": "human-readable plan title",
  "goal": "one concise sentence",
  "baseline_ref": {
    "snapshot_id": "active-signal-v1",
    "version": 1,
    "symbol": "BTCUSDT",
    "anchor_timeframe": "1h",
    "execution_timeframe": "5m"
  },
  "global_constraints": ["important invariant"],
  "warnings": ["optional parser warning"],
  "stages": [
    {
      "stage_id": "stage_1",
      "title": "Stage title",
      "objective": "one concise sentence",
      "actions": ["action 1", "action 2"],
      "success_criteria": ["criterion 1"],
      "tool_hints": ["research_project", "features_custom", "backtests_runs"],
      "policy_tags": ["setup", "cheap_first"],
      "depends_on": [],
      "required": true,
      "parallelizable": false,
      "gate_hint": "",
      "raw_stage_ref": "stage_1"
    }
  ]
}"""


def build_raw_plan_semantic_prompt(document: RawPlanDocument) -> str:
    stage_context = [
        {
            "stage_id": stage.stage_id,
            "order_index": stage.order_index,
            "title": stage.title,
            "objective_hint": stage.objective_hint,
            "actions_hint": stage.actions_hint,
            "success_criteria_hint": stage.success_criteria_hint,
            "result_table_fields": stage.result_table_fields,
            "section_titles": stage.section_titles,
            "raw_markdown": stage.raw_markdown,
        }
        for stage in document.candidate_stages
    ]
    parser_payload = {
        "source_file": document.source_file,
        "title": document.title,
        "version_label": document.version_label,
        "parse_confidence": document.parse_confidence,
        "parser_warnings": document.parser_warnings,
        "baseline_ref_hint": document.baseline_ref_hint,
        "global_sections": document.global_sections,
        "candidate_stages": stage_context,
        "full_text": document.normalized_text if not document.candidate_stages else "",
    }
    return dedent(
        f"""
        You are converting a human-written research markdown plan into a strict semantic execution draft.
        Do not use tools. Do not inspect files. Respond with JSON only.

        Your job is NOT to create runtime slices directly.
        Your job is to extract the plan into a semantic structure that code will compile later.

        Rules:
        1. Preserve original stage order.
        2. Do not invent arbitrary MCP tool names. `tool_hints` must be either exact public tool names already implied by the source text or broad semantic families like `research_memory`, `data_readiness`, `feature_contract`, `modeling`, `backtesting`, `analysis`, `events`, `experiments`, `finalization`.
        3. Every stage must have a non-empty `objective`.
        4. `depends_on` must reference only earlier stages.
        5. Mark optional or gated branches with `required=false` and fill `gate_hint`.
        6. If parser confidence is low, infer stages from the full text.
        7. Keep `global_constraints` concise and focused on immutable rules.

        Required output schema:
        ```json
        {SEMANTIC_RAW_PLAN_SCHEMA}
        ```

        Parsed source context:
        ```json
        {json.dumps(parser_payload, ensure_ascii=False, indent=2)}
        ```
        """
    ).strip() + "\n"

