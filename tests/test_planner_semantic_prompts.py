"""
Unit tests for app.planner_semantic_prompts.
"""

from app.planner_semantic_prompts import build_planner_semantic_prompt


class TestBuildPlannerSemanticPrompt:
    def test_includes_goal(self):
        prompt = build_planner_semantic_prompt(goal="Investigate alpha signals")
        assert "Investigate alpha signals" in prompt

    def test_includes_baseline(self):
        prompt = build_planner_semantic_prompt(
            goal="G",
            baseline_bootstrap={
                "baseline_snapshot_id": "my-snap",
                "baseline_version": 3,
                "symbol": "ETHUSDT",
                "anchor_timeframe": "1h",
                "execution_timeframe": "5m",
            },
        )
        assert "my-snap@3" in prompt
        assert "ETHUSDT" in prompt

    def test_includes_plan_version(self):
        prompt = build_planner_semantic_prompt(goal="G", plan_version=7)
        assert "Plan version: 7" in prompt

    def test_includes_schema(self):
        prompt = build_planner_semantic_prompt(goal="G")
        assert "stage_id" in prompt
        assert "tool_hints" in prompt
        assert "success_criteria" in prompt

    def test_includes_operator_directives(self):
        prompt = build_planner_semantic_prompt(
            goal="G",
            operator_directives="Never use liquidation data",
        )
        assert "Never use liquidation data" in prompt
        assert "Operator directives" in prompt

    def test_omits_operator_directives_when_empty(self):
        prompt = build_planner_semantic_prompt(goal="G", operator_directives="  ")
        assert "Operator directives" not in prompt

    def test_includes_previous_state_summary(self):
        prompt = build_planner_semantic_prompt(
            goal="G",
            previous_state_summary="Last plan failed on dataset sync",
        )
        assert "Last plan failed on dataset sync" in prompt
        assert "Previous execution summary" in prompt

    def test_includes_previous_blockers(self):
        prompt = build_planner_semantic_prompt(
            goal="G",
            previous_blockers=["no_funding_data", "tool_timeout"],
        )
        assert "no_funding_data" in prompt
        assert "tool_timeout" in prompt

    def test_includes_tool_families(self):
        prompt = build_planner_semantic_prompt(
            goal="G",
            available_tools=["backtests_runs", "features_custom"],
        )
        assert "backtesting" in prompt
        assert "backtests_runs" in prompt
        assert "features_custom" in prompt

    def test_includes_worker_slots(self):
        prompt = build_planner_semantic_prompt(goal="G", worker_count=5)
        assert "Parallel worker slots available: 3" in prompt

    def test_includes_no_budget_instruction(self):
        prompt = build_planner_semantic_prompt(goal="G")
        # The prompt explicitly tells the LLM not to set budgets.
        assert "Do NOT set budget numbers" in prompt
        # The output schema should not have budget fields.
        assert '"max_turns"' not in prompt
        assert '"max_tool_calls"' not in prompt
