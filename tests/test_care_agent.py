"""care_agent 的离线结构与验收边界测试；不向 Qwen 发送请求。"""

import unittest

import care_agent
from customer_needs import empty_customer_needs, merge_confirmed_needs


def build_needs(text, updates):
    return merge_confirmed_needs(empty_customer_needs(), updates, text)


class CareAgentStructureTests(unittest.TestCase):
    def test_only_care_business_tools_are_bound(self):
        self.assertEqual(
            [tool.name for tool in care_agent.TOOLS],
            ["search_products", "get_policy", "calculate_quote", "validate_recommendation"],
        )

    def test_submission_boundaries_are_in_system_prompt(self):
        required_phrases = [
            "唯一事实来源",
            "不得自行口算",
            "需要向门店或系统确认",
            "品牌资料未覆盖，无法判断，建议人工复核",
            "不作医疗诊断",
            "不得为了满足预算",
            "已确认需求",
            "为什么这样建议",
            "价格与预算",
            "使用提醒与限制",
            "待确认信息",
        ]
        for phrase in required_phrases:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, care_agent.SYSTEM_PROMPT)

    def test_system_prompt_requires_all_four_tools(self):
        for tool_name in ("search_products", "get_policy", "calculate_quote", "validate_recommendation"):
            with self.subTest(tool_name=tool_name):
                self.assertIn(tool_name, care_agent.SYSTEM_PROMPT)

    def test_validator_schema_hides_injected_state(self):
        schema = care_agent.validate_recommendation.tool_call_schema.model_json_schema()
        self.assertEqual(set(schema["properties"]), {"items"})

    def test_domain_questions_require_grounding(self):
        messages = [care_agent.HumanMessage(content="P101 多少钱？")]
        self.assertTrue(care_agent._turn_requires_grounding(messages))
        self.assertFalse(care_agent._turn_requests_recommendation(messages))

    def test_personalized_budget_turn_requires_recommendation_validation(self):
        messages = [
            care_agent.HumanMessage(
                content="请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算400元。"
            )
        ]
        self.assertTrue(care_agent._turn_requires_grounding(messages))
        self.assertTrue(care_agent._turn_requests_recommendation(messages))

    def test_profile_statement_alone_is_not_a_recommendation_request(self):
        messages = [care_agent.HumanMessage(content="我偏油，预算400元。")]
        self.assertTrue(care_agent._turn_requires_grounding(messages))
        self.assertFalse(care_agent._turn_requests_recommendation(messages))

    def test_category_with_budget_compares_candidates_instead_of_asking_scope(self):
        text = "300预算买洁面"
        needs = build_needs(text, {
            "selection_scope": {"value": "指定品类", "evidence": "买洁面"},
            "desired_category": {"value": "洁面", "evidence": "买洁面"},
            "budget_yuan": {"value": 300, "evidence": "300预算"},
        })
        plan = care_agent._category_route_plan(needs)
        self.assertEqual(plan["action"], "compare")
        self.assertIn("P101", plan["content"])
        self.assertIn("P102", plan["content"])
        self.assertNotIn("先确认你的选择范围", plan["content"])

    def test_oily_fresh_cleanser_route_selects_p102(self):
        text = "我偏油，喜欢清爽肤感，预算300元，买洁面"
        needs = build_needs(text, {
            "selection_scope": {"value": "指定品类", "evidence": "买洁面"},
            "desired_category": {"value": "洁面", "evidence": "买洁面"},
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
            "budget_yuan": {"value": 300, "evidence": "预算300元"},
        })
        plan = care_agent._category_route_plan(needs)
        self.assertEqual(plan, {"action": "validate", "sku": "P102"})

    def test_dry_sensitive_cleanser_route_selects_p101(self):
        text = "我偏干，有敏感倾向，预算300元，买洁面"
        needs = build_needs(text, {
            "selection_scope": {"value": "指定品类", "evidence": "买洁面"},
            "desired_category": {"value": "洁面", "evidence": "买洁面"},
            "skin_tendency": {"value": "偏干", "evidence": "偏干"},
            "sensitive_tendency": {"value": True, "evidence": "有敏感倾向"},
            "budget_yuan": {"value": 300, "evidence": "预算300元"},
        })
        plan = care_agent._category_route_plan(needs)
        self.assertEqual(plan, {"action": "validate", "sku": "P101"})

    def test_greeting_does_not_force_business_tool(self):
        messages = [care_agent.HumanMessage(content="你好")]
        self.assertFalse(care_agent._turn_requires_grounding(messages))
        self.assertFalse(care_agent._turn_requests_recommendation(messages))

    def test_tool_round_limit_is_bounded(self):
        self.assertGreaterEqual(care_agent.MAX_TOOL_ROUNDS, 1)
        self.assertLessEqual(care_agent.MAX_TOOL_ROUNDS, 4)

    def test_thread_config_isolates_conversations(self):
        config = care_agent.thread_config(" visitor-001 ")
        self.assertEqual(config["configurable"]["thread_id"], "visitor-001")
        self.assertEqual(config["recursion_limit"], 12)

    def test_empty_thread_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "thread_id"):
            care_agent.thread_config("  ")

    def test_graph_builds_without_network_request(self):
        graph = care_agent.build_care_agent()
        node_names = set(graph.get_graph().nodes)
        self.assertTrue({"extract_needs", "plan", "assistant", "tools"}.issubset(node_names))

    def test_recommendation_flow_has_no_forced_tool_override(self):
        source = open(care_agent.__file__, encoding="utf-8").read()
        self.assertNotIn('tool_choice="required"', source)
        self.assertNotIn("required_validator_llm", source)


if __name__ == "__main__":
    unittest.main()
