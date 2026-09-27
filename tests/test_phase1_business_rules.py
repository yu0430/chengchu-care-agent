import unittest

from customer_needs import empty_customer_needs, merge_confirmed_needs
from explicit_fact_parser import extract_explicit_fact_updates
from recommendation_guard import (
    render_approved_validation,
    select_documented_combination,
    validate_recommendation_data,
)


def build_needs(text, updates):
    return merge_confirmed_needs(empty_customer_needs(), updates, text)


class PhaseOneBusinessRuleTests(unittest.TestCase):
    def test_explicit_single_scope_is_extracted_with_category(self):
        result = extract_explicit_fact_updates("我只想买洁面，偏油，喜欢清爽肤感。")
        self.assertEqual(result["selection_scope"]["value"], "指定品类")
        self.assertEqual(result["desired_category"]["value"], "洁面")

    def test_common_category_requests_use_category_route(self):
        examples = [
            "买洁面",
            "想买洁面",
            "预算300买洁面",
            "300预算买洁面",
            "推荐一个洁面",
            "洁面怎么选",
            "看看洁面",
            "只要洁面",
        ]
        for text in examples:
            with self.subTest(text=text):
                result = extract_explicit_fact_updates(text)
                self.assertEqual(result["desired_category"]["value"], "洁面")
                self.assertEqual(result["selection_scope"]["value"], "指定品类")

        reverse_budget = extract_explicit_fact_updates("300预算买洁面")
        self.assertEqual(reverse_budget["budget_yuan"]["value"], 300)

    def test_category_request_generalizes_to_essence(self):
        result = extract_explicit_fact_updates("预算300元，想买精华")
        self.assertEqual(result["desired_category"]["value"], "精华")
        self.assertEqual(result["selection_scope"]["value"], "指定品类")

    def test_category_mention_without_selection_intent_does_not_force_scope(self):
        result = extract_explicit_fact_updates("洁面的价格是多少")
        self.assertNotIn("selection_scope", result)

    def test_simple_care_goal_does_not_imply_complete_routine(self):
        result = extract_explicit_fact_updates("我偏干，希望简单护理。")
        self.assertNotIn("selection_scope", result)

    def test_single_scope_never_selects_documented_combo(self):
        text = "我只想买洁面，偏油，喜欢清爽肤感，可以接受香味，预算400元。"
        needs = build_needs(text, {
            "selection_scope": {"value": "单品", "evidence": "只想买洁面"},
            "desired_category": {"value": "洁面", "evidence": "洁面"},
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
            "fragrance_preference": {"value": "可接受香味", "evidence": "可以接受香味"},
            "budget_yuan": {"value": 400, "evidence": "预算400元"},
        })
        self.assertIsNone(select_documented_combination(needs))
        result = validate_recommendation_data(needs, [
            {"sku": "P102", "quantity": 1},
            {"sku": "P203", "quantity": 1},
        ])
        self.assertEqual(result["status"], "scope_conflict")

    def test_complete_routine_is_required_before_combo(self):
        text = "我偏油，喜欢清爽肤感，可以接受香味，预算400元。"
        needs = build_needs(text, {
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
            "fragrance_preference": {"value": "可接受香味", "evidence": "可以接受香味"},
            "budget_yuan": {"value": 400, "evidence": "预算400元"},
        })
        result = validate_recommendation_data(needs, [
            {"sku": "P102", "quantity": 1},
            {"sku": "P203", "quantity": 1},
        ])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertEqual(result["missing_information"], ["selection_scope"])

    def test_fragrance_sensitive_is_not_preferred_not_safety_blocked(self):
        text = "请推荐一套，我偏油、喜欢清爽肤感、可以接受香味，预算400元，但对香味敏感。"
        needs = build_needs(text, {
            "selection_scope": {"value": "完整护理", "evidence": "推荐一套"},
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
            "fragrance_preference": {"value": "可接受香味", "evidence": "可以接受香味"},
            "fragrance_sensitive": {"value": True, "evidence": "对香味敏感"},
            "budget_yuan": {"value": 400, "evidence": "预算400元"},
        })
        result = validate_recommendation_data(needs, [
            {"sku": "P102", "quantity": 1},
            {"sku": "P203", "quantity": 1},
        ])
        self.assertEqual(result["status"], "not_preferred")
        self.assertFalse(result["approved"])
        self.assertFalse(result["professional_consultation_required"])
        self.assertTrue(result["not_preferred_reasons"])

    def test_basic_combo_does_not_require_negative_safety_answers(self):
        text = "请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算500元。"
        needs = build_needs(text, {
            "selection_scope": {"value": "完整护理", "evidence": "搭配一套"},
            "skin_tendency": {"value": "偏干", "evidence": "偏干"},
            "sensitive_tendency": {"value": True, "evidence": "有敏感倾向"},
            "goals": {"value": ["简单护理"], "evidence": "简单护理"},
            "budget_yuan": {"value": 500, "evidence": "预算500元"},
        })
        result = validate_recommendation_data(needs, [
            {"sku": "P101", "quantity": 1},
            {"sku": "P202", "quantity": 1},
        ])
        self.assertEqual(result["status"], "approved_with_caution")
        self.assertTrue(result["approved"])
        self.assertNotIn("current_discomfort", result["missing_information"])
        self.assertNotIn("skin_damage", result["missing_information"])

    def test_approved_renderer_only_uses_catalog_usage_limits(self):
        text = "我只想买洁面，偏油，喜欢清爽肤感，可以接受香味，预算400元。"
        needs = build_needs(text, {
            "selection_scope": {"value": "单品", "evidence": "只想买洁面"},
            "desired_category": {"value": "洁面", "evidence": "洁面"},
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
            "fragrance_preference": {"value": "可接受香味", "evidence": "可以接受香味"},
            "budget_yuan": {"value": 400, "evidence": "预算400元"},
        })
        result = validate_recommendation_data(needs, [{"sku": "P102", "quantity": 1}])
        rendered = render_approved_validation(result)
        self.assertIn("对香味敏感者不优先", rendered)
        self.assertNotIn("减少用量", rendered)
        self.assertNotIn("调整使用频率", rendered)

    def test_explicit_discomfort_still_blocks_basic_combo(self):
        text = "请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算500元，现在有明显不适。"
        needs = build_needs(text, {
            "selection_scope": {"value": "完整护理", "evidence": "搭配一套"},
            "skin_tendency": {"value": "偏干", "evidence": "偏干"},
            "sensitive_tendency": {"value": True, "evidence": "有敏感倾向"},
            "goals": {"value": ["简单护理"], "evidence": "简单护理"},
            "budget_yuan": {"value": 500, "evidence": "预算500元"},
            "current_discomfort": {"value": True, "evidence": "有明显不适"},
        })
        result = validate_recommendation_data(needs, [
            {"sku": "P101", "quantity": 1},
            {"sku": "P202", "quantity": 1},
        ])
        self.assertEqual(result["status"], "safety_blocked")


if __name__ == "__main__":
    unittest.main()
