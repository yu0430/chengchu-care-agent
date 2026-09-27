import unittest

from customer_needs import empty_customer_needs, merge_confirmed_needs
from recommendation_guard import (
    render_unapproved_validation,
    select_documented_combination,
    validate_recommendation_data,
)


def needs_from(text, updates):
    updates = dict(updates)
    if "一套" in text and "selection_scope" not in updates:
        updates["selection_scope"] = {"value": "完整护理", "evidence": "一套"}
    return merge_confirmed_needs(empty_customer_needs(), updates, text)


class RecommendationGuardTests(unittest.TestCase):
    def test_documented_dry_sensitive_combo_reports_budget_conflict(self):
        text = "请帮我搭配一套，请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算400元。"
        needs = needs_from(
            text,
            {
                "skin_tendency": {"value": "偏干", "evidence": "偏干"},
                "sensitive_tendency": {"value": True, "evidence": "有敏感倾向"},
                "goals": {"value": ["简单护理"], "evidence": "简单护理"},
                "budget_yuan": {"value": 400, "evidence": "预算400元"},
            },
        )
        result = validate_recommendation_data(
            needs,
            [{"sku": "P101", "quantity": 1}, {"sku": "P202", "quantity": 1}],
        )
        self.assertEqual(result["status"], "budget_conflict")
        self.assertFalse(result["approved"])
        self.assertEqual(result["matched_rule_ids"], ["C01"])
        self.assertEqual(result["quote"]["total_yuan"], 428)
        self.assertEqual(result["quote"]["over_budget_yuan"], 28)
        rendered = render_unapproved_validation(result)
        self.assertIn("预算需要调整", rendered)
        self.assertIn("暂时不建议直接购买完整组合", rendered)
        self.assertIn("预算高 **28 元**", rendered)
        self.assertNotIn("正式推荐", rendered)

    def test_documented_combo_is_selected_from_confirmed_needs(self):
        text = "请帮我搭配一套，请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算400元。"
        needs = needs_from(
            text,
            {
                "skin_tendency": {"value": "偏干", "evidence": "偏干"},
                "sensitive_tendency": {"value": True, "evidence": "敏感倾向"},
                "goals": {"value": ["简单护理"], "evidence": "简单护理"},
                "budget_yuan": {"value": 400, "evidence": "预算400元"},
            },
        )
        selected = select_documented_combination(needs)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["rule_id"], "C01")
        self.assertEqual(
            selected["items"],
            [{"sku": "P101", "quantity": 1}, {"sku": "P202", "quantity": 1}],
        )

    def test_unsupported_budget_substitution_is_rejected(self):
        text = "请帮我搭配一套，我偏干、有敏感倾向，希望简单护理，预算400元。"
        needs = needs_from(
            text,
            {
                "skin_tendency": {"value": "偏干", "evidence": "偏干"},
                "sensitive_tendency": {"value": True, "evidence": "有敏感倾向"},
                "goals": {"value": ["简单护理"], "evidence": "简单护理"},
                "budget_yuan": {"value": 400, "evidence": "预算400元"},
            },
        )
        result = validate_recommendation_data(
            needs,
            [{"sku": "P101", "quantity": 1}, {"sku": "P203", "quantity": 1}],
        )
        self.assertEqual(result["status"], "not_supported")
        self.assertFalse(result["approved"])
        self.assertIn("没有支持该商品组合", result["violations"][0])
        rendered = render_unapproved_validation(result)
        self.assertIn("品牌资料暂时不能支持", rendered)

    def test_oily_fresh_combo_can_be_approved(self):
        text = "请帮我推荐一套，我偏油，喜欢清爽肤感，可以接受香味，预算400元，对香味不敏感。"
        needs = needs_from(
            text,
            {
                "skin_tendency": {"value": "偏油", "evidence": "偏油"},
                "goals": {"value": ["清爽肤感"], "evidence": "清爽肤感"},
                "fragrance_preference": {"value": "可接受香味", "evidence": "可以接受香味"},
                "fragrance_sensitive": {"value": False, "evidence": "对香味不敏感"},
                "budget_yuan": {"value": 400, "evidence": "预算400元"},
            },
        )
        result = validate_recommendation_data(
            needs,
            [{"sku": "P102", "quantity": 1}, {"sku": "P203", "quantity": 1}],
        )
        self.assertEqual(result["status"], "approved")
        self.assertTrue(result["approved"])
        self.assertEqual(result["quote"]["total_yuan"], 368)

    def test_p201_requires_missing_experience_and_stability(self):
        text = "我想改善粗糙，目前没有明显不适、没有受损，也不是敏感皮肤。"
        needs = needs_from(
            text,
            {
                "goals": {"value": ["改善粗糙"], "evidence": "改善粗糙"},
                "current_discomfort": {"value": False, "evidence": "没有明显不适"},
                "skin_damage": {"value": False, "evidence": "没有受损"},
                "sensitive_tendency": {"value": False, "evidence": "不是敏感皮肤"},
            },
        )
        result = validate_recommendation_data(needs, [{"sku": "P201", "quantity": 1}])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("acid_experience", result["missing_information"])
        self.assertIn("skin_state_stable", result["missing_information"])

    def test_active_discomfort_blocks_skincare_recommendation(self):
        text = "我现在有明显不适，想买果酸精华。"
        needs = needs_from(
            text,
            {
                "current_discomfort": {"value": True, "evidence": "有明显不适"},
                "desired_category": {"value": "精华", "evidence": "精华"},
            },
        )
        result = validate_recommendation_data(needs, [{"sku": "P201", "quantity": 1}])
        self.assertEqual(result["status"], "safety_blocked")
        self.assertTrue(result["professional_consultation_required"])

    def test_travel_bottle_needs_explicit_travel_need(self):
        text = "我想看看旅行配件。"
        needs = needs_from(
            text,
            {"desired_category": {"value": "旅行配件", "evidence": "旅行配件"}},
        )
        result = validate_recommendation_data(needs, [{"sku": "P301", "quantity": 1}])
        self.assertEqual(result["status"], "needs_clarification")
        self.assertIn("travel_need", result["missing_information"])

    def test_persistent_issue_blocks_skincare_but_not_internal_error(self):
        text = "这个问题持续出现，我想买洁面。"
        needs = needs_from(text, {
            "persistent_issue": {"value": True, "evidence": "持续出现"},
            "desired_category": {"value": "洁面", "evidence": "洁面"},
        })
        result = validate_recommendation_data(needs, [{"sku": "P101", "quantity": 1}])
        self.assertEqual(result["status"], "safety_blocked")

    def test_hard_no_fragrance_rejects_scented_candidate(self):
        text = "我偏油，喜欢清爽，只接受无香，只想买洁面。"
        needs = needs_from(text, {
            "selection_scope": {"value": "单品", "evidence": "只想买洁面"},
            "desired_category": {"value": "洁面", "evidence": "洁面"},
            "skin_tendency": {"value": "偏油", "evidence": "偏油"},
            "goals": {"value": ["清爽肤感"], "evidence": "喜欢清爽"},
            "fragrance_preference": {"value": "无香", "evidence": "只接受无香"},
            "fragrance_requirement": {"value": "必须无香", "evidence": "只接受无香"},
        })
        result = validate_recommendation_data(needs, [{"sku": "P102", "quantity": 1}])
        self.assertEqual(result["status"], "not_supported")
        self.assertIn("只接受无香", result["violations"][0])

    def test_empty_candidate_is_system_error_not_customer_clarification(self):
        result = validate_recommendation_data(empty_customer_needs(), [])
        self.assertEqual(result["status"], "system_error")
        self.assertIn("系统", result["next_action"])


if __name__ == "__main__":
    unittest.main()
