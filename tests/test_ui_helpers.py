"""UI 展示辅助函数测试；不启动 Streamlit，也不调用 Qwen。"""

import unittest

from langchain_core.messages import AIMessage, HumanMessage

from ui_helpers import (
    allowed_conclusion,
    customer_answer_text,
    friendly_tool_trace,
    needs_details,
    needs_summary,
    public_text,
    source_label,
    validation_checks,
    validation_meta,
)


class UIHelpersTests(unittest.TestCase):
    def test_customer_answer_hides_internal_rule_id_but_keeps_sku(self):
        answer = customer_answer_text("标准护理组合（C02），商品 P102 + P203。")
        self.assertNotIn("C02", answer)
        self.assertIn("P102", answer)
        self.assertIn("P203", answer)

    def test_exam_filename_is_hidden(self):
        self.assertEqual(
            public_text("来源：08题 澄初个人护理.pdf"),
            "来源：澄初品牌资料",
        )

    def test_source_label_uses_public_brand_name(self):
        source = {
            "document": "08题 澄初个人护理.pdf",
            "section": "三、组合与边界",
            "locator": "P101 + P202",
        }
        self.assertEqual(
            source_label(source),
            "澄初品牌资料 / 三、组合与边界 / P101 + P202",
        )

    def test_needs_summary_only_contains_confirmed_values(self):
        needs = {
            "skin_tendency": "偏干",
            "sensitive_tendency": True,
            "budget_yuan": 400,
            "fragrance_preference": None,
            "goals": ["简单护理"],
            "evidence": {},
        }
        summary = needs_summary(needs)
        self.assertEqual(summary, "偏干 · 有敏感倾向 · 简单护理 · 400 元")
        self.assertNotIn("香味", summary)

    def test_needs_details_keep_user_evidence(self):
        needs = {
            "skin_tendency": "偏干",
            "evidence": {"skin_tendency": "我偏干"},
        }
        rows = needs_details(needs)
        self.assertEqual(rows[0]["label"], "肤质倾向")
        self.assertEqual(rows[0]["evidence"], "我偏干")

    def test_selection_scope_is_visible_in_confirmed_needs(self):
        rows = needs_details({
            "selection_scope": "单品",
            "evidence": {"selection_scope": "只想买洁面"},
        })
        self.assertEqual(rows[0]["label"], "选择范围")
        self.assertEqual(rows[0]["value"], "单品")

    def test_not_preferred_has_non_safety_status(self):
        meta = validation_meta({"status": "not_preferred"})
        self.assertEqual(meta["title"], "当前方案不是首选")
        self.assertEqual(meta["tone"], "warning")

    def test_budget_conflict_uses_user_facing_status(self):
        meta = validation_meta({"status": "budget_conflict"})
        self.assertEqual(meta["title"], "预算需要调整")
        self.assertEqual(meta["tone"], "warning")

    def test_waiting_for_answers_is_not_presented_as_failure(self):
        meta = validation_meta({"status": "awaiting_information"})
        self.assertEqual(meta["title"], "正在了解你的需求")
        self.assertEqual(meta["tone"], "info")

    def test_system_error_does_not_blame_customer_wording(self):
        meta = validation_meta({"status": "system_error"})
        self.assertEqual(meta["title"], "本次核对暂未完成")
        self.assertIn("系统", allowed_conclusion({"status": "system_error"}))

    def test_new_safety_and_preference_fields_are_visible(self):
        rows = needs_details({
            "persistent_issue": True,
            "fragrance_requirement": "必须无香",
            "budget_status": "unlimited",
            "evidence": {
                "persistent_issue": "反复出现",
                "fragrance_requirement": "只接受无香",
                "budget_status": "预算不限",
            },
        })
        values = {row["label"]: row["value"] for row in rows}
        self.assertEqual(values["持续或反复问题"], "问题持续或反复")
        self.assertEqual(values["香味要求"], "必须无香")
        self.assertEqual(values["预算状态"], "预算不限")

    def test_budget_conflict_audit_explains_allowed_conclusion(self):
        validation = {
            "status": "budget_conflict",
            "approved": False,
            "matched_rule_ids": ["C01"],
            "confirmed_needs": {
                "current_discomfort": False,
                "skin_damage": False,
                "evidence": {"skin_tendency": "我偏干", "budget_yuan": "预算400元"},
            },
            "quote": {
                "status": "success",
                "total_yuan": 428,
                "within_budget": False,
                "over_budget_yuan": 28,
            },
        }
        texts = [item["text"] for item in validation_checks(validation)]
        self.assertTrue(any("规则 C01" in text for text in texts))
        self.assertTrue(any("超过当前预算 28 元" in text for text in texts))
        self.assertIn("暂不建议购买完整组合", allowed_conclusion(validation))

    def test_trace_only_uses_latest_user_turn(self):
        messages = [
            HumanMessage(content="第一轮"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_policy",
                        "args": {"topic": "赠品"},
                        "id": "old-call",
                        "type": "tool_call",
                    }
                ],
            ),
            HumanMessage(content="第二轮"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_products",
                        "args": {"category": "洁面"},
                        "id": "new-call",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
        self.assertEqual(friendly_tool_trace(messages), ["查询商品资料：洁面"])


if __name__ == "__main__":
    unittest.main()
