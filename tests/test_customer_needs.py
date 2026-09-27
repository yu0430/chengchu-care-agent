import unittest

from customer_needs import (
    CustomerNeedsError,
    empty_customer_needs,
    merge_confirmed_needs,
    set_pending_questions,
    validate_customer_needs,
)


class CustomerNeedsTests(unittest.TestCase):
    def test_empty_state_keeps_everything_unknown(self):
        needs = validate_customer_needs(empty_customer_needs())
        self.assertIsNone(needs["skin_tendency"])
        self.assertEqual(needs["goals"], [])
        self.assertEqual(needs["evidence"], {})
        self.assertEqual(needs["budget_status"], "unknown")
        self.assertEqual(needs["pending_questions"], [])

    def test_merge_requires_exact_user_evidence(self):
        needs = merge_confirmed_needs(
            empty_customer_needs(),
            {
                "skin_tendency": {"value": "偏干", "evidence": "我偏干"},
                "budget_yuan": {"value": 400, "evidence": "预算400元"},
            },
            "我偏干，预算400元。",
        )
        self.assertEqual(needs["skin_tendency"], "偏干")
        self.assertEqual(needs["budget_yuan"], 400)
        self.assertEqual(needs["evidence"]["skin_tendency"], "我偏干")

    def test_inferred_fact_without_quote_is_rejected(self):
        with self.assertRaisesRegex(CustomerNeedsError, "未出现在"):
            merge_confirmed_needs(
                empty_customer_needs(),
                {"fragrance_preference": {"value": "无香", "evidence": "喜欢无香"}},
                "我有敏感倾向。",
            )

    def test_confirmed_value_without_evidence_is_rejected(self):
        needs = empty_customer_needs()
        needs["skin_tendency"] = "偏干"
        with self.assertRaisesRegex(CustomerNeedsError, "缺少用户原话"):
            validate_customer_needs(needs)

    def test_user_correction_overwrites_previous_value(self):
        initial = merge_confirmed_needs(
            empty_customer_needs(),
            {"sensitive_tendency": {"value": True, "evidence": "有点敏感"}},
            "我皮肤有点敏感。",
        )
        corrected = merge_confirmed_needs(
            initial,
            {"sensitive_tendency": {"value": False, "evidence": "其实不敏感"}},
            "更正一下，我其实不敏感。",
        )
        self.assertIs(corrected["sensitive_tendency"], False)
        self.assertEqual(corrected["evidence"]["sensitive_tendency"], "其实不敏感")

    def test_category_value_must_be_supported_by_category_words(self):
        with self.assertRaisesRegex(CustomerNeedsError, "不能支持该品类"):
            merge_confirmed_needs(
                empty_customer_needs(),
                {"desired_category": {"value": "洁面", "evidence": "偏油清爽"}},
                "偏油清爽",
            )

    def test_budget_status_answers_pending_budget_question(self):
        needs = set_pending_questions(empty_customer_needs(), ["budget_yuan"])
        updated = merge_confirmed_needs(
            needs,
            {
                "budget_status": {
                    "value": "unlimited",
                    "evidence": "预算不限",
                }
            },
            "预算不限",
        )
        self.assertEqual(updated["budget_status"], "unlimited")
        self.assertEqual(updated["pending_questions"], [])


if __name__ == "__main__":
    unittest.main()
