import unittest

from explicit_fact_parser import extract_explicit_fact_updates


class ExplicitFactParserTests(unittest.TestCase):
    def test_sample_extracts_preference_and_not_sensitive_separately(self):
        text = "我偏油，喜欢清爽肤感，可以接受香味，预算400元，对香味不敏感。"
        result = extract_explicit_fact_updates(text)
        self.assertEqual(result["skin_tendency"]["value"], "偏油")
        self.assertEqual(result["fragrance_preference"]["value"], "可接受香味")
        self.assertIs(result["fragrance_sensitive"]["value"], False)
        self.assertEqual(result["fragrance_sensitive"]["evidence"], "对香味不敏感")
        self.assertEqual(result["budget_yuan"]["value"], 400)

    def test_accepting_fragrance_does_not_infer_not_sensitive(self):
        result = extract_explicit_fact_updates("我可以接受香味。")
        self.assertEqual(result["fragrance_preference"]["value"], "可接受香味")
        self.assertNotIn("fragrance_sensitive", result)

    def test_explicit_safety_negatives_are_preserved(self):
        result = extract_explicit_fact_updates("目前没有明显不适，皮肤也没有受损。")
        self.assertIs(result["current_discomfort"]["value"], False)
        self.assertIs(result["skin_damage"]["value"], False)

    def test_uncertain_fact_is_not_forced(self):
        result = extract_explicit_fact_updates("我不确定自己是不是对香味敏感。")
        self.assertNotIn("fragrance_sensitive", result)

    def test_conflicting_values_are_not_resolved_by_rules(self):
        result = extract_explicit_fact_updates("我有时对香味敏感，有时又对香味不敏感。")
        self.assertNotIn("fragrance_sensitive", result)

    def test_negated_category_does_not_override_positive_category(self):
        result = extract_explicit_fact_updates("我不是偏干，我是偏油。")
        self.assertEqual(result["skin_tendency"]["value"], "偏油")

    def test_budget_correction_with_two_amounts_stays_for_model(self):
        result = extract_explicit_fact_updates("预算原来400元，现在调整到450元。")
        self.assertNotIn("budget_yuan", result)

    def test_explicit_sku_is_preserved_for_follow_up_planning(self):
        result = extract_explicit_fact_updates("P201适合我吗？")
        self.assertEqual(result["desired_sku"]["value"], "P201")
        self.assertEqual(result["desired_sku"]["evidence"], "P201")

    def test_multiple_compatible_goals_are_merged(self):
        result = extract_explicit_fact_updates("想要简单护理，肤感清爽轻薄。")
        self.assertEqual(
            result["goals"]["value"],
            ["简单护理", "清爽肤感", "轻薄肤感"],
        )


if __name__ == "__main__":
    unittest.main()
