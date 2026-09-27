import unittest

from customer_needs import CustomerNeedsError, empty_customer_needs
from needs_extractor import (
    EXTRACTION_PROMPT,
    NeedUpdate,
    NeedsExtraction,
    extract_confirmed_needs,
)


class FakeExtractor:
    def __init__(self, extraction):
        self.extraction = extraction

    def invoke(self, _messages):
        return self.extraction


class NeedsExtractorTests(unittest.TestCase):
    def test_sparse_updates_are_merged(self):
        extractor = FakeExtractor(
            NeedsExtraction(
                updates=[
                    NeedUpdate(field="skin_tendency", value="偏干", evidence="我偏干"),
                    NeedUpdate(field="budget_yuan", value=400, evidence="预算400元"),
                ]
            )
        )
        result = extract_confirmed_needs(
            extractor, empty_customer_needs(), "我偏干，预算400元。"
        )
        self.assertEqual(result["skin_tendency"], "偏干")
        self.assertEqual(result["budget_yuan"], 400)
        self.assertIsNone(result["fragrance_preference"])

    def test_rule_channel_fills_explicit_false_missed_by_model(self):
        extractor = FakeExtractor(
            NeedsExtraction(
                updates=[
                    NeedUpdate(
                        field="fragrance_preference",
                        value="可接受香味",
                        evidence="可以接受香味",
                    )
                ]
            )
        )
        text = "我偏油，喜欢清爽肤感，可以接受香味，预算400元，对香味不敏感。"
        result = extract_confirmed_needs(extractor, empty_customer_needs(), text)
        self.assertEqual(result["fragrance_preference"], "可接受香味")
        self.assertIs(result["fragrance_sensitive"], False)
        self.assertEqual(result["evidence"]["fragrance_sensitive"], "对香味不敏感")

    def test_rule_channel_explicit_value_overrides_wrong_model_value(self):
        extractor = FakeExtractor(
            NeedsExtraction(
                updates=[
                    NeedUpdate(
                        field="fragrance_sensitive",
                        value=True,
                        evidence="对香味不敏感",
                    )
                ]
            )
        )
        result = extract_confirmed_needs(
            extractor, empty_customer_needs(), "我对香味不敏感。"
        )
        self.assertIs(result["fragrance_sensitive"], False)

    def test_inferred_evidence_is_rejected_by_business_layer(self):
        extractor = FakeExtractor(
            NeedsExtraction(
                updates=[
                    NeedUpdate(
                        field="fragrance_preference",
                        value="无香",
                        evidence="喜欢无香",
                    )
                ]
            )
        )
        with self.assertRaisesRegex(CustomerNeedsError, "未出现在"):
            extract_confirmed_needs(
                extractor, empty_customer_needs(), "我有敏感倾向。"
            )

    def test_empty_extraction_preserves_state(self):
        current = empty_customer_needs()
        extractor = FakeExtractor(NeedsExtraction(updates=[]))
        self.assertEqual(
            extract_confirmed_needs(extractor, current, "你好"), current
        )

    def test_prompt_prohibits_sensitive_inference(self):
        self.assertIn("不能推断", EXTRACTION_PROMPT)
        self.assertIn("evidence 必须逐字复制", EXTRACTION_PROMPT)
        self.assertIn("预算有限", EXTRACTION_PROMPT)


if __name__ == "__main__":
    unittest.main()
