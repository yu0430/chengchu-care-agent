"""不调用大模型的整段多轮业务回归。"""

import unittest

from candidate_builder import build_candidate
from customer_needs import empty_customer_needs, merge_confirmed_needs, set_pending_questions
from decision_planner import plan_next_action
from explicit_fact_parser import (
    extract_explicit_fact_updates,
    extract_pending_question_updates,
)
from recommendation_guard import validate_recommendation_data
from recommendation_validator import validate_candidate_data
from rule_engine import assess_route


def advance(needs, text):
    updates = extract_explicit_fact_updates(text)
    for field, update in extract_pending_question_updates(
        text, needs.get("pending_questions") or []
    ).items():
        updates.setdefault(field, update)
    needs = merge_confirmed_needs(needs, updates, text)
    plan = plan_next_action(needs, text)
    if plan["action"] == "ASK":
        needs = set_pending_questions(needs, plan["question"]["fields"])
    else:
        needs = set_pending_questions(needs, [])
    return needs, plan


class DecisionFlowTests(unittest.TestCase):
    def test_compact_oily_fresh_phrase_enters_c02_questions(self):
        _needs, plan = advance(empty_customer_needs(), "偏油清爽，推荐什么？")
        self.assertEqual(plan["route_id"], "C02")
        self.assertEqual(plan["question"]["fields"], ["budget_yuan", "fragrance_preference"])

    def test_oily_fresh_route_asks_in_stages_then_builds_combo(self):
        needs, plan = advance(empty_customer_needs(), "我偏油，喜欢清爽，推荐什么？")
        self.assertEqual(plan["action"], "ASK")
        self.assertEqual(plan["question"]["fields"], ["budget_yuan", "fragrance_preference"])

        needs, plan = advance(needs, "400")
        self.assertEqual(needs["budget_yuan"], 400)
        self.assertEqual(plan["question"]["fields"], ["fragrance_preference"])

        needs, plan = advance(needs, "可以")
        self.assertEqual(needs["fragrance_preference"], "可接受香味")
        self.assertEqual(
            plan["question"]["fields"], ["selection_scope", "fragrance_sensitive"]
        )

        needs, plan = advance(needs, "一起，对香味不敏感")
        self.assertEqual(plan["action"], "BUILD_CANDIDATE")
        proposal = build_candidate(plan, needs)
        self.assertEqual(
            proposal.items,
            [{"sku": "P102", "quantity": 1}, {"sku": "P203", "quantity": 1}],
        )
        validation = validate_candidate_data(needs, proposal.items)
        self.assertTrue(validation["candidate_valid"])
        final = validate_recommendation_data(needs, proposal.items)
        self.assertTrue(final["approved"])
        self.assertEqual(final["quote"]["total_yuan"], 368)
        self.assertTrue(final["quote"]["within_budget"])

    def test_dry_sensitive_flow_reaches_exact_budget_conflict(self):
        needs, plan = advance(
            empty_customer_needs(),
            "我偏干，有敏感倾向，希望简单护理，推荐什么？",
        )
        self.assertEqual(plan["question"]["fields"], ["budget_yuan"])
        needs, plan = advance(needs, "400")
        self.assertEqual(plan["question"]["fields"], ["selection_scope"])
        needs, plan = advance(needs, "一起")
        self.assertEqual(plan["action"], "BUILD_CANDIDATE")
        proposal = build_candidate(plan, needs)
        final = validate_recommendation_data(needs, proposal.items)
        self.assertEqual(final["status"], "budget_conflict")
        self.assertEqual(final["quote"]["total_yuan"], 428)
        self.assertEqual(final["quote"]["over_budget_yuan"], 28)

    def test_hard_no_fragrance_excludes_oily_combo_before_budget_question(self):
        needs, plan = advance(
            empty_customer_needs(), "我偏油，喜欢清爽，只接受无香，推荐什么？"
        )
        self.assertEqual(needs["fragrance_requirement"], "必须无香")
        self.assertEqual(plan["action"], "NO_SUPPORTED_OPTION")
        self.assertNotIn("budget_yuan", (plan.get("question") or {}).get("fields", []))

    def test_persistent_issue_stops_skincare_recommendation(self):
        needs, plan = advance(
            empty_customer_needs(), "这个问题反复出现，我想买精华。"
        )
        self.assertIs(needs["persistent_issue"], True)
        self.assertEqual(plan["action"], "STOP_FOR_SAFETY")

    def test_safety_state_still_allows_objective_price_question(self):
        needs, _plan = advance(empty_customer_needs(), "这个问题持续出现，想买洁面。")
        _needs, plan = advance(needs, "P101 多少钱？")
        self.assertEqual(plan["action"], "ANSWER_FACT")

    def test_budget_unlimited_is_confirmed_and_not_reasked(self):
        needs, plan = advance(
            empty_customer_needs(),
            "偏油，喜欢清爽，预算不限，可以接受香味，对香味不敏感，推荐一套。",
        )
        self.assertEqual(needs["budget_status"], "unlimited")
        self.assertEqual(plan["action"], "BUILD_CANDIDATE")

    def test_ambiguous_yes_does_not_answer_two_pending_questions(self):
        needs = set_pending_questions(
            empty_customer_needs(), ["budget_yuan", "fragrance_preference"]
        )
        updates = extract_pending_question_updates("可以", needs["pending_questions"])
        self.assertEqual(updates, {})

    def test_staged_assessment_distinguishes_possible_and_matched(self):
        needs, _plan = advance(empty_customer_needs(), "偏油，喜欢清爽，推荐什么？")
        assessment = assess_route(needs, "C02")
        self.assertEqual(assessment.applicability, "possible")
        self.assertEqual(
            assessment.missing_confirmation_fields,
            ["budget_yuan", "fragrance_preference"],
        )

    def test_policy_question_uses_policy_path(self):
        _needs, plan = advance(empty_customer_needs(), "网上价格更便宜吗？")
        self.assertEqual(plan["action"], "ANSWER_POLICY")
        self.assertEqual(plan["policy_topic"], "线上同价")

    def test_multiple_policy_topics_are_all_preserved(self):
        _needs, plan = advance(empty_customer_needs(), "网上价格更便宜吗？有赠品吗？")
        self.assertEqual(plan["action"], "ANSWER_POLICY")
        self.assertEqual(plan["policy_topics"], ["赠品", "线上同价"])

    def test_specific_sku_suitability_uses_question_plan(self):
        needs, plan = advance(empty_customer_needs(), "P201 适合我吗？")
        self.assertEqual(plan["action"], "ASK")
        self.assertLessEqual(len(plan["question"]["fields"]), 2)
        self.assertIn("sensitive_tendency", plan["question"]["fields"])

        needs, plan = advance(needs, "我不是敏感肌，目前没有明显不适")
        self.assertEqual(plan["action"], "ASK")
        needs, plan = advance(needs, "没有受损，有焕肤经验，状态稳定")
        self.assertEqual(plan["question"]["fields"], ["goals"])
        needs, plan = advance(needs, "想改善粗糙")
        self.assertEqual(plan["action"], "BUILD_CANDIDATE")
        self.assertEqual(plan["candidate_items"], [{"sku": "P201", "quantity": 1}])


if __name__ == "__main__":
    unittest.main()
