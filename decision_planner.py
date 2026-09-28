"""根据已确认需求生成唯一、可解释的下一步动作。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from candidate_builder import proposal_for_category
from rule_engine import (
    PRODUCT_BY_SKU,
    RuleAssessment,
    assess_route,
    assess_routes,
    evaluate_machine_conditions,
)


PlanAction = Literal[
    "ASK",
    "ANSWER_FACT",
    "ANSWER_POLICY",
    "COMPARE_CATEGORY",
    "BUILD_CANDIDATE",
    "STOP_FOR_SAFETY",
    "NO_SUPPORTED_OPTION",
    "CHAT",
]


@dataclass(frozen=True)
class QuestionPlan:
    fields: list[str]
    route_id: str | None = None
    question_id: str | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionPlan:
    action: PlanAction
    route_id: str | None = None
    candidate_items: list[dict[str, Any]] = field(default_factory=list)
    question: QuestionPlan | None = None
    category: str | None = None
    policy_topic: str | None = None
    policy_topics: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RECOMMENDATION_HINTS = (
    "推荐", "建议", "怎么选", "如何选", "选哪个", "哪个好", "适合",
    "搭配", "配一套", "买什么", "想买", "想看", "帮我选", "怎么搭",
)

POLICY_TOPICS = {
    "促销": ("促销", "活动价", "打折", "优惠"),
    "赠品": ("赠品", "送什么", "礼品"),
    "线上同价": ("线上同价", "网上价格", "网上更便宜", "线上价格"),
    "退换货": ("退换货", "退货", "换货"),
    "渠道授权": ("渠道授权", "授权店", "官方授权", "是不是正品"),
}


def _policy_topics(text: str) -> list[str]:
    return [
        topic
        for topic, terms in POLICY_TOPICS.items()
        if any(term in text for term in terms)
    ]


def _asks_recommendation(text: str) -> bool:
    return any(term in text for term in RECOMMENDATION_HINTS)


def _asks_fact(text: str) -> bool:
    has_sku = bool(
        re.search(r"P(?:101|102|201|202|203|301)", text, re.IGNORECASE)
    )
    fact_words = ("多少钱", "价格", "有什么特点", "规格", "容量", "成分", "是什么")
    return has_sku or any(term in text for term in fact_words)


def _specific_skus(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            re.findall(r"P(?:101|102|201|202|203|301)", text.upper())
        )
    )


def _ask(
    fields: list[str],
    *,
    route_id: str | None = None,
    question_id: str | None = None,
    reason: str,
    candidate_items: list[dict[str, Any]] | None = None,
    reason_codes: list[str] | None = None,
) -> DecisionPlan:
    return DecisionPlan(
        action="ASK",
        route_id=route_id,
        candidate_items=candidate_items or [],
        question=QuestionPlan(
            fields=list(dict.fromkeys(fields)),
            route_id=route_id,
            question_id=question_id,
            reason=reason,
        ),
        reason_codes=reason_codes or ["MISSING_REQUIRED_INFORMATION"],
    )


def _plan_specific_sku(needs: dict[str, Any], sku: str) -> DecisionPlan:
    product = PRODUCT_BY_SKU[sku]
    if (
        needs.get("fragrance_requirement") == "必须无香"
        and product.get("derived_tags", {}).get("fragrance") not in {None, "none"}
    ):
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            candidate_items=[{"sku": sku, "quantity": 1}],
            conflicts=["fragrance_requirement"],
            reason_codes=["SKU_CONFLICTS_WITH_HARD_FRAGRANCE_REQUIREMENT"],
        )

    check = evaluate_machine_conditions(
        needs, product.get("recommendation_conditions") or {}
    )
    if check["blocked"]:
        return DecisionPlan(
            action="STOP_FOR_SAFETY",
            conflicts=check["blocked"],
            reason_codes=["SKU_SAFETY_BOUNDARY"],
        )
    if check["violations"]:
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            candidate_items=[{"sku": sku, "quantity": 1}],
            conflicts=check["violations"],
            reason_codes=["SKU_CONDITIONS_NOT_MET"],
        )
    if check["missing"]:
        order = [
            "sensitive_tendency",
            "skin_damage",
            "current_discomfort",
            "acid_experience",
            "skin_state_stable",
            "skin_tendency",
            "goals",
            "travel_need",
        ]
        fields = [field for field in order if field in check["missing"]]
        fields.extend(field for field in check["missing"] if field not in fields)
        if sku == "P201":
            safety = [
                field
                for field in (
                    "sensitive_tendency",
                    "skin_damage",
                    "current_discomfort",
                )
                if field in fields
            ]
            if safety:
                return _ask(
                    safety,
                    route_id="S01",
                    question_id="p201_skin_safety",
                    reason="讨论果酸精华前需要确认资料明确列出的安全边界。",
                    candidate_items=[{"sku": sku, "quantity": 1}],
                    reason_codes=["SKU_NEEDS_CONFIRMATION"],
                )
            experience = [
                field
                for field in ("acid_experience", "skin_state_stable")
                if field in fields
            ]
            if experience:
                return _ask(
                    experience,
                    route_id="S01",
                    question_id="p201_experience_state",
                    reason="讨论果酸精华前需要确认使用经验和当前状态。",
                    candidate_items=[{"sku": sku, "quantity": 1}],
                    reason_codes=["SKU_NEEDS_CONFIRMATION"],
                )
        return _ask(
            fields[:2],
            route_id="S01" if sku == "P201" else None,
            question_id="s01_trigger" if sku == "P201" else f"{sku.lower()}_confirmation",
            reason="判断指定商品是否适合前需要确认关键条件。",
            candidate_items=[{"sku": sku, "quantity": 1}],
            reason_codes=["SKU_NEEDS_CONFIRMATION"],
        )
    return DecisionPlan(
        action="BUILD_CANDIDATE",
        candidate_items=[{"sku": sku, "quantity": 1}],
        cautions=check["cautions"],
    )


def _safety_plan(
    needs: dict[str, Any], *, travel_context: bool = False
) -> DecisionPlan | None:
    # P301 不是护肤功效产品，皮肤状态不阻止回答其资料或旅行携带建议。
    if travel_context:
        return None
    reasons: list[str] = []
    if needs.get("current_discomfort") is True:
        reasons.append("CURRENT_DISCOMFORT")
    if needs.get("skin_damage") is True:
        reasons.append("SKIN_DAMAGE")
    if needs.get("persistent_issue") is True:
        reasons.append("PERSISTENT_ISSUE")
    if reasons:
        return DecisionPlan(action="STOP_FOR_SAFETY", reason_codes=reasons)
    return None


def _route_priority(item: RuleAssessment) -> tuple[int, int, int]:
    # 场景与组合都参与一级路由。优先选择已明确命中字段更多、缺失更少的路线。
    order = {"S01": 4, "C01": 3, "C02": 3, "S02": 2}
    return (
        len(item.matched_fields),
        -len(item.missing_trigger_fields),
        order.get(item.rule_id, 0),
    )


def _question_for_route(route: RuleAssessment) -> DecisionPlan | None:
    if route.missing_trigger_fields:
        trigger_order = {
            "C01": ["skin_tendency", "sensitive_tendency", "goals"],
            "C02": ["skin_tendency", "goals"],
            "S01": ["goals"],
            "S02": ["travel_need"],
        }
        fields = [
            field
            for field in trigger_order.get(route.rule_id, [])
            if field in route.missing_trigger_fields
        ]
        fields.extend(
            field
            for field in route.missing_trigger_fields
            if field not in fields
        )
        if fields:
            return _ask(
                fields[:2],
                route_id=route.rule_id,
                question_id=f"{route.rule_id.lower()}_trigger",
                reason="需要补充能够区分资料路线的关键信息。",
                candidate_items=route.candidate_items,
                reason_codes=["MISSING_ROUTE_TRIGGERS"],
            )

    missing = list(route.missing_confirmation_fields)
    if not missing:
        return None

    if route.rule_id == "S01":
        safety = [
            field
            for field in (
                "sensitive_tendency",
                "skin_damage",
                "current_discomfort",
            )
            if field in missing
        ]
        if safety:
            return _ask(
                safety,
                route_id="S01",
                question_id="p201_skin_safety",
                reason="讨论果酸精华前需要确认敏感、受损和当前不适情况。",
                candidate_items=route.candidate_items,
                reason_codes=["MISSING_REQUIRED_CONFIRMATIONS"],
            )
        experience = [
            field
            for field in ("acid_experience", "skin_state_stable")
            if field in missing
        ]
        return _ask(
            experience[:2],
            route_id="S01",
            question_id="p201_experience_state",
            reason="讨论果酸精华前需要确认使用经验和当前状态。",
            candidate_items=route.candidate_items,
            reason_codes=["MISSING_REQUIRED_CONFIRMATIONS"],
        )

    question_id = {
        "C01": "c01_budget",
        "C02": "c02_budget_fragrance",
        "S02": "s02_travel",
    }.get(route.rule_id)
    return _ask(
        missing[:2],
        route_id=route.rule_id,
        question_id=question_id,
        reason="正式建议前需要确认品牌资料明确要求的信息。",
        candidate_items=route.candidate_items,
        reason_codes=["MISSING_REQUIRED_CONFIRMATIONS"],
    )


def _single_category_question(route_id: str | None = None) -> DecisionPlan:
    return _ask(
        ["desired_category"],
        route_id=route_id,
        question_id="single_category",
        reason="你已经明确想先看单品，需要确认想看的品类。",
        reason_codes=["SINGLE_CATEGORY_REQUIRED"],
    )


def _plan_category(needs: dict[str, Any], category: str) -> DecisionPlan:
    if category == "精华":
        route = assess_route(needs, "S01")
        if route.applicability == "blocked":
            return DecisionPlan(
                action="NO_SUPPORTED_OPTION",
                route_id="S01",
                candidate_items=route.candidate_items,
                conflicts=route.blocking_reasons,
                reason_codes=["P201_NOT_RECOMMENDED_FOR_CONFIRMED_STATE"],
            )
        if route.applicability == "conflicted":
            return DecisionPlan(
                action="NO_SUPPORTED_OPTION",
                route_id="S01",
                candidate_items=route.candidate_items,
                conflicts=route.conflicting_fields,
                reason_codes=["P201_CONDITIONS_NOT_MET"],
            )
        question = _question_for_route(route)
        if question:
            return question
        if route.applicability == "matched":
            return DecisionPlan(
                action="BUILD_CANDIDATE",
                route_id="S01",
                candidate_items=route.candidate_items,
            )
        return _ask(
            ["goals"],
            route_id="S01",
            question_id="s01_trigger",
            reason="需要先确认希望改善的方向。",
            candidate_items=route.candidate_items,
        )

    if category == "旅行配件":
        route = assess_route(needs, "S02")
        if needs.get("travel_need") is None:
            return _ask(
                ["travel_need"],
                route_id="S02",
                question_id="s02_travel",
                reason="旅行携带需求明确时才讨论连带推荐。",
                candidate_items=route.candidate_items,
                reason_codes=["TRAVEL_NEED_REQUIRED"],
            )
        if needs.get("travel_need") is True:
            return DecisionPlan(
                action="BUILD_CANDIDATE",
                route_id="S02",
                candidate_items=route.candidate_items,
            )
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            route_id="S02",
            reason_codes=["NO_TRAVEL_NEED"],
        )

    proposal = proposal_for_category(needs, category)
    if proposal.status == "ready":
        return DecisionPlan(
            action="BUILD_CANDIDATE",
            candidate_items=proposal.items,
            category=category,
        )
    if proposal.status == "none":
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            category=category,
            reason_codes=["NO_CATEGORY_PRODUCT_MATCH"],
        )
    return DecisionPlan(
        action="COMPARE_CATEGORY",
        category=category,
        candidate_items=[
            {"sku": sku, "quantity": 1}
            for sku in proposal.alternatives
        ],
        reason_codes=["CATEGORY_NEEDS_PRODUCT_COMPARISON"],
    )


def plan_next_action(
    needs: dict[str, Any],
    latest_user_text: str,
    active_route_id: str | None = None,
) -> dict[str, Any]:
    """返回本轮唯一动作；不提交空商品列表，也不计算价格。"""
    text = latest_user_text.strip()

    topics = _policy_topics(text)
    if topics:
        return DecisionPlan(
            action="ANSWER_POLICY",
            policy_topic=topics[0],
            policy_topics=topics,
        ).to_dict()

    explicit_recommendation = _asks_recommendation(text)
    if _asks_fact(text) and not explicit_recommendation:
        return DecisionPlan(action="ANSWER_FACT").to_dict()

    pending = needs.get("pending_questions") or []
    profile_present = any(
        needs.get(field) is not None
        for field in (
            "skin_tendency",
            "sensitive_tendency",
            "acid_experience",
            "skin_state_stable",
            "travel_need",
        )
    ) or bool(needs.get("goals"))
    recommendation_requested = bool(
        explicit_recommendation or pending or active_route_id or profile_present
    )

    skus = _specific_skus(text)
    if (
        not skus
        and recommendation_requested
        and needs.get("desired_sku")
    ):
        skus = [needs["desired_sku"]]

    category = needs.get("desired_category")
    travel_context = bool(
        category == "旅行配件"
        or skus == ["P301"]
        or active_route_id == "S02"
        or needs.get("travel_need") is True
    )
    safety = _safety_plan(needs, travel_context=travel_context)
    if safety:
        return safety.to_dict()

    if recommendation_requested and len(skus) == 1:
        return _plan_specific_sku(needs, skus[0]).to_dict()

    if category:
        return _plan_category(needs, category).to_dict()

    scope = needs.get("selection_scope")
    if scope == "单品":
        return _single_category_question(active_route_id).to_dict()

    assessments = assess_routes(needs)
    eligible = [
        item
        for item in assessments
        if item.applicability in {"possible", "matched"}
    ]

    route: RuleAssessment | None = None
    if active_route_id:
        route = next(
            (
                item
                for item in eligible
                if item.rule_id == active_route_id
            ),
            None,
        )
    if route is None and eligible:
        route = max(eligible, key=_route_priority)

    if route is not None:
        question = _question_for_route(route)
        if question:
            return question.to_dict()
        if route.applicability == "matched":
            return DecisionPlan(
                action="BUILD_CANDIDATE",
                route_id=route.rule_id,
                candidate_items=route.candidate_items,
                cautions=route.not_preferred_reasons,
            ).to_dict()

    # P201 的明确安全冲突只排除 P201，不误导为全品牌无可选方案。
    blocked_s01 = next(
        (
            item
            for item in assessments
            if item.rule_id == "S01"
            and item.applicability == "blocked"
            and item.matched_fields
        ),
        None,
    )
    if blocked_s01:
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            route_id="S01",
            candidate_items=blocked_s01.candidate_items,
            conflicts=blocked_s01.blocking_reasons,
            reason_codes=["P201_NOT_RECOMMENDED_FOR_CONFIRMED_STATE"],
        ).to_dict()

    # 明确的无香硬要求与 C02 带香组合冲突时，说明边界并提供转向品类的入口。
    fragrance_conflict = next(
        (
            item
            for item in assessments
            if item.rule_id == "C02"
            and "fragrance_requirement" in item.conflicting_fields
            and len(item.matched_fields) >= 2
        ),
        None,
    )
    if fragrance_conflict:
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            route_id="C02",
            conflicts=["fragrance_requirement"],
            reason_codes=["ROUTE_CONFLICTS_WITH_HARD_FRAGRANCE_REQUIREMENT"],
        ).to_dict()

    if recommendation_requested or profile_present:
        if needs.get("skin_tendency") is None and not needs.get("goals"):
            return _ask(
                ["skin_tendency", "goals"],
                question_id="general_profile",
                reason="需要先了解肤质和护理目标。",
                reason_codes=["PROFILE_REQUIRED"],
            ).to_dict()
        if needs.get("skin_tendency") is None:
            return _ask(
                ["skin_tendency"],
                question_id="general_skin",
                reason="需要先了解肤质。",
                reason_codes=["SKIN_TENDENCY_REQUIRED"],
            ).to_dict()
        if not needs.get("goals"):
            return _ask(
                ["goals"],
                question_id="general_goals",
                reason="需要先了解护理目标。",
                reason_codes=["GOAL_REQUIRED"],
            ).to_dict()
        return _single_category_question().to_dict()

    return DecisionPlan(action="CHAT").to_dict()


__all__ = ["DecisionPlan", "QuestionPlan", "plan_next_action"]
