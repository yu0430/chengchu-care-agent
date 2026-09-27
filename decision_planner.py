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
        result = asdict(self)
        return result


RECOMMENDATION_HINTS = (
    "推荐", "建议", "怎么选", "选哪个", "哪个好", "适合我", "搭配", "配一套", "买什么"
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
    return bool(re.search(r"P(?:101|102|201|202|203|301)", text, re.IGNORECASE)) or any(
        term in text for term in ("多少钱", "价格", "有什么特点", "规格", "容量", "成分")
    )


def _specific_skus(text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"P(?:101|102|201|202|203|301)", text.upper())))


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
            "sensitive_tendency", "current_discomfort", "skin_damage",
            "acid_experience", "skin_state_stable", "skin_tendency", "goals", "travel_need",
        ]
        fields = [field for field in order if field in check["missing"]]
        fields.extend(field for field in check["missing"] if field not in fields)
        return DecisionPlan(
            action="ASK",
            candidate_items=[{"sku": sku, "quantity": 1}],
            question=QuestionPlan(
                fields=fields[:2],
                reason="判断指定商品是否适合前需要确认关键条件。",
            ),
            reason_codes=["SKU_NEEDS_CONFIRMATION"],
        )
    return DecisionPlan(
        action="BUILD_CANDIDATE",
        candidate_items=[{"sku": sku, "quantity": 1}],
        cautions=check["cautions"],
    )


def _safety_plan(needs: dict[str, Any], category: str | None) -> DecisionPlan | None:
    # P301 不是护肤功效产品，皮肤状态不阻止回答其客观信息或旅行携带建议。
    if category == "旅行配件":
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
    order = {"C01": 4, "C02": 3, "S01": 2, "S02": 1}
    return (
        len(item.matched_fields),
        -len(item.missing_trigger_fields),
        order.get(item.rule_id, 0),
    )


def _question_for_route(route: RuleAssessment) -> DecisionPlan | None:
    fields = list(route.missing_confirmation_fields)
    if route.rule_id == "S01":
        # 安全相关项先于使用经验；每轮最多询问两项。
        order = [
            "sensitive_tendency",
            "current_discomfort",
            "skin_damage",
            "acid_experience",
            "skin_state_stable",
        ]
        fields = [field for field in order if field in fields]
    fields = fields[:2]
    if not fields:
        return None
    return DecisionPlan(
        action="ASK",
        route_id=route.rule_id,
        question=QuestionPlan(
            fields=fields,
            route_id=route.rule_id,
            reason="正式建议前需要确认品牌资料明确要求的信息。",
        ),
        candidate_items=route.candidate_items,
        reason_codes=["MISSING_REQUIRED_CONFIRMATIONS"],
    )


def _plan_category(needs: dict[str, Any], category: str) -> DecisionPlan:
    if category == "精华":
        route = assess_route(needs, "S01")
        if route.applicability == "irrelevant" and route.missing_trigger_fields:
            return DecisionPlan(
                action="COMPARE_CATEGORY",
                category=category,
                candidate_items=[{"sku": "P201", "quantity": 1}],
                reason_codes=["ESSENCE_GOAL_NEEDS_CONFIRMATION"],
            )
        question = _question_for_route(route)
        if question:
            return question
        if route.applicability == "blocked":
            return DecisionPlan(
                action="STOP_FOR_SAFETY",
                route_id="S01",
                reason_codes=["P201_SAFETY_BOUNDARY"],
                conflicts=route.blocking_reasons,
            )
        if route.applicability == "conflicted":
            return DecisionPlan(
                action="NO_SUPPORTED_OPTION",
                route_id="S01",
                conflicts=route.conflicting_fields,
                reason_codes=["P201_CONDITIONS_NOT_MET"],
            )
        if route.applicability == "matched":
            return DecisionPlan(
                action="BUILD_CANDIDATE",
                route_id="S01",
                candidate_items=route.candidate_items,
            )
    if category == "旅行配件":
        route = assess_route(needs, "S02")
        if needs.get("travel_need") is None:
            return DecisionPlan(
                action="ASK",
                route_id="S02",
                question=QuestionPlan(
                    fields=["travel_need"],
                    route_id="S02",
                    reason="旅行携带需求明确时才讨论连带推荐。",
                ),
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
        candidate_items=[{"sku": sku, "quantity": 1} for sku in proposal.alternatives],
        reason_codes=["CATEGORY_NEEDS_PRODUCT_COMPARISON"],
    )


def plan_next_action(needs: dict[str, Any], latest_user_text: str) -> dict[str, Any]:
    """返回本轮唯一动作；不会提交空商品列表，也不会计算价格。"""
    text = latest_user_text.strip()
    topics = _policy_topics(text)
    if topics:
        return DecisionPlan(
            action="ANSWER_POLICY",
            policy_topic=topics[0],
            policy_topics=topics,
        ).to_dict()

    pending = needs.get("pending_questions") or []
    evidence = needs.get("evidence") or {}
    profile_reply = any(
        field != "desired_sku" and isinstance(quote, str) and quote and quote in text
        for field, quote in evidence.items()
    )
    recommendation_requested = (
        _asks_recommendation(text)
        or bool(pending)
        or (needs.get("desired_sku") is not None and profile_reply)
    )
    if _asks_fact(text) and not recommendation_requested:
        return DecisionPlan(action="ANSWER_FACT").to_dict()

    category = needs.get("desired_category")
    safety = _safety_plan(needs, category)
    if safety:
        return safety.to_dict()

    skus = _specific_skus(text)
    if not skus and recommendation_requested and needs.get("desired_sku"):
        skus = [needs["desired_sku"]]
    if recommendation_requested and len(skus) == 1:
        return _plan_specific_sku(needs, skus[0]).to_dict()

    if category:
        return _plan_category(needs, category).to_dict()

    scope = needs.get("selection_scope")
    routes = [
        item
        for item in assess_routes(needs)
        if item.kind == "combination" and item.applicability != "irrelevant"
    ]
    viable = [item for item in routes if item.applicability not in {"conflicted", "blocked"}]
    if viable:
        route = max(viable, key=_route_priority)
        # 只有触发条件全部成立后才询问该组合的预算、香味等确认项。
        if not route.missing_trigger_fields:
            question = _question_for_route(route)
            if question:
                return question.to_dict()
            if scope is None:
                scope_fields = ["selection_scope"]
                if route.rule_id == "C02" and needs.get("fragrance_sensitive") is None:
                    scope_fields.append("fragrance_sensitive")
                return DecisionPlan(
                    action="ASK",
                    route_id=route.rule_id,
                    candidate_items=route.candidate_items,
                    question=QuestionPlan(
                        fields=scope_fields,
                        route_id=route.rule_id,
                        reason="提出两件组合前需要尊重顾客的购买范围。",
                    ),
                    reason_codes=["SCOPE_REQUIRED_BEFORE_COMBINATION"],
                ).to_dict()
            if scope == "完整护理":
                if route.rule_id == "C02" and needs.get("fragrance_sensitive") is None:
                    return DecisionPlan(
                        action="ASK",
                        route_id=route.rule_id,
                        candidate_items=route.candidate_items,
                        question=QuestionPlan(
                            fields=["fragrance_sensitive"],
                            route_id=route.rule_id,
                            reason="带香候选正式建议前需要确认香味是否会引起不适。",
                        ),
                        reason_codes=["FRAGRANCE_SENSITIVITY_AFFECTS_PRIORITY"],
                    ).to_dict()
                return DecisionPlan(
                    action="BUILD_CANDIDATE",
                    route_id=route.rule_id,
                    candidate_items=route.candidate_items,
                    cautions=route.not_preferred_reasons,
                ).to_dict()

    conflicted = [item for item in routes if item.applicability == "conflicted"]
    if conflicted and recommendation_requested:
        route = max(conflicted, key=_route_priority)
        return DecisionPlan(
            action="NO_SUPPORTED_OPTION",
            route_id=route.rule_id,
            conflicts=route.conflicting_fields,
            reason_codes=["ROUTE_CONFLICTS_WITH_CONFIRMED_NEEDS"],
        ).to_dict()

    profile_present = any(
        needs.get(field) is not None
        for field in ("skin_tendency", "sensitive_tendency")
    ) or bool(needs.get("goals"))
    if recommendation_requested or profile_present:
        return DecisionPlan(
            action="ASK",
            question=QuestionPlan(
                fields=["selection_scope"],
                reason="需要先确定想看指定品类还是完整护理。",
            ),
            reason_codes=["SELECTION_SCOPE_REQUIRED"],
        ).to_dict()

    if _asks_fact(text):
        return DecisionPlan(action="ANSWER_FACT").to_dict()
    return DecisionPlan(action="CHAT").to_dict()


__all__ = ["DecisionPlan", "QuestionPlan", "plan_next_action"]
