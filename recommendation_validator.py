"""正式候选的确定性适用性与安全校验，不承担报价。"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from customer_needs import CustomerNeedsError, validate_customer_needs
from rule_engine import PRODUCT_BY_SKU, POLICIES_DATA, evaluate_machine_conditions


def _result(status: str, **values: Any) -> dict[str, Any]:
    result = {
        "status": status,
        "approved": False,
        "candidate_valid": False,
        "confirmed_needs": None,
        "items": [],
        "matched_rule_ids": [],
        "missing_information": [],
        "violations": [],
        "not_preferred_reasons": [],
        "cautions": [],
        "professional_consultation_required": False,
        "next_action": "",
    }
    result.update(values)
    return result


def normalize_candidate_items(
    items: Any,
) -> tuple[list[dict[str, int | str]] | None, str | None]:
    if not isinstance(items, list) or not items:
        return None, "候选生成失败：没有可校验的商品"
    quantities: dict[str, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return None, f"候选商品 {index + 1} 的格式无效"
        sku = item.get("sku")
        if not isinstance(sku, str) or not sku.strip():
            return None, f"候选商品 {index + 1} 缺少 SKU"
        normalized_sku = sku.strip().upper()
        if normalized_sku not in PRODUCT_BY_SKU:
            return None, f"品牌资料中没有商品 {normalized_sku}"
        quantity = item.get("quantity", 1)
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            return None, f"{normalized_sku} 的数量必须是正整数"
        quantities[normalized_sku] = quantities.get(normalized_sku, 0) + quantity
    return [
        {"sku": sku, "quantity": quantity}
        for sku, quantity in quantities.items()
    ], None


def _item_signature(items: list[dict[str, Any]]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(item["sku"]), int(item["quantity"])) for item in items))


def _matching_combination(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    signature = _item_signature(items)
    for rule in POLICIES_DATA["combinations"]:
        if _item_signature(rule["items"]) == signature:
            return rule
    return None


def _matching_scenario(sku: str) -> dict[str, Any] | None:
    for rule in POLICIES_DATA["scenario_rules"]:
        if sku in rule.get("skus", []):
            return rule
    return None


def _merge_checks(*checks: dict[str, list[str]]) -> dict[str, list[str]]:
    result = {
        "missing": [],
        "violations": [],
        "blocked": [],
        "not_preferred": [],
        "cautions": [],
    }
    for check in checks:
        for key in result:
            result[key].extend(check.get(key, []))
    return {key: list(dict.fromkeys(value)) for key, value in result.items()}


def validate_candidate_data(
    confirmed_needs: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        needs = validate_customer_needs(confirmed_needs)
    except CustomerNeedsError as exc:
        return _result(
            "invalid_needs",
            violations=[str(exc)],
            next_action="修正需求状态后重新规划。",
        )

    normalized, error = normalize_candidate_items(items)
    if error:
        return _result(
            "system_error",
            confirmed_needs=needs,
            violations=[error],
            next_action="系统应重新生成候选，不要求顾客改写需求。",
        )
    assert normalized is not None

    scope = needs.get("selection_scope")
    if len(normalized) > 1 and scope is None:
        return _result(
            "needs_clarification",
            confirmed_needs=needs,
            items=normalized,
            missing_information=["selection_scope"],
            next_action="提出组合前确认购买范围。",
        )
    if len(normalized) > 1 and scope != "完整护理":
        return _result(
            "scope_conflict",
            confirmed_needs=needs,
            items=normalized,
            violations=["当前选择范围不支持把多个商品作为组合推荐。"],
            next_action="按指定品类或单品范围重新选择。",
        )
    if len(normalized) == 1 and scope == "完整护理":
        return _result(
            "scope_conflict",
            confirmed_needs=needs,
            items=normalized,
            violations=["完整护理候选不能只有一个商品。"],
            next_action="重新生成资料明确支持的护理组合。",
        )

    if len(normalized) == 1 and needs.get("desired_category"):
        actual = PRODUCT_BY_SKU[normalized[0]["sku"]]["derived_tags"].get("category")
        if actual != needs["desired_category"]:
            return _result(
                "scope_conflict",
                confirmed_needs=needs,
                items=normalized,
                violations=[f"顾客指定了{needs['desired_category']}，候选属于{actual}。"],
                next_action="只在顾客指定品类中重新选择。",
            )

    skincare = any(item["sku"] != "P301" for item in normalized)
    safety_reasons: list[str] = []
    if skincare and needs.get("current_discomfort") is True:
        safety_reasons.append("顾客明确提到当前有明显不适。")
    if skincare and needs.get("skin_damage") is True:
        safety_reasons.append("顾客明确提到皮肤受损。")
    if skincare and needs.get("persistent_issue") is True:
        safety_reasons.append("顾客明确提到问题持续或反复出现。")
    if safety_reasons:
        return _result(
            "safety_blocked",
            confirmed_needs=needs,
            items=normalized,
            violations=safety_reasons,
            professional_consultation_required=True,
            next_action="停止刺激性尝试并建议咨询专业人士。",
        )

    if needs.get("fragrance_requirement") == "必须无香":
        scented = [
            item["sku"]
            for item in normalized
            if PRODUCT_BY_SKU[item["sku"]].get("derived_tags", {}).get("fragrance")
            not in {None, "none"}
        ]
        if scented:
            return _result(
                "not_supported",
                confirmed_needs=needs,
                items=normalized,
                violations=[f"顾客只接受无香，候选 {', '.join(scented)} 带香味。"],
                next_action="排除带香候选，不自动放宽无香要求。",
            )

    matched_rule_ids: list[str] = []
    checks: list[dict[str, list[str]]] = []
    if len(normalized) > 1:
        combination = _matching_combination(normalized)
        if combination is None:
            return _result(
                "not_supported",
                confirmed_needs=needs,
                items=normalized,
                violations=["品牌资料没有支持该商品组合。"],
                next_action="改用资料明确列出的组合或人工复核。",
            )
        matched_rule_ids.append(combination["id"])
        checks.append(evaluate_machine_conditions(needs, combination["machine_conditions"]))
        # 单品规则只补充非首选与使用提醒，组合适用性由组合规则决定。
        for item in normalized:
            product_rule = PRODUCT_BY_SKU[item["sku"]].get("recommendation_conditions") or {}
            supplement = evaluate_machine_conditions(
                needs,
                {
                    "required_all": {},
                    "required_any": [],
                    "goals_any": [],
                    "required_confirmations": [],
                    "blocked_if_any": [],
                    "not_preferred_if_any": product_rule.get("not_preferred_if_any", []),
                    "cautions": product_rule.get("cautions", []),
                },
            )
            checks.append(supplement)
    else:
        sku = normalized[0]["sku"]
        product_rule = PRODUCT_BY_SKU[sku].get("recommendation_conditions")
        if not isinstance(product_rule, dict):
            return _result(
                "manual_review",
                confirmed_needs=needs,
                items=normalized,
                violations=[f"{sku} 缺少可执行的推荐条件。"],
                next_action="建议人工复核。",
            )
        checks.append(evaluate_machine_conditions(needs, product_rule))
        scenario = _matching_scenario(sku)
        if scenario:
            matched_rule_ids.append(scenario["id"])
            checks.append(evaluate_machine_conditions(needs, scenario["machine_conditions"]))

    merged = _merge_checks(*checks)
    if merged["blocked"]:
        return _result(
            "safety_blocked",
            confirmed_needs=needs,
            items=normalized,
            matched_rule_ids=matched_rule_ids,
            violations=merged["blocked"],
            cautions=merged["cautions"],
            professional_consultation_required=True,
            next_action="停止当前推荐并说明安全边界。",
        )
    if merged["missing"]:
        return _result(
            "needs_clarification",
            confirmed_needs=needs,
            items=normalized,
            matched_rule_ids=matched_rule_ids,
            missing_information=merged["missing"],
            cautions=merged["cautions"],
            next_action="由决策规划器生成下一轮问题。",
        )
    if merged["violations"]:
        return _result(
            "not_supported",
            confirmed_needs=needs,
            items=normalized,
            matched_rule_ids=matched_rule_ids,
            violations=["已确认需求不满足候选商品或场景的必要条件。"],
            cautions=merged["cautions"],
            next_action="排除候选并重新规划。",
        )
    if merged["not_preferred"]:
        return _result(
            "not_preferred",
            confirmed_needs=needs,
            items=normalized,
            matched_rule_ids=matched_rule_ids,
            not_preferred_reasons=merged["not_preferred"],
            cautions=merged["cautions"],
            next_action="说明不是首选，并询问是否改看其他方向。",
        )
    return _result(
        "candidate_approved_with_caution" if merged["cautions"] else "candidate_approved",
        candidate_valid=True,
        confirmed_needs=needs,
        items=normalized,
        matched_rule_ids=matched_rule_ids,
        cautions=merged["cautions"],
        next_action="候选适用性已通过，可以独立报价。",
    )


__all__ = ["normalize_candidate_items", "validate_candidate_data"]
