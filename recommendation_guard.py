"""正式推荐的校验、报价编排与用户可见渲染。

候选适用性由 recommendation_validator 独立校验；只有候选有效后才调用目录报价，
避免价格计算和商品适用规则互相覆盖。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from catalog_tools import POLICIES_DATA, PRODUCT_BY_SKU, calculate_quote_data
from customer_needs import CustomerNeedsError, validate_customer_needs
from recommendation_validator import validate_candidate_data


def select_documented_combination(
    confirmed_needs: dict[str, Any],
) -> dict[str, Any] | None:
    """只在顾客明确选择完整护理且候选校验通过时返回资料组合。"""
    try:
        needs = validate_customer_needs(confirmed_needs)
    except CustomerNeedsError:
        return None
    if needs.get("selection_scope") != "完整护理":
        return None

    for combination in POLICIES_DATA["combinations"]:
        validation = validate_candidate_data(needs, combination["items"])
        if validation.get("candidate_valid") is True:
            return {
                "rule_id": combination["id"],
                "items": deepcopy(combination["items"]),
                "source": deepcopy(combination["source"]),
                "source_text": combination["source_text"],
            }
    return None


def validate_recommendation_data(
    confirmed_needs: dict[str, Any], items: list[dict[str, Any]]
) -> dict[str, Any]:
    """先校验候选，再独立报价并判断预算状态。"""
    result = validate_candidate_data(confirmed_needs, items)
    if result.get("candidate_valid") is not True:
        return result

    needs = result["confirmed_needs"]
    budget = needs.get("budget_yuan") if needs.get("budget_status") == "amount" else None
    quote = calculate_quote_data(result["items"], budget)
    result["quote"] = quote
    if quote.get("status") != "success":
        result.update(
            {
                "status": "system_error",
                "approved": False,
                "candidate_valid": False,
                "violations": [quote.get("message", "报价计算失败")],
                "next_action": "系统重新计算报价，不要求顾客改写需求。",
            }
        )
        return result

    if quote.get("within_budget") is False:
        result.update(
            {
                "status": "budget_conflict",
                "approved": False,
                "next_action": "说明总价和超出金额，询问调整预算或优先品类。",
            }
        )
        return result

    result.update(
        {
            "status": "approved_with_caution" if result.get("cautions") else "approved",
            "approved": True,
            "next_action": (
                "可以形成正式建议，但必须完整说明使用提醒。"
                if result.get("cautions")
                else "可以依据已匹配规则、报价和限制生成正式推荐。"
            ),
        }
    )
    return result


FIELD_LABELS = {
    "desired_category": "想购买的品类",
    "selection_scope": "当前选择范围",
    "skin_tendency": "肤质倾向",
    "sensitive_tendency": "是否有敏感倾向",
    "current_discomfort": "是否正在明显不适",
    "skin_damage": "皮肤是否受损",
    "persistent_issue": "问题是否持续或反复出现",
    "skin_state_stable": "皮肤状态是否稳定",
    "barrier_fragile": "是否有屏障脆弱倾向",
    "fragrance_preference": "香味偏好",
    "fragrance_requirement": "香味是否为硬性要求",
    "fragrance_sensitive": "是否对香味敏感",
    "acid_experience": "是否有相关焕肤使用经验",
    "goals": "护理目标",
    "budget_yuan": "预算",
    "travel_need": "是否有出差或旅行需求",
}


def render_approved_validation(validation: dict[str, Any]) -> str:
    """只使用目录与校验结果生成正式建议，避免模型补写资料外用法。"""
    if validation.get("approved") is not True:
        raise ValueError("只有 approved=true 的结果才能渲染正式建议")
    quote = validation.get("quote") or {}
    needs = validation.get("confirmed_needs") or {}
    quote_items = quote.get("items") or []
    recommendation_lines = []
    reason_lines = []
    reminder_lines = list(validation.get("cautions") or [])
    for item in quote_items:
        sku = item.get("sku")
        product = PRODUCT_BY_SKU.get(sku, {})
        name = item.get("name") or product.get("name") or sku
        recommendation_lines.append(f"- **{name}**（{sku}）")
        features = "、".join(product.get("features") or [])
        suitability = product.get("suitability")
        detail = "；".join(part for part in (features, suitability) if part)
        if detail:
            reason_lines.append(f"- **{name}：** {detail}")
        reminder_lines.extend(product.get("limitations") or [])

    reminder_lines = list(dict.fromkeys(item.rstrip("。") for item in reminder_lines if item))
    total = quote.get("total_yuan")
    budget = quote.get("budget_yuan")
    price_lines = [
        f"- {item.get('name')}：{item.get('unit_price_yuan')} 元 × {item.get('quantity')}"
        for item in quote_items
    ]
    price_lines.append(f"- **总价：{total} 元**")
    if budget is None:
        if needs.get("budget_status") == "unlimited":
            price_lines.append("- 预算：不限")
        elif needs.get("budget_status") == "declined":
            price_lines.append("- 预算：暂不提供，本次只展示客观价格")
        else:
            price_lines.append("- 预算：待确认")
    elif quote.get("within_budget") is True:
        price_lines.append(
            f"- 在 {budget} 元预算内，剩余 {quote.get('remaining_budget_yuan')} 元"
        )

    reminders = "\n".join(f"- {item}" for item in reminder_lines) or "- 暂无额外提醒"
    confirmed_lines: list[str] = []
    if needs.get("skin_tendency"):
        confirmed_lines.append(f"- 肤质倾向：{needs['skin_tendency']}")
    if needs.get("goals"):
        confirmed_lines.append(f"- 护理目标：{'、'.join(needs['goals'])}")
    if needs.get("fragrance_preference"):
        confirmed_lines.append(f"- 香味偏好：{needs['fragrance_preference']}")
    if needs.get("budget_yuan") is not None:
        confirmed_lines.append(f"- 预算：{needs['budget_yuan']} 元")
    elif needs.get("budget_status") == "unlimited":
        confirmed_lines.append("- 预算：不限")
    elif needs.get("budget_status") == "declined":
        confirmed_lines.append("- 预算：暂不提供")
    confirmed_text = "\n".join(confirmed_lines) or "- 已按你明确提供的信息核对"
    return (
        "### 已确认需求\n\n"
        + confirmed_text
        + "\n\n### 推荐建议\n\n"
        + "\n".join(recommendation_lines)
        + "\n\n### 为什么这样建议\n\n"
        + "\n".join(reason_lines)
        + "\n\n### 价格与预算\n\n"
        + "\n".join(price_lines)
        + "\n\n### 使用提醒与限制\n\n"
        + reminders
    )


def render_unapproved_validation(validation: dict[str, Any]) -> str:
    """把未通过校验的结果渲染为面向顾客的回复。"""
    status = validation.get("status")
    quote = validation.get("quote") or {}
    items = quote.get("items") or []
    item_text = "、".join(
        f"{item['name']}（{item['sku']}）" for item in items
    ) or "当前候选方案"

    if status == "budget_conflict":
        total = quote.get("total_yuan")
        budget = quote.get("budget_yuan")
        over = quote.get("over_budget_yuan")
        return (
            "### 预算需要调整\n\n"
            f"根据你目前提供的情况，可以考虑的日常护理组合是：{item_text}。"
            f"两件商品合计 **{total} 元**，比当前 **{budget} 元**预算高 "
            f"**{over} 元**，所以我暂时不建议直接购买完整组合。\n\n"
            "以上价格按品牌商品资料中的单价计算，暂未包含未经确认的优惠或促销信息。\n\n"
            "### 接下来\n\n"
            "你愿意提高一些预算，还是想先从洁面和保湿中选择一个更优先的品类？"
        )
    if status == "needs_clarification":
        labels = [
            FIELD_LABELS.get(field, field)
            for field in validation.get("missing_information", [])[:2]
        ]
        questions = "、".join(labels) or "影响选择的关键信息"
        return (
            "### 还需要了解一些信息\n\n"
            "为了给出更合适的建议，请先告诉我："
            f"**{questions}**。\n\n"
            "确认后，我会继续核对商品特点、使用限制和预算。"
        )
    if status == "awaiting_information":
        labels = [
            FIELD_LABELS.get(field, field)
            for field in validation.get("missing_information", [])[:2]
        ]
        return "为了继续核对，请先确认：" + "、".join(labels) + "。"
    if status == "scope_conflict":
        return (
            "### 按你的选择范围重新调整\n\n"
            "当前候选商品与你明确的单品或完整护理范围不一致，因此不会直接展示为推荐。\n\n"
            "### 接下来\n\n我会只在你指定的范围和品类中重新选择。"
        )
    if status == "not_preferred":
        reasons = "；".join(
            validation.get("not_preferred_reasons")
            or ["当前方案不是资料中的优先选择"]
        )
        return (
            "### 当前方案不是首选\n\n"
            f"{reasons}\n\n"
            "这不等同于安全禁用，但现有资料不支持把它作为你的首选推荐。"
            "如果你愿意，我可以继续核对其他更合适的方向。"
        )
    if status == "safety_blocked":
        reasons = "；".join(
            validation.get("violations") or ["当前情况需要谨慎处理"]
        )
        return (
            "### 当前暂不建议\n\n"
            f"{reasons}\n\n"
            "建议先停止刺激性尝试，并咨询专业人士。这里不作医疗诊断。"
        )
    if status in {"not_supported", "manual_review"}:
        return (
            "### 当前暂不建议\n\n"
            "品牌资料暂时不能支持把当前方案作为可靠建议。\n\n"
            "### 接下来\n\n"
            "我可以根据资料中明确支持的商品重新为你选择；"
            "如果现有资料仍未覆盖，则需要人工进一步确认。"
        )
    if status in {"system_error", "invalid_request", "invalid_needs"}:
        return (
            "### 本次核对暂未完成\n\n"
            "系统没有生成有效的商品或报价结果。本轮不会要求你重新描述肤质或偏好；"
            "请稍后重试，仍无法完成时建议人工复核。"
        )
    return (
        "### 还需要重新确认\n\n"
        "本次商品或需求信息不完整，请补充或换一种明确说法后重新尝试。"
    )


__all__ = [
    "render_approved_validation",
    "render_unapproved_validation",
    "select_documented_combination",
    "validate_recommendation_data",
]

