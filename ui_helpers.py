"""澄初 Streamlit 页面的纯展示辅助函数。

本模块不调用模型，也不改变业务判断，只负责把 Agent 状态转换成用户能理解的
中文摘要，并隐藏考试文件名等不应出现在公开演示页面的信息。
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage


NEED_LABELS = {
    "desired_sku": "关注商品",
    "desired_category": "关注品类",
    "selection_scope": "选择范围",
    "skin_tendency": "肤质倾向",
    "sensitive_tendency": "敏感倾向",
    "current_discomfort": "当前明显不适",
    "skin_damage": "皮肤受损",
    "persistent_issue": "持续或反复问题",
    "skin_state_stable": "皮肤状态",
    "barrier_fragile": "屏障脆弱倾向",
    "fragrance_preference": "香味偏好",
    "fragrance_requirement": "香味要求",
    "fragrance_sensitive": "对香味敏感",
    "acid_experience": "焕肤使用经验",
    "goals": "护理目标",
    "budget_yuan": "预算",
    "budget_status": "预算状态",
    "travel_need": "旅行携带需求",
}

NEED_ORDER = (
    "desired_sku",
    "skin_tendency",
    "sensitive_tendency",
    "goals",
    "desired_category",
    "selection_scope",
    "fragrance_preference",
    "fragrance_requirement",
    "fragrance_sensitive",
    "budget_yuan",
    "budget_status",
    "current_discomfort",
    "skin_damage",
    "persistent_issue",
    "skin_state_stable",
    "barrier_fragile",
    "acid_experience",
    "travel_need",
)

STATUS_META = {
    "approved": {
        "tone": "success",
        "title": "方案已核对",
        "description": "已通过适用条件、安全边界和预算检查。",
    },
    "approved_with_caution": {
        "tone": "warning",
        "title": "方案已核对，请留意使用提醒",
        "description": "方案符合条件，使用前仍需查看品牌资料中的注意事项。",
    },
    "not_preferred": {
        "tone": "warning",
        "title": "当前方案不是首选",
        "description": "没有触发安全禁用，但品牌资料不建议把它作为优先选择。",
    },
    "scope_conflict": {
        "tone": "info",
        "title": "按你的选择范围重新调整",
        "description": "当前候选与单品或完整护理范围不一致。",
    },
    "budget_conflict": {
        "tone": "warning",
        "title": "预算需要调整",
        "description": "候选方案符合资料条件，但超过了当前预算。",
    },
    "needs_clarification": {
        "tone": "info",
        "title": "还需要了解一些信息",
        "description": "补充关键信息后才能继续核对建议。",
    },
    "awaiting_information": {
        "tone": "info",
        "title": "正在了解你的需求",
        "description": "补充当前关键问题后，我会继续核对，不会重复询问已确认的信息。",
    },
    "safety_blocked": {
        "tone": "danger",
        "title": "当前暂不建议",
        "description": "当前情况触发了品牌资料中的安全边界。",
    },
    "not_supported": {
        "tone": "neutral",
        "title": "当前方案缺少资料支持",
        "description": "现有品牌资料不能支持把该方案作为正式推荐。",
    },
    "manual_review": {
        "tone": "neutral",
        "title": "建议人工确认",
        "description": "现有规则不足以完成可靠判断。",
    },
    "invalid_needs": {
        "tone": "neutral",
        "title": "需要重新确认信息",
        "description": "当前需求信息未通过一致性检查。",
    },
    "invalid_request": {
        "tone": "neutral",
        "title": "本次核对未完成",
        "description": "系统提交的商品或数量信息需要内部修正。",
    },
    "system_error": {
        "tone": "neutral",
        "title": "本次核对暂未完成",
        "description": "系统未能生成有效候选或报价，请稍后重试。",
    },
}

TOOL_LABELS = {
    "search_products": "查询商品资料",
    "get_policy": "核对品牌规则与服务边界",
    "calculate_quote": "计算价格与预算",
    "validate_recommendation": "检查适用条件、安全边界和预算",
}


def public_text(value: Any) -> str:
    """替换只适合开发阶段出现的考试文件名称。"""
    text = "" if value is None else str(value)
    replacements = {
        "08题 澄初个人护理.pdf": "澄初品牌资料",
        "08题澄初个人护理.pdf": "澄初品牌资料",
        "08题 澄初个人护理": "澄初品牌资料",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def customer_answer_text(value: Any) -> str:
    """清理顾客正文中的内部规则编号；SKU 等商品标识仍保留。"""
    text = public_text(value)
    text = re.sub(r"[（(]\s*C\d{2,}\s*[）)]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?<![A-Za-z0-9])C\d{2,}(?![A-Za-z0-9])",
        "品牌资料中的组合规则",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _display_value(field: str, value: Any) -> str:
    if field == "budget_yuan":
        return f"{value:g} 元" if isinstance(value, float) else f"{value} 元"
    if field == "goals":
        return "、".join(value)
    if field == "budget_status":
        return {"unlimited": "预算不限", "declined": "暂不提供预算"}.get(value, str(value))
    if isinstance(value, bool):
        if field == "current_discomfort":
            return "有明显不适" if value else "无明显不适"
        if field == "skin_damage":
            return "有受损" if value else "无受损"
        if field == "persistent_issue":
            return "问题持续或反复" if value else "无持续或反复问题"
        if field == "skin_state_stable":
            return "稳定" if value else "不稳定"
        if field == "sensitive_tendency":
            return "有敏感倾向" if value else "无敏感倾向"
        if field == "fragrance_sensitive":
            return "对香味敏感" if value else "对香味不敏感"
        if field == "acid_experience":
            return "有相关经验" if value else "无相关经验"
        if field == "travel_need":
            return "有旅行携带需求" if value else "无旅行携带需求"
        if field == "barrier_fragile":
            return "有屏障脆弱倾向" if value else "无屏障脆弱倾向"
        return "是" if value else "否"
    return public_text(value)


def needs_details(needs: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(needs, dict):
        return []
    evidence = needs.get("evidence") if isinstance(needs.get("evidence"), dict) else {}
    rows: list[dict[str, str]] = []
    for field in NEED_ORDER:
        value = needs.get(field)
        if value is None or value == [] or (
            field == "budget_status" and value in {"unknown", "amount"}
        ):
            continue
        rows.append(
            {
                "field": field,
                "label": NEED_LABELS[field],
                "value": _display_value(field, value),
                "evidence": public_text(evidence.get(field, "")),
            }
        )
    return rows


def needs_summary(needs: dict[str, Any] | None, max_items: int = 6) -> str:
    rows = needs_details(needs)
    if not rows:
        return "尚未确认具体需求"
    values = [row["value"] for row in rows[:max_items]]
    if len(rows) > max_items:
        values.append(f"另有 {len(rows) - max_items} 项")
    return " · ".join(values)


def validation_meta(validation: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(validation, dict):
        return None
    status = validation.get("status")
    if status in STATUS_META:
        return {"status": status, **STATUS_META[status]}
    return {
        "status": public_text(status or "unknown"),
        "tone": "neutral",
        "title": "本次结果需要确认",
        "description": "当前返回了页面尚未识别的状态。",
    }


ALLOWED_CONCLUSIONS = {
    "approved": "可以根据已确认需求形成正式建议。",
    "approved_with_caution": "可以形成正式建议，但必须同时展示使用提醒。",
    "not_preferred": "只能说明当前方案不是首选，不能包装成安全禁用或正式首选推荐。",
    "scope_conflict": "按用户明确的单品或完整护理范围重新选择。",
    "budget_conflict": "暂不建议购买完整组合，并询问是否调整预算或优先选择一个品类。",
    "needs_clarification": "先补充最关键的信息，暂不形成商品建议。",
    "awaiting_information": "回答当前关键问题后继续核对，暂不进入商品校验。",
    "safety_blocked": "当前不继续推荐护肤商品，并提示咨询专业人士。",
    "not_supported": "当前方案不能作为正式建议，应改用资料明确支持的方案或人工确认。",
    "manual_review": "现有资料不足以判断，需要人工进一步确认。",
    "invalid_needs": "重新确认用户需求后再继续。",
    "invalid_request": "由系统修正商品或数量参数，不要求用户重新表述需求。",
    "system_error": "本轮没有形成顾客需要承担的业务结论，应由系统重试或人工复核。",
}


def allowed_conclusion(validation: dict[str, Any] | None) -> str:
    if not isinstance(validation, dict):
        return "本轮没有形成需要核对的商品结论。"
    return ALLOWED_CONCLUSIONS.get(
        validation.get("status"), "当前结论需要进一步确认。"
    )


def validation_checks(validation: dict[str, Any] | None) -> list[dict[str, str]]:
    """生成人能直接判断对错的核对项，不展示 Python 字段和值。"""
    if not isinstance(validation, dict):
        return []
    checks: list[dict[str, str]] = []
    needs = validation.get("confirmed_needs") or {}
    evidence = needs.get("evidence") if isinstance(needs.get("evidence"), dict) else {}
    if evidence:
        checks.append(
            {
                "level": "pass",
                "text": f"已确认 {len(evidence)} 项用户信息，并保留了对应原话。",
            }
        )

    rule_ids = validation.get("matched_rule_ids") or []
    if rule_ids:
        checks.append(
            {
                "level": "pass",
                "text": f"候选商品来自品牌资料支持的方案（规则 {'、'.join(rule_ids)}）。",
            }
        )
    elif validation.get("status") in {"not_supported", "manual_review"}:
        checks.append(
            {"level": "warning", "text": "当前方案没有匹配到足够的品牌规则支持。"}
        )

    quote = validation.get("quote") or {}
    if quote.get("status") == "success":
        checks.append(
            {
                "level": "pass",
                "text": f"商品价格已按品牌目录核算，总价为 {quote.get('total_yuan')} 元。",
            }
        )
        if quote.get("within_budget") is True:
            checks.append(
                {
                    "level": "pass",
                    "text": f"方案在预算内，剩余 {quote.get('remaining_budget_yuan')} 元。",
                }
            )
        elif quote.get("within_budget") is False:
            checks.append(
                {
                    "level": "warning",
                    "text": f"方案超过当前预算 {quote.get('over_budget_yuan')} 元。",
                }
            )

    if needs.get("current_discomfort") is False and needs.get("skin_damage") is False:
        checks.append(
            {"level": "pass", "text": "用户已明确当前没有明显不适或皮肤受损。"}
        )
    if validation.get("status") == "not_preferred":
        checks.append(
            {"level": "warning", "text": "当前候选属于非首选，不等同于安全禁用。"}
        )
    for caution in validation.get("cautions") or []:
        checks.append({"level": "info", "text": public_text(caution)})
    if validation.get("status") == "safety_blocked":
        checks.append(
            {"level": "stop", "text": "当前情况触发安全边界，不应继续正式推荐。"}
        )
    if validation.get("manual_review_required"):
        checks.append(
            {"level": "warning", "text": "现有资料不能完成可靠判断，需要人工确认。"}
        )
    return checks


def source_label(source: dict[str, Any] | None) -> str:
    if not isinstance(source, dict):
        return "澄初品牌资料"
    parts = ["澄初品牌资料"]
    for key in ("section", "locator"):
        value = source.get(key)
        if value:
            clean = public_text(value)
            if clean not in parts:
                parts.append(clean)
    return " / ".join(parts)


def quote_rows(validation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(validation, dict):
        return []
    quote = validation.get("quote")
    if not isinstance(quote, dict):
        return []
    rows = []
    for item in quote.get("items") or []:
        rows.append(
            {
                "name": public_text(item.get("name", "未知商品")),
                "sku": public_text(item.get("sku", "")),
                "unit_price_yuan": item.get("unit_price_yuan"),
                "quantity": item.get("quantity"),
                "subtotal_yuan": item.get("subtotal_yuan"),
            }
        )
    return rows


def current_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def friendly_tool_trace(messages: list[AnyMessage] | None) -> list[str]:
    if not messages:
        return []
    steps: list[str] = []
    seen: set[str] = set()
    for message in current_turn_messages(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            name = call.get("name", "")
            label = TOOL_LABELS.get(name, "核对业务资料")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if name == "search_products" and args.get("category"):
                label += f"：{public_text(args['category'])}"
            elif name == "get_policy" and (args.get("topic") or args.get("sku")):
                target = args.get("topic") or args.get("sku")
                label += f"：{public_text(target)}"
            key = f"{name}:{label}"
            if key not in seen:
                seen.add(key)
                steps.append(label)
    return steps


def latest_ai_content(messages: list[AnyMessage] | None) -> str:
    for message in reversed(messages or []):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content.strip():
                return customer_answer_text(message.content)
    return "本次没有生成可展示的回答，请换一种说法重试。"
