"""澄初导购的顾客需求状态与显式证据校验。

需求状态只记录顾客明确表达的信息。每个已确认字段都保留用户原话，避免把
模型推断误当成事实；未知信息统一使用 None（goals 使用空列表）。
"""

from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, TypedDict


class CustomerNeedsError(ValueError):
    """需求字段、取值或证据不符合约定。"""


class CustomerNeeds(TypedDict):
    desired_sku: str | None
    desired_category: str | None
    selection_scope: str | None
    skin_tendency: str | None
    sensitive_tendency: bool | None
    current_discomfort: bool | None
    skin_damage: bool | None
    persistent_issue: bool | None
    skin_state_stable: bool | None
    barrier_fragile: bool | None
    fragrance_preference: str | None
    fragrance_requirement: str | None
    fragrance_sensitive: bool | None
    acid_experience: bool | None
    goals: list[str]
    budget_yuan: int | float | None
    budget_status: str
    travel_need: bool | None
    pending_questions: list[str]
    evidence: dict[str, str]


NEED_FIELDS = (
    "desired_sku",
    "desired_category",
    "selection_scope",
    "skin_tendency",
    "sensitive_tendency",
    "current_discomfort",
    "skin_damage",
    "persistent_issue",
    "skin_state_stable",
    "barrier_fragile",
    "fragrance_preference",
    "fragrance_requirement",
    "fragrance_sensitive",
    "acid_experience",
    "goals",
    "budget_yuan",
    "budget_status",
    "travel_need",
)

BOOLEAN_FIELDS = {
    "sensitive_tendency",
    "current_discomfort",
    "skin_damage",
    "persistent_issue",
    "skin_state_stable",
    "barrier_fragile",
    "fragrance_sensitive",
    "acid_experience",
    "travel_need",
}

ALLOWED_VALUES = {
    "desired_sku": {"P101", "P102", "P201", "P202", "P203", "P301"},
    "desired_category": {"洁面", "保湿", "精华", "旅行配件"},
    # “指定品类”表示顾客已经限定在洁面、保湿等某一品类内，但没有声称只买
    # 一件。它与“单品”分开，避免把“买洁面”错误理解为数量限制。
    "selection_scope": {"单品", "指定品类", "完整护理"},
    "skin_tendency": {"偏干", "偏油", "中性", "混合"},
    "fragrance_preference": {"无香", "可接受香味", "不限"},
    "fragrance_requirement": {"必须无香", "偏好无香", "无硬性要求"},
    "budget_status": {"unknown", "amount", "unlimited", "declined"},
}

SYSTEM_FIELDS = {"pending_questions"}

CATEGORY_EVIDENCE_TERMS = {
    "洁面": ("洁面", "洗面奶", "洗脸", "清洁产品"),
    "保湿": ("保湿", "乳液", "面霜"),
    "精华": ("精华",),
    "旅行配件": ("旅行配件", "旅行装", "分装瓶"),
}


def empty_customer_needs() -> CustomerNeeds:
    """创建一份全部未知的需求状态。"""
    return {
        "desired_sku": None,
        "desired_category": None,
        "selection_scope": None,
        "skin_tendency": None,
        "sensitive_tendency": None,
        "current_discomfort": None,
        "skin_damage": None,
        "persistent_issue": None,
        "skin_state_stable": None,
        "barrier_fragile": None,
        "fragrance_preference": None,
        "fragrance_requirement": None,
        "fragrance_sensitive": None,
        "acid_experience": None,
        "goals": [],
        "budget_yuan": None,
        "budget_status": "unknown",
        "travel_need": None,
        "pending_questions": [],
        "evidence": {},
    }


def _normalize_budget(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise CustomerNeedsError("budget_yuan 必须是非负金额")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise CustomerNeedsError("budget_yuan 必须是非负金额") from exc
    if not amount.is_finite() or amount < 0:
        raise CustomerNeedsError("budget_yuan 必须是非负金额")
    if amount * 100 != (amount * 100).to_integral_value():
        raise CustomerNeedsError("budget_yuan 最多支持两位小数")
    return int(amount) if amount == amount.to_integral_value() else float(amount)


def validate_customer_needs(
    needs: dict[str, Any], *, require_evidence: bool = True
) -> CustomerNeeds:
    """验证并规范化完整需求状态，不推断缺失字段。"""
    if not isinstance(needs, dict):
        raise CustomerNeedsError("顾客需求必须是对象")
    required = set(NEED_FIELDS) | SYSTEM_FIELDS | {"evidence"}
    missing = required - needs.keys()
    unknown = set(needs) - required
    if missing:
        raise CustomerNeedsError(f"顾客需求缺少字段：{sorted(missing)}")
    if unknown:
        raise CustomerNeedsError(f"顾客需求包含未知字段：{sorted(unknown)}")

    result = deepcopy(needs)
    evidence = result["evidence"]
    if not isinstance(evidence, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value.strip()
        for key, value in evidence.items()
    ):
        raise CustomerNeedsError("evidence 必须是字段名到非空用户原话的映射")
    if set(evidence) - set(NEED_FIELDS):
        raise CustomerNeedsError("evidence 包含未知需求字段")

    for field in BOOLEAN_FIELDS:
        value = result[field]
        if value is not None and not isinstance(value, bool):
            raise CustomerNeedsError(f"{field} 必须是 true、false 或 null")

    for field, allowed in ALLOWED_VALUES.items():
        value = result[field]
        if value is not None:
            if not isinstance(value, str) or value not in allowed:
                raise CustomerNeedsError(
                    f"{field} 只允许：{sorted(allowed)} 或 null"
                )

    goals = result["goals"]
    if not isinstance(goals, list) or not all(
        isinstance(goal, str) and goal.strip() for goal in goals
    ):
        raise CustomerNeedsError("goals 必须是非空字符串组成的列表")
    result["goals"] = list(dict.fromkeys(goal.strip() for goal in goals))

    if result["budget_yuan"] is not None:
        result["budget_yuan"] = _normalize_budget(result["budget_yuan"])
        if result["budget_status"] not in {"amount", "unknown"}:
            raise CustomerNeedsError("预算金额与预算状态冲突")
        result["budget_status"] = "amount"
    elif result["budget_status"] == "amount":
        raise CustomerNeedsError("budget_status=amount 时必须提供预算金额")

    pending = result["pending_questions"]
    if not isinstance(pending, list) or not all(
        isinstance(field, str) and field in NEED_FIELDS for field in pending
    ):
        raise CustomerNeedsError("pending_questions 必须是有效需求字段组成的列表")
    result["pending_questions"] = list(dict.fromkeys(pending))

    if require_evidence:
        for field in NEED_FIELDS:
            value = result[field]
            if field == "budget_status":
                confirmed = value in {"unlimited", "declined"}
            else:
                confirmed = bool(value) if field == "goals" else value is not None
            if confirmed and field not in evidence:
                raise CustomerNeedsError(f"已确认字段 {field} 缺少用户原话证据")

    category = result["desired_category"]
    sku = result["desired_sku"]
    if sku is not None and require_evidence:
        quote = evidence.get("desired_sku", "").upper()
        if sku not in quote:
            raise CustomerNeedsError("desired_sku 的原话不能支持该商品")
    if category is not None and require_evidence:
        quote = evidence.get("desired_category", "")
        if not any(term in quote for term in CATEGORY_EVIDENCE_TERMS[category]):
            raise CustomerNeedsError("desired_category 的原话不能支持该品类")
    if result["selection_scope"] == "指定品类" and category is None:
        raise CustomerNeedsError("指定品类范围必须同时包含明确品类")

    result["evidence"] = {key: value.strip() for key, value in evidence.items()}
    return result  # type: ignore[return-value]


def merge_confirmed_needs(
    current: dict[str, Any],
    updates: dict[str, dict[str, Any]],
    latest_user_text: str,
) -> CustomerNeeds:
    """合并本轮明确需求；每项 evidence 必须逐字出现在最新用户消息中。

    updates 示例：
        {"skin_tendency": {"value": "偏干", "evidence": "我偏干"}}
    """
    state = validate_customer_needs(current)
    if not isinstance(updates, dict):
        raise CustomerNeedsError("updates 必须是对象")
    if not isinstance(latest_user_text, str) or not latest_user_text.strip():
        raise CustomerNeedsError("latest_user_text 必须是非空字符串")

    for field, update in updates.items():
        if field not in NEED_FIELDS:
            raise CustomerNeedsError(f"未知需求字段：{field}")
        if not isinstance(update, dict) or set(update) != {"value", "evidence"}:
            raise CustomerNeedsError(
                f"{field} 更新必须只包含 value 和 evidence"
            )
        quote = update["evidence"]
        if not isinstance(quote, str) or not quote.strip():
            raise CustomerNeedsError(f"{field} 缺少非空 evidence")
        quote = quote.strip()
        if quote not in latest_user_text:
            raise CustomerNeedsError(f"{field} 的 evidence 未出现在最新用户消息中")

        state[field] = deepcopy(update["value"])
        if update["value"] is None or update["value"] == []:
            state["evidence"].pop(field, None)
        else:
            state["evidence"][field] = quote

        if field == "budget_yuan" and update["value"] is not None:
            state["budget_status"] = "amount"
            state["evidence"].pop("budget_status", None)
        elif field == "budget_status" and update["value"] != "amount":
            state["budget_yuan"] = None
            state["evidence"].pop("budget_yuan", None)

    answered = set(updates)
    if "budget_yuan" in answered:
        answered.add("budget_status")
    if "budget_status" in answered and state["budget_status"] in {"unlimited", "declined"}:
        answered.add("budget_yuan")
    state["pending_questions"] = [
        field for field in state["pending_questions"] if field not in answered
    ]

    return validate_customer_needs(state)


def set_pending_questions(
    needs: dict[str, Any], fields: list[str] | tuple[str, ...]
) -> CustomerNeeds:
    """记录系统已经提出但用户尚未回答的问题，不伪造用户证据。"""
    state = validate_customer_needs(needs)
    invalid = [field for field in fields if field not in NEED_FIELDS]
    if invalid:
        raise CustomerNeedsError(f"待确认问题包含未知字段：{invalid}")
    state["pending_questions"] = list(dict.fromkeys(fields))
    return validate_customer_needs(state)


__all__ = [
    "CustomerNeeds",
    "CustomerNeedsError",
    "empty_customer_needs",
    "merge_confirmed_needs",
    "set_pending_questions",
    "validate_customer_needs",
]
