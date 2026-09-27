"""用结构化模型输出提取最新一轮中明确表达的顾客需求。"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from customer_needs import CustomerNeeds, merge_confirmed_needs
from explicit_fact_parser import (
    extract_explicit_fact_updates,
    extract_pending_question_updates,
)


NeedField = Literal[
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
]


class NeedUpdate(BaseModel):
    """一个字段的显式更新及其用户原话证据。"""

    field: NeedField
    value: str | bool | float | list[str] | None = Field(
        description="规范化后的字段值；布尔字段使用 true/false，未知不要输出。"
    )
    evidence: str = Field(
        description="必须逐字复制自最新用户消息，且直接支持该字段值。"
    )


class NeedsExtraction(BaseModel):
    """最新一条用户消息中明确出现的稀疏更新。"""

    updates: list[NeedUpdate] = Field(default_factory=list)


EXTRACTION_PROMPT = """
你是澄初个人护理导购的需求信息提取器。只提取“最新用户消息”明确表达的事实，
不得根据常识、肤质、商品特征或旧状态推断未说出的信息。

字段和值：
- desired_sku：P101、P102、P201、P202、P203、P301 之一；只在最新消息明确出现时提取；
- desired_category：洁面、保湿、精华、旅行配件之一；
- selection_scope：单品、指定品类、完整护理之一。“只要一件/只看单品”记为单品；
  用户明确购买、查看或选择某个品类时记为指定品类；“搭配一套/完整护理”记为完整护理；
- skin_tendency：偏干、偏油、中性、混合之一；
- sensitive_tendency、current_discomfort、skin_damage、persistent_issue、skin_state_stable、
  barrier_fragile、fragrance_sensitive、acid_experience、travel_need：布尔值；
- fragrance_preference：无香、可接受香味、不限之一；
- fragrance_requirement：必须无香、偏好无香、无硬性要求之一；只有“只接受/必须”
  等明确措辞才能记为必须无香；
- goals：可使用简单护理、清爽肤感、轻薄肤感、改善粗糙等明确目标组成的列表；
- budget_yuan：只有明确金额时提取数字，“预算有限”不能补成金额。
- budget_status：unknown、amount、unlimited、declined 之一；只有用户明确说预算不限或
  拒绝提供预算时才输出 unlimited 或 declined。金额由规则通道同步为 amount。

要求：
1. evidence 必须逐字复制自最新用户消息，不能改写、概括或引用旧消息。
2. “敏感倾向”不能推断为“正在不适”“皮肤受损”“无香偏好”。
3. “目前没有明显不适”应提取 current_discomfort=false。
4. 用户纠正旧信息时输出新值；没有涉及的字段不要输出。
5. 允许把自然表达规范化，例如“想简单一点”可记为 goals=["简单护理"]，
   但 evidence 仍必须保留原话“想简单一点”。
6. “可以接受香味”只表示 fragrance_preference=“可接受香味”；如果用户还明确说
   “对香味不敏感”，必须另外提取 fragrance_sensitive=false。
7. “简单护理”是护理目标，不等于“完整护理”；不得据此推断 selection_scope。
8. 没有任何明确事实时返回空 updates。
9. “买洁面”“想找精华”“保湿怎么选”等表达应同时提取 desired_category
   和 selection_scope=“指定品类”。这表示限定品类，不推断购买数量。
10. “持续不适/问题反复出现”应提取 persistent_issue=true；不能把普通敏感倾向
    推断成持续问题。
""".strip()


def create_needs_extractor(llm: Any):
    """把基础聊天模型约束为 NeedsExtraction 结构化输出。"""
    return llm.with_structured_output(NeedsExtraction, method="function_calling")


def extract_confirmed_needs(
    extractor: Any,
    current_needs: CustomerNeeds,
    latest_user_text: str,
) -> CustomerNeeds:
    """提取本轮稀疏更新，并通过原话校验后合并到现有状态。"""
    response = extractor.invoke(
        [
            SystemMessage(content=EXTRACTION_PROMPT),
            HumanMessage(
                content=(
                    "当前需求状态（仅供识别纠正，不得把旧值作为本轮证据）：\n"
                    + json.dumps(current_needs, ensure_ascii=False, sort_keys=True)
                    + "\n\n最新用户消息：\n"
                    + latest_user_text
                )
            ),
        ]
    )
    if isinstance(response, dict):
        extraction = NeedsExtraction.model_validate(response)
    elif isinstance(response, NeedsExtraction):
        extraction = response
    else:
        raise ValueError("需求提取器没有返回 NeedsExtraction")

    updates: dict[str, dict[str, Any]] = {}
    for update in extraction.updates:
        if update.field in updates:
            raise ValueError(f"需求提取器重复输出字段：{update.field}")
        updates[update.field] = {
            "value": update.value,
            "evidence": update.evidence,
        }
    # 双通道合并：Qwen 负责灵活表达，规则通道补足无歧义的明确事实。
    # 最新原话中的明确值优先；最终仍由 merge_confirmed_needs 验证 evidence。
    updates.update(extract_explicit_fact_updates(latest_user_text))
    contextual = extract_pending_question_updates(
        latest_user_text, current_needs.get("pending_questions") or []
    )
    for field, update in contextual.items():
        updates.setdefault(field, update)
    return merge_confirmed_needs(current_needs, updates, latest_user_text)


__all__ = [
    "EXTRACTION_PROMPT",
    "NeedUpdate",
    "NeedsExtraction",
    "create_needs_extractor",
    "extract_confirmed_needs",
]
