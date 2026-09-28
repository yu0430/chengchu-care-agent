"""把确定性计划和业务结果转换为顾客可读的中文。"""

from __future__ import annotations

from typing import Any


QUESTION_TEXT = {
    "budget_yuan": "你的整体预算大概是多少？如果不设上限或暂时不方便提供，也可以直接告诉我。",
    "fragrance_preference": "香味方面更偏好无香，还是可以接受带香味的产品？",
    "fragrance_sensitive": "香味是否会让你感到不舒服或明显敏感？",
    "selection_scope": "你想先看一件单品或指定品类，还是一起看看完整的基础护理组合？",
    "desired_category": "如果只先看一类，你想看洁面、保湿、精华还是旅行配件？",
    "sensitive_tendency": "你的皮肤是否有敏感倾向？",
    "current_discomfort": "目前是否正在出现明显不适？",
    "skin_damage": "目前皮肤是否有受损情况？",
    "persistent_issue": "这个问题是否持续或反复出现？",
    "acid_experience": "你是否有果酸或相关焕肤产品的使用经验？",
    "skin_state_stable": "你目前的皮肤状态是否稳定？",
    "travel_need": "你近期是否有明确的出差或旅行携带需求？",
    "skin_tendency": "你的肤质更偏干、偏油、中性还是混合？",
    "goals": "你主要希望改善什么，或更偏好怎样的使用感受？",
}


def render_question_plan(plan: dict[str, Any], needs: dict[str, Any]) -> str:
    question = plan.get("question") or {}
    fields = question.get("fields") or []
    question_id = question.get("question_id")
    confirmed: list[str] = []
    if needs.get("skin_tendency"):
        confirmed.append(str(needs["skin_tendency"]))
    if needs.get("goals"):
        confirmed.extend(str(value) for value in needs["goals"])
    prefix = f"了解，你提到的是{'、'.join(confirmed)}。\n\n" if confirmed else ""
    if question_id == "p201_skin_safety":
        questions = [
            "目前是否有敏感倾向、皮肤受损或明显不适的情况？"
            "如果都没有，可以直接说“都没有”。"
        ]
    elif question_id == "p201_experience_state":
        missing = set(fields)
        if missing == {"acid_experience", "skin_state_stable"}:
            questions = ["你是否有果酸或相关焕肤产品的使用经验，目前皮肤状态是否稳定？"]
        else:
            questions = [QUESTION_TEXT.get(field, f"请确认{field}。") for field in fields]
    elif question_id == "single_category":
        questions = [QUESTION_TEXT["desired_category"]]
    else:
        questions = [QUESTION_TEXT.get(field, f"请确认{field}。") for field in fields]
    return prefix + "\n\n".join(questions)


def render_safety_plan(plan: dict[str, Any]) -> str:
    reasons = set(plan.get("reason_codes") or [])
    if "PERSISTENT_ISSUE" in reasons:
        situation = "问题持续或反复出现"
    elif "SKIN_DAMAGE" in reasons:
        situation = "皮肤有受损情况"
    else:
        situation = "目前有明显不适"
    return (
        f"你提到{ situation }，目前先不继续推荐护肤产品。建议停止刺激性尝试，"
        "并咨询专业人士。这里不作医疗诊断；如果你只是想查询商品价格或客观资料，我仍可以继续说明。"
    )


def render_no_supported_option(plan: dict[str, Any], needs: dict[str, Any]) -> str:
    conflicts = set(plan.get("conflicts") or [])
    if "fragrance_requirement" in conflicts or needs.get("fragrance_requirement") == "必须无香":
        return (
            "你明确只接受无香产品，而资料中的偏油清爽组合包含带香味商品，因此不能把它作为建议。"
            "我不会自动放宽你的无香要求；如果你愿意，可以继续按指定品类核对资料中的无香选项。"
        )
    if "NO_TRAVEL_NEED" in set(plan.get("reason_codes") or []):
        return "既然目前没有旅行携带需求，我不会把旅行分装瓶作为连带建议。"
    if plan.get("route_id") == "S01":
        return (
            "根据你确认的敏感、受损或当前不适情况，资料明确不建议使用 P201 果酸精华。"
            "这只是在排除这款商品，不代表没有任何护理方向；你仍可以查询其他品类的客观资料。"
        )
    return (
        "根据目前已确认的信息，品牌资料没有能够可靠支持的建议。"
        "我不会为了凑出方案而放宽条件；可以改看指定品类，或由人工进一步确认。"
    )


def render_category_comparison(
    category: str, products: list[dict[str, Any]], needs: dict[str, Any]
) -> str:
    lines: list[str] = []
    for product in products:
        features = "、".join(product.get("features") or [])
        suitability = product.get("suitability") or "适用条件需要进一步确认"
        lines.append(
            f"- **{product['name']}（{product['sku']}）｜{product['price_yuan']} 元**："
            f"{features}；{suitability}"
        )
    questions = {
        "洁面": "你更偏干或有敏感倾向，还是偏油并喜欢清爽肤感？",
        "保湿": "你目前更在意干燥或屏障脆弱，还是偏油并喜欢轻薄肤感？",
        "精华": "你主要想改善什么问题？目前是否敏感、受损或有明显不适？",
        "旅行配件": "你近期是否有明确的出差或旅行携带需求？",
    }
    return (
        f"### {category}候选\n\n"
        + "\n".join(lines)
        + "\n\n这些是资料中的候选差异，还不是正式推荐。\n\n"
        + questions.get(category, "你更看重哪种使用感受或适用方向？")
    )


def render_policy_answer(result: dict[str, Any]) -> str:
    if result.get("status") == "not_provided":
        query = result.get("query") or {}
        topic = query.get("canonical_topic") or query.get("topic") or "这项政策"
        return (
            f"关于{topic}，现有品牌资料没有提供确定信息，"
            f"{result.get('message', '需要向门店或系统确认')}。"
            "这里不能把“资料未提供”解释成一定没有该政策。"
        )
    if result.get("status") == "found":
        texts = [rule.get("source_text", "") for rule in result.get("policies", [])]
        return "\n\n".join(text for text in texts if text)
    return "品牌资料未覆盖这个问题，无法判断，建议人工复核。"


def render_fact_answer(text: str, products: list[dict[str, Any]]) -> str | None:
    if not products:
        return None
    lines: list[str] = []
    for product in products:
        features = "、".join(product.get("features") or [])
        limitations = "；".join(product.get("limitations") or [])
        specification = product.get("specification") or "资料未提供具体规格"
        lines.append(
            f"- **{product['name']}（{product['sku']}）**：{product['price_yuan']} 元，"
            f"规格 {specification}。{features}。重要限制：{limitations}。"
        )
    return "\n".join(lines)


__all__ = [
    "render_category_comparison",
    "render_fact_answer",
    "render_no_supported_option",
    "render_policy_answer",
    "render_question_plan",
    "render_safety_plan",
]
