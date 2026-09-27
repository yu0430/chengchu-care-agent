"""从最新用户原话中提取高置信度的明确需求事实。

这是需求提取的规则通道，只处理无需推断的直接表达。灵活或含糊的自然语言仍由
Qwen 结构化提取通道处理；两路结果最终都要经过 customer_needs 的原话校验。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhraseRule:
    field: str
    value: Any
    patterns: tuple[str, ...]
    ignore_if_negated: bool = False


# 只收录能够直接支持标准字段值的表达。这里不写商品适配逻辑，也不从肤质、
# 偏好或商品特征推断另一个字段。
PHRASE_RULES = (
    PhraseRule(
        "selection_scope",
        "单品",
        (
            r"只看单品",
            r"只买单品",
            r"只要一件",
        ),
        True,
    ),
    PhraseRule(
        "selection_scope",
        "完整护理",
        (r"推荐一套", r"搭配一套", r"配一套", r"完整护理", r"整套护理", r"基础护理组合"),
        True,
    ),
    PhraseRule("skin_tendency", "偏干", (r"偏干", r"干性(?:皮肤|肤质)", r"干皮"), True),
    PhraseRule("skin_tendency", "偏油", (r"偏油", r"油性(?:皮肤|肤质)", r"油皮"), True),
    PhraseRule("skin_tendency", "中性", (r"中性(?:皮肤|肤质)",), True),
    PhraseRule("skin_tendency", "混合", (r"混合(?:性)?(?:皮肤|肤质)", r"混合皮"), True),
    PhraseRule(
        "sensitive_tendency",
        False,
        (r"没有敏感倾向", r"无敏感倾向", r"不是敏感肌"),
    ),
    PhraseRule(
        "sensitive_tendency",
        True,
        (r"有敏感倾向", r"敏感倾向", r"属于敏感肌", r"我是敏感肌", r"皮肤敏感"),
        True,
    ),
    PhraseRule(
        "current_discomfort",
        False,
        (r"目前没有明显不适", r"现在没有明显不适", r"没有明显不适", r"无明显不适"),
    ),
    PhraseRule(
        "current_discomfort",
        True,
        (r"目前有明显不适", r"现在有明显不适", r"正在明显不适", r"出现明显不适"),
        True,
    ),
    PhraseRule(
        "skin_damage",
        False,
        (r"皮肤没有受损", r"没有皮肤受损", r"没有受损", r"无受损"),
    ),
    PhraseRule(
        "skin_damage",
        True,
        (r"皮肤已经受损", r"皮肤受损了", r"皮肤有受损"),
        True,
    ),
    PhraseRule(
        "persistent_issue",
        False,
        (r"没有持续问题", r"没有反复出现", r"不是持续不适"),
    ),
    PhraseRule(
        "persistent_issue",
        True,
        (r"持续不适", r"持续问题", r"持续(?:出现|存在)(?:问题|症状|不适)", r"反复(?:出现|不适)"),
        True,
    ),
    PhraseRule(
        "skin_state_stable",
        False,
        (r"皮肤状态不稳定", r"状态不稳定"),
    ),
    PhraseRule(
        "skin_state_stable",
        True,
        (r"皮肤状态稳定", r"状态比较稳定", r"状态稳定"),
        True,
    ),
    PhraseRule(
        "barrier_fragile",
        True,
        (r"屏障比较脆弱", r"屏障脆弱", r"屏障受损倾向"),
        True,
    ),
    PhraseRule(
        "barrier_fragile",
        False,
        (r"没有屏障脆弱倾向", r"屏障不脆弱"),
    ),
    PhraseRule(
        "fragrance_sensitive",
        False,
        (r"对香味不敏感", r"对香味没有敏感", r"香味不会让我不舒服"),
    ),
    PhraseRule(
        "fragrance_sensitive",
        True,
        (r"对香味敏感", r"香味会让我不舒服"),
        True,
    ),
    PhraseRule(
        "acid_experience",
        False,
        (r"没有用过果酸", r"没用过果酸", r"没有焕肤经验"),
    ),
    PhraseRule(
        "acid_experience",
        True,
        (r"有果酸使用经验", r"用过果酸", r"有焕肤经验"),
        True,
    ),
    PhraseRule(
        "travel_need",
        False,
        (r"没有旅行需求", r"没有出差需求", r"近期不旅行"),
    ),
    PhraseRule(
        "travel_need",
        True,
        (r"有旅行需求", r"需要出差", r"准备旅行", r"旅行携带"),
        True,
    ),
    PhraseRule(
        "fragrance_preference",
        "无香",
        (r"偏好无香", r"喜欢无香", r"想要无香", r"只接受无香"),
        True,
    ),
    PhraseRule(
        "fragrance_requirement",
        "必须无香",
        (r"只接受无香", r"必须无香", r"不要有香味", r"不能有香味"),
        True,
    ),
    PhraseRule(
        "fragrance_requirement",
        "偏好无香",
        (r"偏好无香", r"喜欢无香", r"想要无香"),
        True,
    ),
    PhraseRule(
        "fragrance_preference",
        "可接受香味",
        (r"可以接受香味", r"能接受香味", r"不介意香味"),
        True,
    ),
    PhraseRule(
        "fragrance_requirement",
        "无硬性要求",
        (r"可以接受香味", r"能接受香味", r"不介意香味", r"香味不限", r"香味都可以"),
        True,
    ),
    PhraseRule(
        "fragrance_preference",
        "不限",
        (r"香味不限", r"香味都可以"),
        True,
    ),
    PhraseRule("goals", ["简单护理"], (r"简单护理", r"护理简单一点"), True),
    PhraseRule(
        "goals",
        ["清爽肤感"],
        (r"清爽肤感", r"喜欢清爽", r"偏好清爽", r"清爽感", r"清爽"),
        True,
    ),
    PhraseRule(
        "goals",
        ["轻薄肤感"],
        (r"轻薄肤感", r"喜欢轻薄", r"偏好轻薄", r"轻薄"),
        True,
    ),
    PhraseRule("goals", ["改善粗糙"], (r"改善粗糙", r"皮肤粗糙"), True),
    PhraseRule(
        "budget_status",
        "unlimited",
        (r"预算不限", r"预算没有上限", r"不设预算上限"),
        True,
    ),
    PhraseRule(
        "budget_status",
        "declined",
        (r"不想(?:提供|说)预算", r"不方便说预算", r"先不谈预算"),
        True,
    ),
)

# 品类路线判断使用“选择/购买意图 + 品类实体”，并覆盖“品类 + 怎么选”的倒装
# 说法。这里负责高置信度直述，口语化同义表达仍交给 Qwen 的语义提取通道。
CATEGORY_REQUEST_PATTERNS: dict[str, tuple[str, ...]] = {
    "洁面": (
        r"(?:只(?:想)?(?:买|要|看|选)|想(?:买|看|选|找)|买|看看|推荐|找)"
        r"(?:一件|一个|一款)?\s*(?:洁面|洗面奶|洗脸产品|清洁产品)",
        r"(?:洁面|洗面奶|洗脸产品)(?:怎么选|如何选|选哪个|哪个好)",
    ),
    "保湿": (
        r"(?:只(?:想)?(?:买|要|看|选)|想(?:买|看|选|找)|买|看看|推荐|找)"
        r"(?:一件|一个|一款)?\s*(?:保湿(?:乳|产品)?|乳液|面霜)",
        r"(?:保湿(?:乳|产品)?|乳液|面霜)(?:怎么选|如何选|选哪个|哪个好)",
    ),
    "精华": (
        r"(?:只(?:想)?(?:买|要|看|选)|想(?:买|看|选|找)|买|看看|推荐|找)"
        r"(?:一件|一个|一款)?\s*(?:精华|精华液)",
        r"(?:精华|精华液)(?:怎么选|如何选|选哪个|哪个好)",
    ),
    "旅行配件": (
        r"(?:只(?:想)?(?:买|要|看|选)|想(?:买|看|选|找)|买|看看|推荐|找)"
        r"(?:一件|一个|一款)?\s*(?:旅行配件|旅行装|分装瓶)",
        r"(?:旅行配件|旅行装|分装瓶)(?:怎么选|如何选|选哪个|哪个好)",
    ),
}

BUDGET_PATTERN = re.compile(
    r"(?:预算(?:是|为|大概|大约|可以提高到|提高到|调整到|改成)?\s*"
    r"[¥￥]?\s*(\d+(?:\.\d{1,2})?)\s*元?"
    r"|[¥￥]?\s*(\d+(?:\.\d{1,2})?)\s*元?\s*(?:的)?预算)"
)
UNCERTAINTY_WORDS = ("不确定", "不知道", "不清楚", "说不准", "可能")
NEGATION_PREFIXES = ("不是", "并非", "并不", "不算", "没有", "没", "无")


def _context_window(text: str, start: int, end: int, radius: int = 8) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _is_uncertain(text: str, start: int, end: int) -> bool:
    window = _context_window(text, start, end)
    return any(word in window for word in UNCERTAINTY_WORDS)


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 4) : start]
    return any(prefix.endswith(word) for word in NEGATION_PREFIXES)


def _collect_phrase_candidates(text: str) -> dict[str, list[dict[str, Any]]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    for rule in PHRASE_RULES:
        for pattern in rule.patterns:
            for match in re.finditer(pattern, text):
                if _is_uncertain(text, match.start(), match.end()):
                    continue
                if rule.ignore_if_negated and _is_negated(text, match.start()):
                    continue
                candidates.setdefault(rule.field, []).append(
                    {
                        "value": rule.value,
                        "evidence": match.group(0),
                        "start": match.start(),
                        "length": match.end() - match.start(),
                    }
                )
    return candidates


def _resolve_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    # 同一条消息同时出现互相冲突的明确值时不由规则通道裁决，交给模型理解或追问。
    distinct = {repr(item["value"]) for item in candidates}
    if len(distinct) != 1:
        return None
    selected = max(candidates, key=lambda item: (item["length"], item["start"]))
    return {"value": selected["value"], "evidence": selected["evidence"]}


def _extract_budget(text: str) -> dict[str, Any] | None:
    matches = [
        match
        for match in BUDGET_PATTERN.finditer(text)
        if not _is_uncertain(text, match.start(), match.end())
    ]
    if not matches:
        return None
    amounts = {match.group(1) or match.group(2) for match in matches}
    if len(amounts) != 1:
        return None
    match = max(matches, key=lambda item: item.end())
    raw_amount = match.group(1) or match.group(2)
    amount = float(raw_amount)
    value: int | float = int(amount) if amount.is_integer() else amount
    return {"value": value, "evidence": match.group(0)}


def _extract_category_request(text: str) -> dict[str, dict[str, Any]]:
    """识别明确的品类选择请求，并保留能直接核对的完整原话片段。"""
    matches: list[tuple[str, re.Match[str]]] = []
    for category, patterns in CATEGORY_REQUEST_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                if _is_uncertain(text, match.start(), match.end()):
                    continue
                if _is_negated(text, match.start()):
                    continue
                matches.append((category, match))

    categories = {category for category, _match in matches}
    if len(categories) != 1:
        # 同时指定多个品类时，不擅自裁决为哪个单一品类或完整护理。
        return {}
    category = next(iter(categories))
    selected = max(
        (match for item_category, match in matches if item_category == category),
        key=lambda item: (item.end() - item.start(), item.start()),
    )
    evidence = selected.group(0)
    return {
        "desired_category": {"value": category, "evidence": evidence},
        "selection_scope": {"value": "指定品类", "evidence": evidence},
    }


def extract_explicit_fact_updates(text: str) -> dict[str, dict[str, Any]]:
    """返回高置信度更新；含糊或冲突字段保持未提取。"""
    if not isinstance(text, str) or not text.strip():
        return {}
    updates: dict[str, dict[str, Any]] = {}

    skus = list(
        dict.fromkeys(
            match.upper()
            for match in re.findall(
                r"(?<![A-Za-z0-9])P(?:101|102|201|202|203|301)(?![A-Za-z0-9])",
                text,
                flags=re.IGNORECASE,
            )
        )
    )
    if len(skus) == 1:
        match = re.search(skus[0], text, flags=re.IGNORECASE)
        if match:
            updates["desired_sku"] = {
                "value": skus[0],
                "evidence": match.group(0),
            }

    for field, candidates in _collect_phrase_candidates(text).items():
        if field == "goals":
            # 目标可以并存，例如“简单、清爽、轻薄”；按首次出现顺序合并，
            # 证据保留覆盖全部命中项的原话片段。
            ordered = sorted(candidates, key=lambda item: item["start"])
            values: list[str] = []
            for item in ordered:
                for value in item["value"]:
                    if value not in values:
                        values.append(value)
            if values:
                start = min(item["start"] for item in ordered)
                end = max(item["start"] + item["length"] for item in ordered)
                updates[field] = {"value": values, "evidence": text[start:end]}
            continue
        resolved = _resolve_candidates(candidates)
        if resolved:
            updates[field] = resolved
    # 明确的品类购买/选择请求优先于宽泛的范围词；例如“只要洁面”表示限定
    # 在洁面品类内，不等于系统已经知道顾客只购买一件。
    updates.update(_extract_category_request(text))
    budget = _extract_budget(text)
    if budget:
        updates["budget_yuan"] = budget
    return updates


def extract_pending_question_updates(
    text: str, pending_questions: list[str] | tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    """在问题上下文唯一明确时解析简短回答，避免要求顾客重复字段名。"""
    if not isinstance(text, str) or not text.strip():
        return {}
    pending = list(dict.fromkeys(pending_questions))
    normalized = text.strip().strip("。！!？?")
    updates: dict[str, dict[str, Any]] = {}

    if "budget_yuan" in pending:
        match = re.fullmatch(r"[¥￥]?\s*(\d+(?:\.\d{1,2})?)\s*元?", normalized)
        if match:
            amount = float(match.group(1))
            value: int | float = int(amount) if amount.is_integer() else amount
            updates["budget_yuan"] = {"value": value, "evidence": text.strip()}

    # “可以”只有在上一轮只问香味时才足够明确；同时问了两项时不能一答多填。
    if pending == ["fragrance_preference"] and normalized in {
        "可以", "能接受", "可以接受", "不介意", "都可以"
    }:
        updates["fragrance_preference"] = {
            "value": "可接受香味",
            "evidence": text.strip(),
        }
        updates["fragrance_requirement"] = {
            "value": "无硬性要求",
            "evidence": text.strip(),
        }
    if "fragrance_preference" in pending and normalized in {"无香", "只要无香", "必须无香"}:
        updates["fragrance_preference"] = {"value": "无香", "evidence": text.strip()}
        updates["fragrance_requirement"] = {
            "value": "必须无香" if normalized != "无香" else "偏好无香",
            "evidence": text.strip(),
        }
    if "selection_scope" in pending:
        complete_term = next(
            (term for term in ("完整护理", "都看看", "一起", "一套", "整套") if term in normalized),
            None,
        )
        single_term = next(
            (term for term in ("只看一件", "只买一件", "单品") if term in normalized),
            None,
        )
        if complete_term:
            updates["selection_scope"] = {
                "value": "完整护理",
                "evidence": complete_term,
            }
        elif single_term:
            updates["selection_scope"] = {
                "value": "单品",
                "evidence": single_term,
            }
    return updates


__all__ = ["extract_explicit_fact_updates", "extract_pending_question_updates"]
