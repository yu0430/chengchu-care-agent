"""把品牌资料转换为可解释的分阶段规则评估。

规则引擎只回答“已匹配、还缺什么、哪里冲突、是否阻断”，不生成回复，
也不计算价格。这样同一套规则可以同时用于追问规划和最终推荐校验。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


DATA_DIR = Path(__file__).resolve().parent / "data"
with (DATA_DIR / "products.json").open("r", encoding="utf-8") as stream:
    PRODUCTS_DATA = json.load(stream)
with (DATA_DIR / "policies.json").open("r", encoding="utf-8") as stream:
    POLICIES_DATA = json.load(stream)

PRODUCT_BY_SKU = {item["sku"]: item for item in PRODUCTS_DATA["products"]}


@dataclass(frozen=True)
class RuleAssessment:
    rule_id: str
    kind: Literal["combination", "scenario", "product"]
    applicability: Literal["irrelevant", "possible", "matched", "conflicted", "blocked"]
    candidate_items: list[dict[str, Any]] = field(default_factory=list)
    matched_fields: list[str] = field(default_factory=list)
    missing_trigger_fields: list[str] = field(default_factory=list)
    missing_confirmation_fields: list[str] = field(default_factory=list)
    conflicting_fields: list[str] = field(default_factory=list)
    blocking_reasons: list[str] = field(default_factory=list)
    not_preferred_reasons: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    source: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# “触发条件”用于识别可能路线；“确认条件”用于正式推荐。原始依据仍保留在
# policies.json 的 source_text 与 machine_conditions 中。
ROUTE_SPECS: dict[str, dict[str, Any]] = {
    "C01": {
        "kind": "combination",
        "trigger": {
            "skin_tendency": ["偏干"],
            "sensitive_tendency": True,
            "goals": ["简单护理"],
        },
        "confirm": ["budget_yuan"],
    },
    "C02": {
        "kind": "combination",
        "trigger": {
            "skin_tendency": ["偏油"],
            "goals": ["清爽肤感"],
        },
        "confirm": ["budget_yuan", "fragrance_preference"],
        "supported": {"fragrance_preference": ["可接受香味", "不限"]},
    },
    "S01": {
        "kind": "scenario",
        "trigger": {"goals": ["改善粗糙"]},
        "confirm": [
            "acid_experience",
            "skin_state_stable",
            "sensitive_tendency",
            "current_discomfort",
            "skin_damage",
        ],
        "candidate_items": [{"sku": "P201", "quantity": 1}],
    },
    "S02": {
        "kind": "scenario",
        "trigger": {"travel_need": True},
        "confirm": ["travel_need"],
        "candidate_items": [{"sku": "P301", "quantity": 1}],
    },
}


def _is_missing(needs: dict[str, Any], field_name: str) -> bool:
    if field_name == "budget_yuan":
        return needs.get("budget_status", "unknown") == "unknown"
    value = needs.get(field_name)
    return value is None or (field_name == "goals" and not value)


def _matches_value(actual: Any, expected: Any, field_name: str) -> bool:
    if field_name == "goals":
        values = expected if isinstance(expected, list) else [expected]
        return bool(set(actual or []) & set(values))
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def _matching_rule(rule_id: str) -> dict[str, Any]:
    for group in (POLICIES_DATA["combinations"], POLICIES_DATA["scenario_rules"]):
        for rule in group:
            if rule["id"] == rule_id:
                return rule
    raise KeyError(rule_id)


def _global_safety_reasons(needs: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if needs.get("current_discomfort") is True:
        reasons.append("顾客明确提到当前有明显不适。")
    if needs.get("skin_damage") is True:
        reasons.append("顾客明确提到皮肤受损。")
    if needs.get("persistent_issue") is True:
        reasons.append("顾客明确提到问题持续或反复出现。")
    return reasons


def assess_route(needs: dict[str, Any], rule_id: str) -> RuleAssessment:
    spec = ROUTE_SPECS[rule_id]
    rule = _matching_rule(rule_id)
    trigger = spec["trigger"]
    matched: list[str] = []
    missing_trigger: list[str] = []
    conflicts: list[str] = []

    for field_name, expected in trigger.items():
        if _is_missing(needs, field_name):
            missing_trigger.append(field_name)
        elif _matches_value(needs.get(field_name), expected, field_name):
            matched.append(field_name)
        else:
            conflicts.append(field_name)

    supported = spec.get("supported", {})
    for field_name, expected in supported.items():
        if not _is_missing(needs, field_name) and not _matches_value(
            needs.get(field_name), expected, field_name
        ):
            conflicts.append(field_name)

    candidate_items = deepcopy(spec.get("candidate_items") or rule.get("items") or [])
    if needs.get("fragrance_requirement") == "必须无香":
        if any(
            PRODUCT_BY_SKU[item["sku"]].get("derived_tags", {}).get("fragrance")
            not in {None, "none"}
            for item in candidate_items
        ):
            conflicts.append("fragrance_requirement")

    blocking = _global_safety_reasons(needs)
    machine = rule.get("machine_conditions", {})
    for entry in machine.get("blocked_if_any", []):
        if all(
            not _is_missing(needs, name)
            and _matches_value(needs.get(name), expected, name)
            for name, expected in entry.get("when", {}).items()
        ):
            blocking.append(entry["reason"])

    not_preferred: list[str] = []
    for entry in machine.get("not_preferred_if_any", []):
        if all(
            not _is_missing(needs, name)
            and _matches_value(needs.get(name), expected, name)
            for name, expected in entry.get("when", {}).items()
        ):
            not_preferred.append(entry["reason"])

    missing_confirm = [
        name for name in spec.get("confirm", []) if _is_missing(needs, name)
    ]
    if blocking:
        applicability = "blocked"
    elif conflicts:
        applicability = "conflicted"
    elif not matched:
        applicability = "irrelevant"
    elif missing_trigger or missing_confirm:
        applicability = "possible"
    else:
        applicability = "matched"

    return RuleAssessment(
        rule_id=rule_id,
        kind=spec["kind"],
        applicability=applicability,
        candidate_items=candidate_items,
        matched_fields=matched,
        missing_trigger_fields=missing_trigger,
        missing_confirmation_fields=missing_confirm,
        conflicting_fields=list(dict.fromkeys(conflicts)),
        blocking_reasons=list(dict.fromkeys(blocking)),
        not_preferred_reasons=list(dict.fromkeys(not_preferred)),
        cautions=[],
        source=deepcopy(rule.get("source")),
    )


def assess_routes(needs: dict[str, Any]) -> list[RuleAssessment]:
    return [assess_route(needs, rule_id) for rule_id in ROUTE_SPECS]


def evaluate_machine_conditions(
    needs: dict[str, Any], rule: dict[str, Any]
) -> dict[str, list[str]]:
    """最终校验用的完整条件评估；不计算价格。"""
    missing: list[str] = []
    violations: list[str] = []
    blocked: list[str] = []
    not_preferred: list[str] = []
    cautions: list[str] = []

    for field_name in rule.get("required_confirmations", []):
        if _is_missing(needs, field_name):
            missing.append(field_name)

    required_all = rule.get("required_all", {})
    for field_name, expected in required_all.items():
        if _is_missing(needs, field_name):
            missing.append(field_name)
        elif not _matches_value(needs.get(field_name), expected, field_name):
            violations.append(field_name)

    required_any = rule.get("required_any", [])
    if required_any:
        matches = []
        possible_fields: list[str] = []
        for condition in required_any:
            condition_missing = [name for name in condition if _is_missing(needs, name)]
            possible_fields.extend(condition_missing)
            matches.append(
                not condition_missing
                and all(
                    _matches_value(needs.get(name), expected, name)
                    for name, expected in condition.items()
                )
            )
        if not any(matches):
            if possible_fields:
                missing.extend(possible_fields)
            else:
                violations.append("required_any")

    goals_any = rule.get("goals_any", [])
    if goals_any:
        if not needs.get("goals"):
            missing.append("goals")
        elif not set(needs["goals"]) & set(goals_any):
            violations.append("goals")

    for target, output in (
        ("blocked_if_any", blocked),
        ("not_preferred_if_any", not_preferred),
        ("cautions", cautions),
    ):
        for entry in rule.get(target, []):
            if all(
                not _is_missing(needs, name)
                and _matches_value(needs.get(name), expected, name)
                for name, expected in entry.get("when", {}).items()
            ):
                output.append(entry["reason"])

    return {
        "missing": list(dict.fromkeys(missing)),
        "violations": list(dict.fromkeys(violations)),
        "blocked": list(dict.fromkeys(blocked)),
        "not_preferred": list(dict.fromkeys(not_preferred)),
        "cautions": list(dict.fromkeys(cautions)),
    }


__all__ = [
    "PRODUCT_BY_SKU",
    "POLICIES_DATA",
    "RuleAssessment",
    "assess_route",
    "assess_routes",
    "evaluate_machine_conditions",
]
