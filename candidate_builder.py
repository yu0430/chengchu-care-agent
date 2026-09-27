"""从已选路线生成候选商品；不负责最终批准或报价。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from rule_engine import PRODUCT_BY_SKU, POLICIES_DATA, evaluate_machine_conditions


@dataclass(frozen=True)
class CandidateProposal:
    status: Literal["ready", "ambiguous", "none"]
    items: list[dict[str, Any]] = field(default_factory=list)
    route_id: str | None = None
    category: str | None = None
    alternatives: list[str] = field(default_factory=list)
    source: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _route_rule(route_id: str) -> dict[str, Any] | None:
    for group in (POLICIES_DATA["combinations"], POLICIES_DATA["scenario_rules"]):
        for rule in group:
            if rule["id"] == route_id:
                return rule
    return None


def proposal_for_route(route_id: str) -> CandidateProposal:
    rule = _route_rule(route_id)
    if not rule:
        return CandidateProposal(status="none", route_id=route_id)
    items = deepcopy(rule.get("items") or [])
    if not items:
        items = [{"sku": sku, "quantity": 1} for sku in rule.get("skus", [])]
    return CandidateProposal(
        status="ready",
        items=items,
        route_id=route_id,
        source=deepcopy(rule.get("source")),
    )


def proposal_for_category(
    needs: dict[str, Any], category: str
) -> CandidateProposal:
    products = [
        product
        for product in PRODUCT_BY_SKU.values()
        if product.get("derived_tags", {}).get("category") == category
    ]
    matched: list[str] = []
    possible: list[str] = []
    for product in products:
        if (
            needs.get("fragrance_requirement") == "必须无香"
            and product.get("derived_tags", {}).get("fragrance") not in {None, "none"}
        ):
            continue
        check = evaluate_machine_conditions(
            needs, product.get("recommendation_conditions") or {}
        )
        if check["blocked"] or check["violations"]:
            continue
        if check["missing"]:
            possible.append(product["sku"])
        else:
            matched.append(product["sku"])

    if len(matched) == 1:
        return CandidateProposal(
            status="ready",
            items=[{"sku": matched[0], "quantity": 1}],
            category=category,
        )
    alternatives = matched or possible or [product["sku"] for product in products]
    return CandidateProposal(
        status="ambiguous" if alternatives else "none",
        category=category,
        alternatives=alternatives,
    )


def build_candidate(plan: dict[str, Any], needs: dict[str, Any]) -> CandidateProposal:
    route_id = plan.get("route_id")
    if route_id:
        return proposal_for_route(route_id)
    category = plan.get("category") or needs.get("desired_category")
    if category:
        return proposal_for_category(needs, category)
    items = plan.get("candidate_items") or []
    if items:
        return CandidateProposal(status="ready", items=deepcopy(items))
    return CandidateProposal(status="none")


__all__ = [
    "CandidateProposal",
    "build_candidate",
    "proposal_for_category",
    "proposal_for_route",
]
