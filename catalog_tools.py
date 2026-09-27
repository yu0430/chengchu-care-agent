"""澄初个人护理的商品、政策与报价工具。

本模块只处理可核验的业务事实：商品信息来自 data/products.json，政策来自
data/policies.json，金额由 Python 按分计算。模型负责理解问题和选择工具，
不能向这些函数传入自拟单价、折扣或政策结论。
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from langchain_core.tools import tool


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PRODUCTS_FILE = DATA_DIR / "products.json"
POLICIES_FILE = DATA_DIR / "policies.json"


class CatalogDataError(RuntimeError):
    """商品或政策源数据不完整、不一致时抛出。"""


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON，并把文件问题转换为容易定位的启动错误。"""
    if not path.is_file():
        raise CatalogDataError(f"数据文件不存在：{path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise CatalogDataError(
            f"JSON 格式错误：{path}（第 {exc.lineno} 行，第 {exc.colno} 列）"
        ) from exc
    except OSError as exc:
        raise CatalogDataError(f"无法读取数据文件：{path}") from exc

    if not isinstance(data, dict):
        raise CatalogDataError(f"数据文件根节点必须是对象：{path}")
    return data


def _is_int(value: Any) -> bool:
    """bool 是 int 的子类，但不能作为价格或数量。"""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_source(source: Any, context: str) -> None:
    required = {"document", "page", "section", "locator"}
    if not isinstance(source, dict) or required - source.keys():
        raise CatalogDataError(f"{context} 缺少完整来源字段：{sorted(required)}")
    if not _is_int(source["page"]) or source["page"] <= 0:
        raise CatalogDataError(f"{context} 的来源页码必须是正整数")


def _validate_machine_conditions(value: Any, context: str) -> None:
    required_keys = {
        "required_all",
        "required_any",
        "goals_any",
        "required_confirmations",
        "blocked_if_any",
    }
    if not isinstance(value, dict) or required_keys - value.keys():
        raise CatalogDataError(f"{context} 缺少完整 machine conditions")
    if not isinstance(value["required_all"], dict):
        raise CatalogDataError(f"{context}.required_all 必须是对象")
    if not isinstance(value["required_any"], list):
        raise CatalogDataError(f"{context}.required_any 必须是列表")
    if not isinstance(value["goals_any"], list):
        raise CatalogDataError(f"{context}.goals_any 必须是列表")
    if not isinstance(value["required_confirmations"], list):
        raise CatalogDataError(f"{context}.required_confirmations 必须是列表")
    if not isinstance(value["blocked_if_any"], list):
        raise CatalogDataError(f"{context}.blocked_if_any 必须是列表")
    for condition_group in ("blocked_if_any", "not_preferred_if_any", "cautions"):
        entries = value.get(condition_group, [])
        if not isinstance(entries, list):
            raise CatalogDataError(f"{context}.{condition_group} 必须是列表")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("when"), dict) or not isinstance(entry.get("reason"), str):
                raise CatalogDataError(f"{context}.{condition_group} 包含无效条件")


def _validate_catalog_data(
    products_data: dict[str, Any], policies_data: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """校验两份数据的一致性，并返回 SKU 索引。"""
    if products_data.get("currency") != "CNY":
        raise CatalogDataError("products.json 的 currency 必须是 CNY")
    if policies_data.get("currency") != products_data.get("currency"):
        raise CatalogDataError("商品与政策文件的 currency 不一致")

    products = products_data.get("products")
    if not isinstance(products, list) or not products:
        raise CatalogDataError("products.json 缺少非空 products 列表")

    required_product_fields = {
        "sku",
        "name",
        "name_status",
        "specification",
        "price_yuan",
        "price_fen",
        "features",
        "suitability",
        "limitations",
        "derived_tags",
        "recommendation_conditions",
        "source",
    }
    product_by_sku: dict[str, dict[str, Any]] = {}
    for index, product in enumerate(products):
        context = f"products[{index}]"
        if not isinstance(product, dict):
            raise CatalogDataError(f"{context} 必须是对象")
        missing = required_product_fields - product.keys()
        if missing:
            raise CatalogDataError(f"{context} 缺少字段：{sorted(missing)}")

        sku = product["sku"]
        if not isinstance(sku, str) or not sku.strip():
            raise CatalogDataError(f"{context} 的 sku 必须是非空字符串")
        if sku != sku.strip().upper():
            raise CatalogDataError(f"SKU 必须使用去除空格后的大写格式：{sku!r}")
        if sku in product_by_sku:
            raise CatalogDataError(f"发现重复 SKU：{sku}")

        if not isinstance(product["name"], str) or not product["name"].strip():
            raise CatalogDataError(f"{sku} 的 name 必须是非空字符串")
        if not _is_int(product["price_yuan"]) or product["price_yuan"] < 0:
            raise CatalogDataError(f"{sku} 的 price_yuan 必须是非负整数")
        if not _is_int(product["price_fen"]) or product["price_fen"] < 0:
            raise CatalogDataError(f"{sku} 的 price_fen 必须是非负整数")
        if product["price_fen"] != product["price_yuan"] * 100:
            raise CatalogDataError(f"{sku} 的 price_yuan 与 price_fen 不一致")
        if not isinstance(product["features"], list):
            raise CatalogDataError(f"{sku} 的 features 必须是列表")
        if not isinstance(product["limitations"], list) or not product["limitations"]:
            raise CatalogDataError(f"{sku} 必须保留至少一条 limitations")
        if not isinstance(product["derived_tags"], dict):
            raise CatalogDataError(f"{sku} 的 derived_tags 必须是对象")
        _validate_machine_conditions(product["recommendation_conditions"], f"{sku}.recommendation_conditions")
        _require_source(product["source"], sku)
        product_by_sku[sku] = product

    policy_groups = {
        "brand_principles",
        "product_rules",
        "combinations",
        "scenario_rules",
        "unprovided_policies",
    }
    for group_name in policy_groups:
        if not isinstance(policies_data.get(group_name), list):
            raise CatalogDataError(f"policies.json 的 {group_name} 必须是列表")
        for rule in policies_data[group_name]:
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
                raise CatalogDataError(f"{group_name} 包含无效规则")
            _require_source(rule.get("source"), f"规则 {rule.get('id', '?')}")

    for group_name in ("product_rules", "scenario_rules"):
        for rule in policies_data[group_name]:
            skus = rule.get("skus")
            if not isinstance(skus, list) or not skus:
                raise CatalogDataError(f"规则 {rule['id']} 缺少 skus")
            unknown = [sku for sku in skus if sku not in product_by_sku]
            if unknown:
                raise CatalogDataError(f"规则 {rule['id']} 引用了未知 SKU：{unknown}")

    for combination in policies_data["combinations"]:
        _validate_machine_conditions(combination.get("machine_conditions"), f"组合 {combination['id']}")
        items = combination.get("items")
        if not isinstance(items, list) or not items:
            raise CatalogDataError(f"组合 {combination['id']} 缺少商品")
        calculated_fen = 0
        for item in items:
            if not isinstance(item, dict):
                raise CatalogDataError(f"组合 {combination['id']} 包含无效商品")
            sku = item.get("sku")
            quantity = item.get("quantity")
            if sku not in product_by_sku:
                raise CatalogDataError(
                    f"组合 {combination['id']} 引用了未知 SKU：{sku}"
                )
            if not _is_int(quantity) or quantity <= 0:
                raise CatalogDataError(
                    f"组合 {combination['id']} 中 {sku} 的数量必须是正整数"
                )
            calculated_fen += product_by_sku[sku]["price_fen"] * quantity
        if combination.get("total_fen") != calculated_fen:
            raise CatalogDataError(f"组合 {combination['id']} 的总价与商品单价不一致")
        if combination.get("total_yuan") * 100 != calculated_fen:
            raise CatalogDataError(f"组合 {combination['id']} 的元与分金额不一致")

    for scenario in policies_data["scenario_rules"]:
        _validate_machine_conditions(scenario.get("machine_conditions"), f"场景 {scenario['id']}")

    for policy in policies_data["unprovided_policies"]:
        if policy.get("status") != "not_provided":
            raise CatalogDataError(f"未提供政策 {policy['id']} 的 status 必须为 not_provided")
        if policy.get("response") != "需要向门店或系统确认":
            raise CatalogDataError(f"未提供政策 {policy['id']} 缺少规定回复")

    return product_by_sku


PRODUCTS_DATA = _load_json(PRODUCTS_FILE)
POLICIES_DATA = _load_json(POLICIES_FILE)
PRODUCT_BY_SKU = _validate_catalog_data(PRODUCTS_DATA, POLICIES_DATA)


# 模型或用户可能使用“洗面奶”“面霜”等自然说法，统一映射到目录标签。
# 这里只处理同义词，不根据肤质推断品类。
CATEGORY_ALIASES = {
    "洁面": "洁面",
    "洁面产品": "洁面",
    "洗面奶": "洁面",
    "洁面乳": "洁面",
    "洁面啫喱": "洁面",
    "保湿": "保湿",
    "保湿产品": "保湿",
    "保湿霜": "保湿",
    "保湿乳": "保湿",
    "面霜": "保湿",
    "凝露": "保湿",
    "精华": "精华",
    "精华液": "精华",
    "果酸精华": "精华",
    "旅行配件": "旅行配件",
    "旅行用品": "旅行配件",
    "分装瓶": "旅行配件",
}

FRAGRANCE_ALIASES = {
    "none": "none",
    "无香": "none",
    "无香型": "none",
    "scented": "scented",
    "有香": "scented",
    "带香味": "scented",
    "lightly_scented": "lightly_scented",
    "淡香": "lightly_scented",
    "带淡香": "lightly_scented",
    "accepts_scent": "accepts_scent",
    "可接受香味": "accepts_scent",
    "接受香味": "accepts_scent",
    "any": "any",
    "不限": "any",
}

POLICY_ALIASES = {
    "优惠": "促销",
    "折扣": "促销",
    "活动": "促销",
    "送东西": "赠品",
    "礼物": "赠品",
    "网店价格": "线上同价",
    "网上价格": "线上同价",
    "价格保护": "线上同价",
    "价保": "线上同价",
    "退款": "退换货",
    "换货": "退换货",
    "授权": "渠道授权",
    "官方授权": "渠道授权",
}


def _error(status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"status": status, "message": message, **details}


def _yuan_to_fen(value: Any, field_name: str) -> int:
    """把最多两位小数的元转换成整数分；输入不合法时抛 ValueError。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{field_name} 必须是非负金额")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} 必须是非负金额") from exc
    if not amount.is_finite() or amount < 0:
        raise ValueError(f"{field_name} 必须是非负金额")
    fen = amount * 100
    if fen != fen.to_integral_value():
        raise ValueError(f"{field_name} 最多支持两位小数")
    return int(fen)


def _display_yuan(fen: int) -> int | float:
    return fen // 100 if fen % 100 == 0 else fen / 100


def search_products_data(
    sku: str | None = None,
    category: str | None = None,
    max_price_yuan: float | None = None,
    fragrance_preference: str | None = None,
) -> dict[str, Any]:
    """按 SKU、品类、单件最高价格和香味偏好查询商品事实。"""
    normalized_sku = sku.strip().upper() if isinstance(sku, str) else None
    raw_category = category.strip().casefold() if isinstance(category, str) else None
    normalized_category = CATEGORY_ALIASES.get(raw_category, raw_category)

    max_price_fen = None
    if max_price_yuan is not None:
        try:
            max_price_fen = _yuan_to_fen(max_price_yuan, "max_price_yuan")
        except ValueError as exc:
            return _error("invalid_request", str(exc))

    fragrance = None
    if fragrance_preference is not None:
        if not isinstance(fragrance_preference, str):
            return _error("invalid_request", "fragrance_preference 必须是字符串")
        fragrance = FRAGRANCE_ALIASES.get(fragrance_preference.strip().casefold())
        if fragrance is None:
            return _error(
                "invalid_filter",
                "无法识别香味偏好",
                allowed_values=["无香", "有香", "淡香", "可接受香味", "不限"],
            )

    results = []
    for product in PRODUCTS_DATA["products"]:
        if normalized_sku and product["sku"] != normalized_sku:
            continue
        product_category = str(product["derived_tags"].get("category", "")).casefold()
        if normalized_category and product_category != normalized_category:
            continue
        if max_price_fen is not None and product["price_fen"] > max_price_fen:
            continue
        product_fragrance = product["derived_tags"].get("fragrance")
        if fragrance == "none" and product_fragrance != "none":
            continue
        if fragrance == "scented" and product_fragrance != "scented":
            continue
        if fragrance == "lightly_scented" and product_fragrance != "lightly_scented":
            continue
        if fragrance == "accepts_scent" and product_fragrance not in {
            "scented",
            "lightly_scented",
        }:
            continue
        results.append(deepcopy(product))

    applied_filters = {
        key: value
        for key, value in {
            "sku": normalized_sku,
            "category": category.strip() if isinstance(category, str) else None,
            "max_price_yuan": max_price_yuan,
            "fragrance_preference": fragrance_preference,
        }.items()
        if value is not None
    }
    status = "success" if results else "no_match"
    return {
        "status": status,
        "count": len(results),
        "applied_filters": applied_filters,
        "products": results,
        "available_categories": sorted(
            {item["derived_tags"]["category"] for item in PRODUCTS_DATA["products"]}
        ),
        "message": (
            "找到符合条件的商品。"
            if results
            else "没有符合全部条件的商品；工具未自动放宽筛选条件。"
        ),
    }


def _policy_query_terms(topic: str) -> list[str]:
    """把“敏感倾向与简单护理组合边界”拆成可核对的规则关键词。"""
    normalized = topic.strip().casefold()
    for suffix in ("组合边界", "相关政策", "相关规则", "使用限制", "政策", "规则", "限制", "边界"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
            break
    terms = [
        term.strip()
        for term in re.split(r"(?:\s+|[、,，/]|以及|与|和)", normalized)
        if term.strip()
    ]
    return terms or [topic.strip().casefold()]


def _rule_text(rule: dict[str, Any]) -> str:
    fields = (
        "title",
        "topic",
        "source_text",
        "condition",
        "customer_situation",
        "recommendation_scope",
        "effect",
    )
    return " ".join(str(rule.get(field, "")) for field in fields).casefold()


def get_policy_data(
    topic: str | None = None, sku: str | None = None
) -> dict[str, Any]:
    """查询品牌原则、商品规则、推荐场景、组合条件或资料缺失政策。"""
    if topic is None and sku is None:
        return _error("invalid_request", "topic 和 sku 至少提供一个")

    normalized_sku = sku.strip().upper() if isinstance(sku, str) else None
    if normalized_sku and normalized_sku not in PRODUCT_BY_SKU:
        return _error("invalid_sku", f"商品目录中不存在 SKU：{normalized_sku}")

    normalized_topic = topic.strip() if isinstance(topic, str) else None
    if topic is not None and not normalized_topic:
        return _error("invalid_request", "topic 不能为空")
    canonical_topic = POLICY_ALIASES.get(normalized_topic, normalized_topic)
    topic_key = canonical_topic.casefold() if canonical_topic else None
    topic_terms = _policy_query_terms(canonical_topic) if canonical_topic else []

    # 手册明确声明未提供的政策优先返回，避免把“未提供”误说成“不支持”。
    if topic_key:
        for policy in POLICIES_DATA["unprovided_policies"]:
            if policy["topic"].casefold() == topic_key:
                return {
                    "status": "not_provided",
                    "query": {"topic": topic, "canonical_topic": policy["topic"]},
                    "message": policy["response"],
                    "policies": [deepcopy(policy)],
                }

    matches: list[dict[str, Any]] = []
    groups = (
        "brand_principles",
        "product_rules",
        "combinations",
        "scenario_rules",
    )
    for group_name in groups:
        for rule in POLICIES_DATA[group_name]:
            rule_skus = set(rule.get("skus", []))
            if group_name == "combinations":
                rule_skus = {item["sku"] for item in rule.get("items", [])}
            if normalized_sku and normalized_sku not in rule_skus:
                continue
            rule_text = _rule_text(rule)
            if topic_terms and not all(term in rule_text for term in topic_terms):
                continue
            match = deepcopy(rule)
            match["rule_group"] = group_name
            matches.append(match)

    if not matches:
        return {
            "status": "not_found",
            "query": {"topic": topic, "sku": normalized_sku},
            "message": "品牌手册中未找到相关规则，不能据此补充或推断政策。",
            "policies": [],
        }
    return {
        "status": "found",
        "query": {"topic": topic, "sku": normalized_sku},
        "message": "已找到品牌手册中的相关规则。",
        "policies": matches,
    }


def calculate_quote_data(
    items: list[dict[str, Any]], budget_yuan: float | None = None
) -> dict[str, Any]:
    """从商品目录读取权威单价，按整数分计算明细、总价与预算差额。"""
    if not isinstance(items, list) or not items:
        return _error("invalid_items", "items 必须是非空商品列表")

    budget_fen = None
    if budget_yuan is not None:
        try:
            budget_fen = _yuan_to_fen(budget_yuan, "budget_yuan")
        except ValueError as exc:
            return _error("invalid_budget", str(exc))

    # 相同 SKU 自动合并，避免重复行造成回答混乱。
    quantities: dict[str, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            return _error("invalid_items", f"items[{index}] 必须是对象")
        sku = item.get("sku")
        quantity = item.get("quantity", 1)
        if not isinstance(sku, str) or not sku.strip():
            return _error("invalid_sku", f"items[{index}] 缺少有效 SKU")
        normalized_sku = sku.strip().upper()
        if normalized_sku not in PRODUCT_BY_SKU:
            return _error(
                "invalid_sku",
                f"商品目录中不存在 SKU：{normalized_sku}",
                sku=normalized_sku,
            )
        if not _is_int(quantity) or quantity <= 0:
            return _error(
                "invalid_quantity",
                f"{normalized_sku} 的 quantity 必须是正整数",
                sku=normalized_sku,
            )
        quantities[normalized_sku] = quantities.get(normalized_sku, 0) + quantity

    quote_items = []
    total_fen = 0
    for sku, quantity in quantities.items():
        product = PRODUCT_BY_SKU[sku]
        subtotal_fen = product["price_fen"] * quantity
        total_fen += subtotal_fen
        quote_items.append(
            {
                "sku": sku,
                "name": product["name"],
                "unit_price_yuan": product["price_yuan"],
                "quantity": quantity,
                "subtotal_yuan": _display_yuan(subtotal_fen),
                "source": deepcopy(product["source"]),
            }
        )

    result: dict[str, Any] = {
        "status": "success",
        "currency": PRODUCTS_DATA["currency"],
        "items": quote_items,
        "total_yuan": _display_yuan(total_fen),
        "pricing_note": "单价来自商品目录；未应用任何折扣或促销。",
    }
    if budget_fen is not None:
        difference_fen = budget_fen - total_fen
        result.update(
            {
                "budget_yuan": _display_yuan(budget_fen),
                "within_budget": difference_fen >= 0,
                "remaining_budget_yuan": _display_yuan(max(difference_fen, 0)),
                "over_budget_yuan": _display_yuan(max(-difference_fen, 0)),
            }
        )
    return result


@tool
def search_products(
    sku: str | None = None,
    category: str | None = None,
    max_price_yuan: float | None = None,
    fragrance_preference: str | None = None,
) -> dict[str, Any]:
    """查询澄初商品事实。需要商品名称、类别、单价、特点或候选商品时调用。

    可按 SKU、品类、单件最高价格和香味偏好筛选；目录品类为“洁面、保湿、
    精华、旅行配件”，工具也接受洗面奶、面霜等常见同义词。不要用它判断医疗问题。
    返回的适用提示、限制与来源必须一起用于回答。
    """
    return search_products_data(
        sku=sku,
        category=category,
        max_price_yuan=max_price_yuan,
        fragrance_preference=fragrance_preference,
    )


@tool
def get_policy(topic: str | None = None, sku: str | None = None) -> dict[str, Any]:
    """查询澄初品牌原则、商品限制、组合条件和政策边界。

    用户询问敏感或不适、果酸使用、组合推荐、促销、赠品、线上同价、
    退换货或渠道授权时调用。not_provided 表示资料未提供，不表示不支持。
    """
    return get_policy_data(topic=topic, sku=sku)


@tool
def calculate_quote(
    items: list[dict[str, Any]], budget_yuan: float | None = None
) -> dict[str, Any]:
    """按商品 SKU 和数量计算澄初商品总价及预算差额。

    用户询问单品或组合价格、总价、预算是否足够时调用。只传 SKU、数量和
    可选预算；单价由工具从目录读取，禁止自行传入价格、折扣或促销。
    """
    return calculate_quote_data(items=items, budget_yuan=budget_yuan)


__all__ = [
    "CatalogDataError",
    "calculate_quote",
    "calculate_quote_data",
    "get_policy",
    "get_policy_data",
    "search_products",
    "search_products_data",
]
