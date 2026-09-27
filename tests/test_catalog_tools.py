"""catalog_tools 的确定性业务测试；不启动 UI，也不调用大模型。"""

import copy
import unittest

import catalog_tools as catalog


class ProductSearchTests(unittest.TestCase):
    def test_exact_sku_contains_price_limit_and_source(self):
        result = catalog.search_products_data(sku="p101")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["count"], 1)
        product = result["products"][0]
        self.assertEqual(product["price_yuan"], 169)
        self.assertTrue(product["limitations"])
        self.assertEqual(product["source"]["page"], 1)

    def test_category_and_no_fragrance_filter(self):
        result = catalog.search_products_data(
            category="洁面", fragrance_preference="无香"
        )
        self.assertEqual([item["sku"] for item in result["products"]], ["P101"])

    def test_natural_category_alias(self):
        result = catalog.search_products_data(category="保湿霜")
        self.assertEqual(
            {item["sku"] for item in result["products"]}, {"P202", "P203"}
        )

    def test_max_price_does_not_relax_filter(self):
        result = catalog.search_products_data(
            category="洁面", max_price_yuan=100
        )
        self.assertEqual(result["status"], "no_match")
        self.assertEqual(result["products"], [])
        self.assertIn("未自动放宽", result["message"])

    def test_accepts_scent_matches_two_scented_products(self):
        result = catalog.search_products_data(fragrance_preference="可接受香味")
        self.assertEqual(
            {item["sku"] for item in result["products"]}, {"P102", "P203"}
        )

    def test_unknown_fragrance_value_is_rejected(self):
        result = catalog.search_products_data(fragrance_preference="完全未知")
        self.assertEqual(result["status"], "invalid_filter")


class PolicyTests(unittest.TestCase):
    def test_gift_policy_is_not_provided(self):
        result = catalog.get_policy_data(topic="赠品")
        self.assertEqual(result["status"], "not_provided")
        self.assertEqual(result["message"], "需要向门店或系统确认")

    def test_online_price_alias_maps_to_policy(self):
        result = catalog.get_policy_data(topic="网上价格")
        self.assertEqual(result["status"], "not_provided")
        self.assertEqual(result["query"]["canonical_topic"], "线上同价")

    def test_p201_returns_product_and_scenario_rules(self):
        result = catalog.get_policy_data(sku="P201")
        self.assertEqual(result["status"], "found")
        ids = {rule["id"] for rule in result["policies"]}
        self.assertTrue({"R201", "R202", "S01"}.issubset(ids))
        combined = " ".join(rule["source_text"] for rule in result["policies"])
        self.assertIn("不推荐", combined)
        self.assertIn("防晒", combined)
        self.assertIn("使用经验", combined)

    def test_composite_natural_language_policy_query(self):
        result = catalog.get_policy_data(topic="敏感倾向与简单护理组合边界")
        self.assertEqual(result["status"], "found")
        self.assertIn("C01", {rule["id"] for rule in result["policies"]})

    def test_unknown_policy_is_not_invented(self):
        result = catalog.get_policy_data(topic="终身免费保养")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["policies"], [])

    def test_unknown_sku_is_rejected(self):
        result = catalog.get_policy_data(sku="P999")
        self.assertEqual(result["status"], "invalid_sku")


class QuoteTests(unittest.TestCase):
    def test_dry_sensitive_combo_exceeds_budget(self):
        result = catalog.calculate_quote_data(
            [
                {"sku": "P101", "quantity": 1},
                {"sku": "P202", "quantity": 1},
            ],
            budget_yuan=400,
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["total_yuan"], 428)
        self.assertFalse(result["within_budget"])
        self.assertEqual(result["over_budget_yuan"], 28)

    def test_oily_combo_total(self):
        result = catalog.calculate_quote_data(
            [{"sku": "P102", "quantity": 1}, {"sku": "P203", "quantity": 1}]
        )
        self.assertEqual(result["total_yuan"], 368)
        self.assertNotIn("within_budget", result)

    def test_duplicate_sku_is_merged(self):
        result = catalog.calculate_quote_data(
            [{"sku": "P301", "quantity": 1}, {"sku": "p301", "quantity": 2}]
        )
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["quantity"], 3)
        self.assertEqual(result["total_yuan"], 147)

    def test_invalid_sku_fails_entire_quote(self):
        result = catalog.calculate_quote_data(
            [{"sku": "P101", "quantity": 1}, {"sku": "P999", "quantity": 1}]
        )
        self.assertEqual(result["status"], "invalid_sku")
        self.assertNotIn("total_yuan", result)

    def test_invalid_quantities_are_rejected(self):
        for quantity in (0, -1, 1.5, True, "2"):
            with self.subTest(quantity=quantity):
                result = catalog.calculate_quote_data(
                    [{"sku": "P101", "quantity": quantity}]
                )
                self.assertEqual(result["status"], "invalid_quantity")

    def test_decimal_budget_uses_integer_fen(self):
        result = catalog.calculate_quote_data(
            [{"sku": "P101", "quantity": 1}], budget_yuan=169.01
        )
        self.assertTrue(result["within_budget"])
        self.assertEqual(result["remaining_budget_yuan"], 0.01)


class DataValidationTests(unittest.TestCase):
    def test_duplicate_sku_is_rejected_at_load_time(self):
        products = copy.deepcopy(catalog.PRODUCTS_DATA)
        products["products"].append(copy.deepcopy(products["products"][0]))
        with self.assertRaisesRegex(catalog.CatalogDataError, "重复 SKU"):
            catalog._validate_catalog_data(products, catalog.POLICIES_DATA)

    def test_inconsistent_price_is_rejected_at_load_time(self):
        products = copy.deepcopy(catalog.PRODUCTS_DATA)
        products["products"][0]["price_fen"] += 1
        with self.assertRaisesRegex(catalog.CatalogDataError, "不一致"):
            catalog._validate_catalog_data(products, catalog.POLICIES_DATA)

    def test_tool_wrappers_are_available_for_langgraph(self):
        self.assertEqual(catalog.search_products.name, "search_products")
        self.assertEqual(catalog.get_policy.name, "get_policy")
        self.assertEqual(catalog.calculate_quote.name, "calculate_quote")


if __name__ == "__main__":
    unittest.main()
