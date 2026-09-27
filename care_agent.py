"""澄初个人护理导购的确定性业务编排。"""

from __future__ import annotations

import json
import re
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from typing_extensions import TypedDict

from candidate_builder import build_candidate, proposal_for_category
from catalog_tools import (
    calculate_quote,
    get_policy,
    get_policy_data,
    search_products,
    search_products_data,
)
from customer_needs import (
    CustomerNeeds,
    empty_customer_needs,
    merge_confirmed_needs,
    set_pending_questions,
)
from decision_planner import plan_next_action
from explicit_fact_parser import (
    extract_explicit_fact_updates,
    extract_pending_question_updates,
)
from llm import get_llm
from needs_extractor import create_needs_extractor, extract_confirmed_needs
from recommendation_guard import (
    render_approved_validation,
    render_unapproved_validation,
    validate_recommendation_data,
)
from response_renderer import (
    render_category_comparison,
    render_fact_answer,
    render_no_supported_option,
    render_policy_answer,
    render_question_plan,
    render_safety_plan,
)


CATALOG_TOOLS = [search_products, get_policy, calculate_quote]
MAX_TOOL_ROUNDS = 4
GROUNDING_HINTS = (
    "商品", "产品", "洁面", "保湿", "精华", "果酸", "神经酰胺", "P101", "P102",
    "P201", "P202", "P203", "P301", "价格", "多少钱", "预算", "偏油", "偏干",
    "敏感", "清爽", "香味", "促销", "赠品", "退换", "授权",
)
RECOMMENDATION_HINTS = (
    "推荐", "建议", "怎么选", "选哪个", "哪个好", "适合我", "搭配", "配一套", "买什么",
)
DIRECT_FACT_HINTS = ("多少钱", "价格", "规格", "容量", "特点", "政策")


def _latest_user_text(messages: list[AnyMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def _turn_requires_grounding(messages: list[AnyMessage]) -> bool:
    text = _latest_user_text(messages)
    return any(hint.lower() in text.lower() for hint in GROUNDING_HINTS)


def _turn_requests_recommendation(messages: list[AnyMessage]) -> bool:
    text = _latest_user_text(messages)
    return any(hint in text for hint in RECOMMENDATION_HINTS)


def _turn_updates_profile(
    messages: list[AnyMessage], customer_needs: dict[str, Any]
) -> bool:
    text = _latest_user_text(messages)
    evidence = customer_needs.get("evidence") or {}
    return any(
        isinstance(quote, str) and quote and quote in text
        for quote in evidence.values()
    )


def _scope_question() -> AIMessage:
    return AIMessage(
        content=(
            "你想先看一件单品或指定品类，还是一起看看完整的基础护理组合？"
            "我会只按你确认的范围继续核对。"
        )
    )


def _current_turn_confirms_category_route(
    messages: list[AnyMessage], customer_needs: dict[str, Any]
) -> bool:
    if customer_needs.get("selection_scope") != "指定品类":
        return False
    if customer_needs.get("desired_category") is None:
        return False
    return _turn_updates_profile(messages, customer_needs)


def _category_route_plan(customer_needs: dict[str, Any]) -> dict[str, Any] | None:
    """兼容旧调用：返回确定性单品或候选比较。"""
    category = customer_needs.get("desired_category")
    if customer_needs.get("selection_scope") != "指定品类" or not category:
        return None
    proposal = proposal_for_category(customer_needs, category)
    if proposal.status == "ready":
        return {"action": "validate", "sku": proposal.items[0]["sku"]}
    products = search_products_data(category=category).get("products") or []
    if proposal.alternatives:
        allowed = set(proposal.alternatives)
        products = [product for product in products if product["sku"] in allowed]
    return {
        "action": "compare",
        "content": render_category_comparison(category, products, customer_needs),
    }


SYSTEM_PROMPT = """
你是澄初个人护理的智能导购。品牌目录、规则工具和确定性校验结果是商品、价格、
组合与政策的唯一事实来源，不得用常识补充品牌没有提供的内容。

search_products 用于查询商品客观资料；get_policy 用于查询服务与安全边界；
calculate_quote 用于报价，涉及金额时不得自行口算；validate_recommendation 用于正式推荐校验。

不作医疗诊断，不得声称治愈、治疗、保证有效、绝对不过敏或马上见效。
资料没有提供促销、赠品、线上同价、退换货或渠道授权时，应说明“需要向门店或系统确认”。
规则未覆盖时说明“品牌资料未覆盖，无法判断，建议人工复核”。不得为了满足预算而替换成
适用条件不符的商品。

正式建议应清楚呈现：已确认需求、推荐建议、为什么这样建议、价格与预算、使用提醒与限制、
待确认信息。不得展示内部状态码、工具名、规则编号或考试资料文件名。
""".strip()


class CareAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    customer_needs: CustomerNeeds
    decision_plan: dict[str, Any] | None
    recommendation_validation: dict[str, Any] | None
    needs_extraction_error: str | None


@tool
def validate_recommendation(
    items: list[dict[str, Any]],
    state: Annotated[CareAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """校验正式候选；需求由会话状态注入，模型只能提交 SKU 与正整数数量。"""
    needs = state.get("customer_needs") or empty_customer_needs()
    result = validate_recommendation_data(needs, items)
    return Command(
        update={
            "recommendation_validation": result,
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    artifact=result,
                    tool_call_id=tool_call_id,
                    name="validate_recommendation",
                )
            ],
        }
    )


TOOLS = [*CATALOG_TOOLS, validate_recommendation]


def _messages_with_system_prompt(
    messages: list[AnyMessage], customer_needs: CustomerNeeds, extraction_error: str | None
) -> list[AnyMessage]:
    context = (
        "当前已确认需求状态（未知字段不得猜测）：\n"
        + json.dumps(customer_needs, ensure_ascii=False, sort_keys=True)
    )
    if extraction_error:
        context += f"\n需求提取警告：{extraction_error}"
    return [SystemMessage(content=SYSTEM_PROMPT), SystemMessage(content=context), *messages]


def _current_turn_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def _facts_for_text(text: str) -> list[dict[str, Any]]:
    skus = list(dict.fromkeys(re.findall(r"P(?:101|102|201|202|203|301)", text.upper())))
    products: list[dict[str, Any]] = []
    for sku in skus:
        result = search_products_data(sku=sku)
        products.extend(result.get("products") or [])
    if not products:
        category_terms = {
            "洁面": ("洁面", "洗面奶"),
            "保湿": ("保湿", "乳液", "面霜"),
            "精华": ("精华",),
            "旅行配件": ("旅行配件", "旅行装", "分装瓶"),
        }
        for category, terms in category_terms.items():
            if any(term in text for term in terms):
                products.extend(search_products_data(category=category).get("products") or [])
                break
    return products


def _question_validation(plan: dict[str, Any], needs: dict[str, Any]) -> dict[str, Any]:
    question = plan.get("question") or {}
    return {
        "status": "awaiting_information",
        "approved": False,
        "confirmed_needs": needs,
        "items": [],
        "matched_rule_ids": [plan["route_id"]] if plan.get("route_id") else [],
        "missing_information": question.get("fields") or [],
        "violations": [],
        "cautions": [],
        "next_action": "等待顾客补充当前问题，不进入商品校验。",
    }


def build_care_agent(checkpointer: Any | None = None):
    base_llm = get_llm()
    needs_extractor = create_needs_extractor(base_llm)
    llm_with_tools = base_llm.bind_tools(TOOLS)

    def extract_needs_node(state: CareAgentState) -> dict[str, Any]:
        current = state.get("customer_needs") or empty_customer_needs()
        text = _latest_user_text(state.get("messages", []))
        if not text:
            return {
                "customer_needs": current,
                "recommendation_validation": None,
                "needs_extraction_error": None,
            }
        try:
            updated = extract_confirmed_needs(needs_extractor, current, text)
            error = None
        except Exception as exc:
            # 模型提取异常时仍使用高置信度规则通道，不让明确事实丢失。
            try:
                fallback_updates = extract_explicit_fact_updates(text)
                for field, update in extract_pending_question_updates(
                    text, current.get("pending_questions") or []
                ).items():
                    fallback_updates.setdefault(field, update)
                updated = merge_confirmed_needs(
                    current, fallback_updates, text
                )
            except Exception:
                updated = current
            error = str(exc)
        return {
            "customer_needs": updated,
            "recommendation_validation": None,
            "needs_extraction_error": error,
        }

    def plan_node(state: CareAgentState) -> dict[str, Any]:
        needs = state.get("customer_needs") or empty_customer_needs()
        text = _latest_user_text(state.get("messages", []))
        plan = plan_next_action(needs, text)
        if plan.get("action") == "ASK":
            fields = (plan.get("question") or {}).get("fields") or []
            needs = set_pending_questions(needs, fields)
            validation = _question_validation(plan, needs)
        elif plan.get("action") == "STOP_FOR_SAFETY":
            needs = set_pending_questions(needs, [])
            validation = {
                "status": "safety_blocked",
                "approved": False,
                "confirmed_needs": needs,
                "items": [],
                "matched_rule_ids": [],
                "missing_information": [],
                "violations": plan.get("reason_codes") or [],
                "cautions": [],
                "professional_consultation_required": True,
                "next_action": "停止相关护肤推荐并说明安全边界。",
            }
        elif plan.get("action") == "NO_SUPPORTED_OPTION":
            needs = set_pending_questions(needs, [])
            validation = {
                "status": "not_supported",
                "approved": False,
                "confirmed_needs": needs,
                "items": [],
                "matched_rule_ids": [plan["route_id"]] if plan.get("route_id") else [],
                "missing_information": [],
                "violations": plan.get("conflicts") or plan.get("reason_codes") or [],
                "cautions": [],
                "next_action": "不放宽已确认条件，改看其他方向或人工确认。",
            }
        else:
            validation = None
        return {
            "customer_needs": needs,
            "decision_plan": plan,
            "recommendation_validation": validation,
        }

    def assistant_node(state: CareAgentState) -> dict[str, Any]:
        plan = state.get("decision_plan") or {"action": "CHAT"}
        action = plan.get("action")
        needs = state.get("customer_needs") or empty_customer_needs()
        messages = state.get("messages", [])
        current_turn = _current_turn_messages(messages)

        if current_turn and isinstance(current_turn[-1], ToolMessage):
            response = base_llm.invoke(
                _messages_with_system_prompt(messages, needs, state.get("needs_extraction_error"))
            )
            return {"messages": [response]}

        if action == "ASK":
            return {"messages": [AIMessage(content=render_question_plan(plan, needs))]}
        if action == "STOP_FOR_SAFETY":
            return {"messages": [AIMessage(content=render_safety_plan(plan))]}
        if action == "NO_SUPPORTED_OPTION":
            return {"messages": [AIMessage(content=render_no_supported_option(plan, needs))]}
        if action == "ANSWER_POLICY":
            topics = plan.get("policy_topics") or [plan.get("policy_topic")]
            answers = [
                render_policy_answer(get_policy_data(topic=topic))
                for topic in topics
                if topic
            ]
            return {"messages": [AIMessage(content="\n\n".join(dict.fromkeys(answers)))]}
        if action == "COMPARE_CATEGORY":
            category = plan.get("category") or needs.get("desired_category")
            result = search_products_data(category=category)
            products = result.get("products") or []
            allowed = {item["sku"] for item in plan.get("candidate_items") or []}
            if allowed:
                products = [product for product in products if product["sku"] in allowed]
            return {
                "messages": [
                    AIMessage(content=render_category_comparison(category, products, needs))
                ]
            }
        if action == "BUILD_CANDIDATE":
            proposal = build_candidate(plan, needs)
            if proposal.status != "ready":
                validation = {
                    "status": "system_error",
                    "approved": False,
                    "confirmed_needs": needs,
                    "items": [],
                    "matched_rule_ids": [],
                    "missing_information": [],
                    "violations": ["候选生成没有得到唯一有效商品。"],
                    "cautions": [],
                    "next_action": "系统重新规划候选。",
                }
                return {
                    "recommendation_validation": validation,
                    "messages": [AIMessage(content=render_unapproved_validation(validation))],
                }
            validation = validate_recommendation_data(needs, proposal.items)
            if proposal.source:
                validation["candidate_selection"] = {
                    "method": "deterministic_plan",
                    "rule_id": proposal.route_id,
                    "selected_items": proposal.items,
                    "source": proposal.source,
                }
            content = (
                render_approved_validation(validation)
                if validation.get("approved") is True
                else render_unapproved_validation(validation)
            )
            return {
                "recommendation_validation": validation,
                "messages": [AIMessage(content=content)],
            }
        if action == "ANSWER_FACT":
            text = _latest_user_text(messages)
            products = _facts_for_text(text)
            rendered = render_fact_answer(text, products)
            if rendered:
                return {"messages": [AIMessage(content=rendered)]}

        response = llm_with_tools.invoke(
            _messages_with_system_prompt(messages, needs, state.get("needs_extraction_error"))
        )
        return {"messages": [response]}

    graph = StateGraph(CareAgentState)
    graph.add_node("extract_needs", extract_needs_node)
    graph.add_node("plan", plan_node)
    graph.add_node("assistant", assistant_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_edge(START, "extract_needs")
    graph.add_edge("extract_needs", "plan")
    graph.add_edge("plan", "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def thread_config(thread_id: str, recursion_limit: int = 12) -> dict[str, Any]:
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("thread_id 必须是非空字符串")
    if not isinstance(recursion_limit, int) or isinstance(recursion_limit, bool) or recursion_limit <= 0:
        raise ValueError("recursion_limit 必须是正整数")
    return {
        "configurable": {"thread_id": thread_id.strip()},
        "recursion_limit": recursion_limit,
    }


__all__ = [
    "CareAgentState",
    "MAX_TOOL_ROUNDS",
    "SYSTEM_PROMPT",
    "TOOLS",
    "build_care_agent",
    "thread_config",
    "validate_recommendation",
]
