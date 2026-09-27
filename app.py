"""澄初个人护理智能导购的 Streamlit 用户界面。"""

from __future__ import annotations

import copy
import html
import logging
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from care_agent import build_care_agent, thread_config
from ui_helpers import (
    allowed_conclusion,
    friendly_tool_trace,
    latest_ai_content,
    needs_details,
    needs_summary,
    public_text,
    quote_rows,
    source_label,
    validation_checks,
    validation_meta,
)


load_dotenv()
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="澄初个人护理智能导购",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --care-green: #486957;
        --care-green-soft: #edf3ee;
        --care-ink: #26322b;
        --care-muted: #6d786f;
        --care-line: #dfe7e1;
        --care-warm: #fbfaf6;
    }
    .block-container {
        max-width: 900px;
        padding-top: 2.2rem;
        padding-bottom: 7rem;
    }
    [data-testid="stSidebar"] {
        border-right: 1px solid var(--care-line);
    }
    .care-brand {
        color: var(--care-green);
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .16em;
        margin-bottom: .45rem;
    }
    .care-hero h1 {
        color: var(--care-ink);
        font-size: 2.05rem;
        line-height: 1.2;
        margin: 0 0 .6rem 0;
    }
    .care-hero p {
        color: var(--care-muted);
        font-size: 1rem;
        line-height: 1.75;
        max-width: 620px;
        margin: 0 0 1.35rem 0;
    }
    .care-welcome {
        background: linear-gradient(145deg, #f7faf7 0%, #fffdf8 100%);
        border: 1px solid var(--care-line);
        border-radius: 20px;
        padding: 1.5rem 1.55rem .9rem 1.55rem;
        margin: .75rem 0 1.1rem 0;
    }
    .care-status {
        border-radius: 12px;
        padding: .72rem .85rem;
        margin: .1rem 0 .8rem 0;
        border: 1px solid var(--care-line);
    }
    .care-status strong { display: block; margin-bottom: .12rem; }
    .care-status span { color: #566259; font-size: .88rem; }
    .care-success { background: #edf7f0; border-color: #cfe4d4; }
    .care-warning { background: #fff7e8; border-color: #ecd8aa; }
    .care-info { background: #eef5f8; border-color: #cfdee5; }
    .care-danger { background: #fff0ef; border-color: #eccdca; }
    .care-neutral { background: #f4f5f3; border-color: #dfe2dc; }
    .care-context {
        color: var(--care-muted);
        font-size: .86rem;
        border-left: 3px solid #b8cabd;
        padding-left: .7rem;
        margin: .75rem 0 .25rem 0;
    }
    .care-side-note {
        color: var(--care-muted);
        font-size: .86rem;
        line-height: 1.65;
        margin: .35rem 0 .8rem 0;
    }
    div[data-testid="stChatMessage"] {
        border-bottom: 1px solid #eef1ee;
        padding-bottom: .65rem;
    }
    div[data-testid="stExpander"] {
        border-color: var(--care-line);
        border-radius: 12px;
    }
    .stButton > button {
        border-radius: 12px;
        border-color: #d8e2da;
        min-height: 2.7rem;
    }
    .stButton > button:hover {
        border-color: var(--care-green);
        color: var(--care-green);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


EXAMPLES = [
    (
        "偏干敏感，预算 400 元",
        "请帮我搭配一套简单日常护理。我偏干、有敏感倾向，预算400元。",
    ),
    (
        "偏油清爽，可以接受香味",
        "请帮我推荐一套基础护理。我偏油，喜欢清爽肤感，可以接受香味，预算400元。",
    ),
    (
        "明显不适，想咨询果酸",
        "我现在皮肤有明显不适，还能使用果酸精华吗？",
    ),
    (
        "购买是否有赠品",
        "现在购买澄初产品有赠品吗？",
    ),
]


@st.cache_resource
def get_agent():
    # 编译后的图必须缓存，否则 Streamlit 每次刷新都会丢失内存会话。
    return build_care_agent()


def initialize_session() -> None:
    if "care_thread_id" not in st.session_state:
        st.session_state.care_thread_id = str(uuid.uuid4())
    if "care_history" not in st.session_state:
        st.session_state.care_history = []


def reset_conversation() -> None:
    st.session_state.care_thread_id = str(uuid.uuid4())
    st.session_state.care_history = []


def render_header() -> None:
    st.markdown(
        """
        <div class="care-hero">
          <div class="care-brand">CHENGCHU · PERSONAL CARE</div>
          <h1>澄初个人护理智能导购</h1>
          <p>说说你的肤质、护理偏好和预算。我会结合品牌资料，核对商品适用条件、使用边界和价格。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(validation: dict[str, Any] | None) -> None:
    meta = validation_meta(validation)
    if not meta:
        return
    title = html.escape(meta["title"])
    description = html.escape(meta["description"])
    tone = html.escape(meta["tone"])
    st.markdown(
        f'<div class="care-status care-{tone}"><strong>{title}</strong>'
        f'<span>{description}</span></div>',
        unsafe_allow_html=True,
    )


def render_evidence(entry: dict[str, Any]) -> None:
    validation = entry.get("validation")
    needs = entry.get("needs")
    details = needs_details(needs)
    rows = quote_rows(validation)
    if not details and not isinstance(validation, dict):
        return

    with st.expander("查看本次判断依据"):
        if details:
            st.markdown("**1. 已确认的用户情况**")
            for row in details:
                evidence = (
                    f"  · 原话：\“{row['evidence']}\”" if row.get("evidence") else ""
                )
                st.markdown(f"- **{row['label']}：** {row['value']}{evidence}")
            st.caption("这里只使用用户明确表达过的信息。")

        if isinstance(validation, dict):
            rule_ids = validation.get("matched_rule_ids") or []
            if rows or rule_ids:
                st.markdown("**2. 对应的品牌资料**")
                if rows:
                    for row in rows:
                        st.markdown(f"- {row['name']}（{row['sku']}）")
                if rule_ids:
                    st.caption(f"依据规则：{'、'.join(public_text(rule_id) for rule_id in rule_ids)}")

                selection = validation.get("candidate_selection")
                if isinstance(selection, dict) and selection.get("source"):
                    st.caption(f"资料位置：{source_label(selection['source'])}")

            if rows:
                st.markdown("**3. 价格核对**")
                table = ["| 商品 | 单价 | 数量 | 小计 |", "|---|---:|---:|---:|"]
                for row in rows:
                    table.append(
                        f"| {row['name']}（{row['sku']}） | {row['unit_price_yuan']} 元 | "
                        f"{row['quantity']} | {row['subtotal_yuan']} 元 |"
                    )
                st.markdown("\n".join(table))
                quote = validation.get("quote") or {}
                summary = [f"总价 **{quote.get('total_yuan')} 元**"]
                if quote.get("budget_yuan") is not None:
                    summary.append(f"预算 **{quote.get('budget_yuan')} 元**")
                if quote.get("within_budget") is True:
                    summary.append(f"剩余 **{quote.get('remaining_budget_yuan')} 元**")
                elif quote.get("within_budget") is False:
                    summary.append(f"超出 **{quote.get('over_budget_yuan')} 元**")
                st.markdown("；".join(summary))

            checks = validation_checks(validation)
            if checks:
                st.markdown("**4. 结论核对**")
                icons = {"pass": "✓", "warning": "!", "stop": "×", "info": "•"}
                for check in checks:
                    st.markdown(f"{icons.get(check['level'], '•')} {check['text']}")

            st.info(f"**允许输出的结论：** {allowed_conclusion(validation)}")


def render_trace(entry: dict[str, Any]) -> None:
    steps = entry.get("trace") or []
    validation = entry.get("validation")
    if not steps and not isinstance(validation, dict):
        return
    with st.expander("技术详情（演示）"):
        if steps:
            st.markdown("**本次已完成**")
            for step in steps:
                st.markdown(f"✓ {public_text(step)}")
        if isinstance(validation, dict):
            rule_ids = validation.get("matched_rule_ids") or []
            meta = validation_meta(validation)
            if rule_ids:
                st.caption(f"规则编号：{'、'.join(rule_ids)}")
            if meta:
                status = validation.get("status")
                if status == "awaiting_information":
                    recommendation = "尚未进入"
                elif status == "system_error":
                    recommendation = "系统待重试"
                else:
                    recommendation = "通过" if validation.get("approved") else "未通过"
                st.caption(f"系统结果：{meta['title']} · 正式推荐检查：{recommendation}")
        st.caption("此区域用于演示 Agent 的业务执行记录，不展示模型内部推理过程。")


def render_history() -> None:
    for entry in st.session_state.care_history:
        role = entry.get("role")
        avatar = "🌿" if role == "assistant" else None
        with st.chat_message(role, avatar=avatar):
            if role == "assistant":
                render_status(entry.get("validation"))
            st.markdown(public_text(entry.get("content", "")))
            if role == "assistant" and entry.get("validation"):
                summary = needs_summary(entry.get("needs"))
                st.markdown(
                    f'<div class="care-context"><strong>当前情况：</strong>'
                    f'{html.escape(summary)}</div>',
                    unsafe_allow_html=True,
                )
            if role == "assistant":
                render_evidence(entry)
                render_trace(entry)


def run_agent(prompt: str) -> dict[str, Any]:
    result = get_agent().invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=thread_config(st.session_state.care_thread_id),
    )
    return {
        "role": "assistant",
        "content": latest_ai_content(result.get("messages")),
        "needs": copy.deepcopy(result.get("customer_needs")),
        "validation": copy.deepcopy(result.get("recommendation_validation")),
        "trace": friendly_tool_trace(result.get("messages")),
        "needs_extraction_warning": bool(result.get("needs_extraction_error")),
    }


def render_welcome_examples() -> str | None:
    st.markdown(
        """
        <div class="care-welcome">
          <strong>可以从这些问题开始</strong>
          <p style="color:#6d786f;margin:.3rem 0 .7rem 0;font-size:.9rem;">
            不必一次提供全部信息，我会在需要时继续询问。
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected = None
    for row_start in range(0, len(EXAMPLES), 2):
        columns = st.columns(2)
        for offset, column in enumerate(columns):
            index = row_start + offset
            if index >= len(EXAMPLES):
                continue
            label, prompt = EXAMPLES[index]
            with column:
                if st.button(label, key=f"welcome_example_{index}", use_container_width=True):
                    selected = prompt
    return selected


initialize_session()
render_header()

selected_example = None
with st.sidebar:
    st.markdown("### 澄初")
    if st.button("＋ 开始新咨询", use_container_width=True, type="primary"):
        reset_conversation()
        st.rerun()

    st.markdown(
        '<div class="care-side-note">可以告诉我你的肤质、敏感或不适情况、护理偏好和预算，不必一次说完。</div>',
        unsafe_allow_html=True,
    )

    with st.expander("示例问题"):
        for index, (label, prompt) in enumerate(EXAMPLES):
            if st.button(label, key=f"sidebar_example_{index}", use_container_width=True):
                selected_example = prompt

    with st.expander("关于本助手"):
        st.write(
            "建议依据澄初品牌商品资料和服务规则生成。资料不足时会继续询问，"
            "或提示需要门店、系统或人工确认。本助手不作医疗诊断。"
        )

    st.caption("商品信息与价格以品牌资料和实际门店系统为准。")

if not st.session_state.care_history:
    selected_example = render_welcome_examples() or selected_example

render_history()

typed_prompt = st.chat_input("例如：我偏干、有敏感倾向，预算400元……")
prompt = typed_prompt or selected_example

if prompt:
    prompt = prompt.strip()
    if prompt:
        st.session_state.care_history.append({"role": "user", "content": prompt})
        try:
            with st.spinner("正在核对商品资料、适用规则和预算……"):
                response = run_agent(prompt)
        except Exception:
            LOGGER.exception("Care agent request failed")
            response = {
                "role": "assistant",
                "content": (
                    "暂时无法连接导购服务，请稍后重试。"
                    "如果问题持续出现，请检查模型配置或联系维护人员。"
                ),
                "needs": None,
                "validation": None,
                "trace": [],
                "error": True,
            }
        st.session_state.care_history.append(response)
        st.rerun()
