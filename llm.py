"""通过阿里云百炼 DashScope 的 OpenAI 兼容接口调用 Qwen。

ChatOpenAI 是 LangChain 客户端名称；实际服务商由 base_url 指定，
本项目默认请求阿里云，不使用 OpenAI 官方服务、Groq 或本地 Ollama。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# 明确读取本项目旁的 .env，避免从其他工作目录启动时读错配置。
# 已有系统环境变量优先，便于以后在部署平台中配置密钥。
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def get_llm() -> ChatOpenAI:
    """创建 Qwen 客户端；真正的网络请求发生在 invoke() 时。"""
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请参考 .env.example 配置本项目的 .env。")

    # 这是接口根地址，客户端会调用其下的 /chat/completions。
    # Key、接口区域和模型访问权限须匹配；不要把 Key 填入 GROQ_API_KEY。
    base_url = os.getenv(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ).strip().rstrip("/")
    model = os.getenv("QWEN_CHAT_MODEL", "qwen3.6-flash").strip()
    if not base_url or not model:
        raise ValueError("DASHSCOPE_BASE_URL 和 QWEN_CHAT_MODEL 不能为空。")

    return ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=0.1,  # 较低随机性，减少演示时回答波动；不保证完全确定。
        timeout=60,      # 单次 HTTP 请求超时（秒）。
        max_retries=1,   # 暂时性错误最多重试一次。
        # 使用非思考模式降低演示延迟；更换模型时需确认该参数兼容。
        extra_body={"enable_thinking": False},
    )
