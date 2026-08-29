"""
【案例】使用 OpenAI 官方 SDK 直接调用大模型（不经过 LangChain）

对应教程章节：第 11 章 - Model I/O 与模型接入 → 3、接入大模型

知识点速览：
- 本案例故意不使用 LangChain，目的是帮助你和 `ChatOpenAI`、`init_chat_model` 做边界对比。
- 适合「仅需简单 API 调用」、暂时不打算接 Prompt、Parser、LCEL、Agent 等 LangChain 组件的场景。
- 返回结果是 OpenAI SDK 的原生响应结构，不是 LangChain 的 `AIMessage`；读取正文需要走 `response.choices[0].message.content`。
- 依赖 `openai` 包，运行前在 `.env` 中配置 DeepSeek 的 API Key（变量名 `DEEPSEEK_API_KEY`）。
"""

# ========== 1. 导入与环境 ==========
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(encoding="utf-8")

# ========== 2. 初始化客户端（底层 API，直接请求厂商接口） ==========
# 这里以 DeepSeek 官方兼容接口为例；若你切到别的 OpenAI 兼容平台，通常只需调整 base_url、api_key、model。
client = OpenAI(
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 从环境变量读取（.env 里的 DEEPSEEK_API_KEY）
    base_url="https://api.deepseek.com",  # 可改为其他 OpenAI 兼容地址（如阿里百炼）
)

# ========== 3. 发起对话并打印回复 ==========
# 注意：这里的 messages 是 OpenAI SDK 语义下的消息列表，和 LangChain 中常见的“消息列表”长得相似，
# 但最终返回值结构并不一样。
response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello，你是谁？"},
    ],
    stream=False,
)

# OpenAI SDK 的返回值需要按原生结构逐层取值：
# response -> choices[0] -> message -> content
print(response.choices[0].message.content)
