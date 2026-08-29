"""
【案例】invoke / ainvoke 的多种输入类型（字符串、Message 列表、元组列表、字典列表）

对应教程章节：第 13 章 → 2、调用大模型的入参类型

知识点速览：
- 聊天模型的 `invoke` 不只接受字符串，也常接受消息对象列表、`(role, content)` 元组列表、
  `{"role": "...", "content": "..."}` 字典列表。
- 这些写法的目标都是表达“这次输入由哪些角色、哪些内容组成”；LangChain 会在内部转成统一的消息表示。
- 初学者若想把角色关系看得最清楚，优先用 `SystemMessage`、`HumanMessage` 等 Message 类写法。
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# model = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 变量名要与 .env 里配置的一致，否则取到 None 会导致调用失败
    base_url="https://api.deepseek.com",
    temperature=0.7,  # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_completion_tokens=256,  # 可选：限制单次回复长度（新版 langchain-openai 已把 max_tokens 改名为 max_completion_tokens）
)


def demo_message_objects():
    """推荐：显式 Message 对象，角色与字段最清晰。"""
    messages = [
        SystemMessage(content="你是一个专业的数学助手，回答要简短。"),
        HumanMessage(content="你好，你是谁？"),
    ]
    resp = model.invoke(messages)
    print(type(resp), resp.content[:80] if resp.content else "")


def demo_tuple_list():
    """元组列表：与 ChatPromptTemplate.from_messages 的写法一致。"""
    messages = [
        ("system", "你是一个专业的数学助手，回答要简短。"),
        ("human", "你好，你是谁？"),
    ]
    resp = model.invoke(messages)
    print(type(resp), resp.content[:80] if resp.content else "")


def demo_dict_list():
    """字典列表：与 OpenAI Chat Completions 等 API 的请求体形状接近。"""
    messages = [
        {"role": "system", "content": "你是一个专业的数学助手，回答要简短。"},
        {"role": "user", "content": "你好，你是谁？"},
    ]
    resp = model.invoke(messages)
    print(type(resp), resp.content[:80] if resp.content else "")


async def demo_ainvoke_tuple():
    """异步调用同样支持元组简写。"""
    resp = await model.ainvoke([("user", "用一句话说明什么是素数")])
    print(type(resp), resp.content[:80] if resp.content else "")


if __name__ == "__main__":
    print("--- Message 对象列表 ---")
    demo_message_objects()
    print("--- 元组列表 ---")
    demo_tuple_list()
    print("--- 字典列表 ---")
    demo_dict_list()
    print("--- ainvoke + 元组 ---")
    asyncio.run(demo_ainvoke_tuple())

"""
【输出示例】
--- Message 对象列表 ---
<class 'langchain_core.messages.ai.AIMessage'> 我是Qwen，阿里巴巴研发的超大规模语言模型。
--- 元组列表 ---
<class 'langchain_core.messages.ai.AIMessage'> 我是Qwen，阿里巴巴研发的超大规模语言模型。
--- 字典列表 ---
<class 'langchain_core.messages.ai.AIMessage'> 我是Qwen，阿里巴巴研发的超大规模语言模型。
--- ainvoke + 元组 ---
<class 'langchain_core.messages.ai.AIMessage'> 素数（或质数）是指大于1的自然数中，除了1和它本身之外不再有其他正因数的数。
"""
