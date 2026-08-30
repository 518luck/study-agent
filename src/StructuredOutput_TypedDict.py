"""
【案例】用 TypedDict + Annotated 做结构化输出

对应教程章节：第 14 章 - 输出解析器 → 3、结构化输出（TypedDict / Pydantic / Annotated）

知识点速览：
一、什么是「结构化输出」？
  - 希望模型返回的不是随便一段话，而是固定结构的数据（如指定字段的 dict），方便程序后续处理。
  - LangChain 支持用 TypedDict（Python 标准库）或 Pydantic 定义结构，再由模型按该结构生成并解析。

二、with_structured_output 做什么？
  - 在模型上调用 llm.with_structured_output(某个类型)，会返回一个「带了结构化输出能力」的可调用对象。
  - 调用 .invoke(messages) 时，模型会被引导输出符合该结构的 JSON，并自动解析成 Python 的 dict（或对应类型），无需再手写 Parser。

三、本案例：用 typing.TypedDict 定义 Animal（动物名 + emoji）、AnimalList（动物列表）；用 Annotated 给字段加描述，便于模型理解各字段含义；最后用 llm.with_structured_output(AnimalList) 一次得到结构化结果。
"""

import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv(encoding="utf-8")

# llm = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 变量名要与 .env 里配置的一致，否则取到 None 会导致调用失败
    base_url="https://api.deepseek.com",
    # 关闭思考模式：deepseek-v4-flash 默认开启思考，而思考模式下不允许 with_structured_output
    # 的 function_calling 方式强制指定 tool_choice，会报 "Thinking mode does not support this tool_choice"
    extra_body={"thinking": {"type": "disabled"}},
    # max_completion_tokens=256,  # 可选：限制单次回复长度（新版 langchain-openai 已把 max_tokens 改名为 max_completion_tokens）
)


# 用 TypedDict 定义「一个动物」的结构；Annotated 里的字符串是给模型看的描述，便于生成合适内容
class Animal(TypedDict):
    animal: Annotated[str, "动物"]
    emoji: Annotated[str, "表情"]


# 定义「动物列表」：一个字段 animals，类型是 Animal 的列表
class AnimalList(TypedDict):
    animals: Annotated[list[Animal], "动物与表情列表"]


# 普通对话消息
messages = [{"role": "user", "content": "任意生成三种动物，以及他们的 emoji 表情"}]

# 给模型绑定「结构化输出」：按 AnimalList 的结构返回并解析
# DeepSeek 不支持 response_format 的 json_schema 类型，需显式指定 function_calling（工具调用方式）
llm_with_structured_output = llm.with_structured_output(
    AnimalList, method="function_calling"
)
resp = llm_with_structured_output.invoke(messages)
print(
    resp
)  # 得到符合 AnimalList 的 dict，如 {"animals": [{"animal": "猫", "emoji": "🐱"}, ...]}

"""
【输出示例】
{'animals': [{'animal': '狗', 'emoji': '🐶'}, {'animal': '猫', 'emoji': '🐱'}, {'animal': '鸟', 'emoji': '🐦'}]}
"""
