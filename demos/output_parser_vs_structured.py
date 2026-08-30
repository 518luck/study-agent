"""
【Demo】输出解析器 vs 结构化输出 —— 三种使用方式实测对比
对应教程：第 14 章 → 1.5 输出解析器与结构化输出的关系

任务：从一段自我介绍文本里提取人物信息（name / age / skills），
对比三种"让模型输出变成程序可用结构"的方式。

运行（study-agent 目录下）：
    .venv/bin/python demos/output_parser_vs_structured.py
"""

import os

from dotenv import load_dotenv

load_dotenv()  # 读取 study-agent/.env

from langchain.chat_models import init_chat_model

# ========== 公共部分：根据手头的 key 自动选模型 ==========
if os.getenv("aliQwen-api"):
    model = init_chat_model(
        model="qwen-plus",
        model_provider="openai",
        api_key=os.getenv("aliQwen-api"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    provider = "qwen-plus"
elif os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek-api"):
    key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek-api")
    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=key,
    )
    provider = "deepseek-chat"
else:
    raise SystemExit("请先在 .env 里配置 aliQwen-api 或 DEEPSEEK_API_KEY")

RAW = "亮仔，今年 28 岁，是一名后端程序员，擅长 Python 和 LangChain，住在杭州。"
print(f"使用模型：{provider}")
print(f"原始文本：{RAW}")
print("=" * 70)


# ========== 方式一：只用输出解析器（后处理） ==========
# 思路：prompt 里注入"格式指令"约束输出 → 模型回文本 → parser 解析成 dict
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

parser = JsonOutputParser()

prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "从文本中提取人物信息。{format_instructions}"),
        ("human", "{text}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
#                     ↑ 解析器自动生成的"请按如下 JSON 格式输出"指令，拼进 prompt

chain1 = prompt1 | model | parser  # parser 就是管道最后一段

d1 = chain1.invoke({"text": RAW})

print("【方式一】只用输出解析器：prompt | model | JsonOutputParser")
print("返回类型：", type(d1).__name__)
print("内容：", d1)
print("取值：d1['name'] =", d1.get("name"), "  d1['age'] =", d1.get("age"))
print("⚠️ 它只是普通 dict：没有字段校验，模型少给/类型给错不会报错")
print("=" * 70)


# ========== 方式二：只用结构化输出（前约束 + 自动解析，无手写 parser） ==========
# 思路：with_structured_output 内部用厂商的原生能力(工具调用/JSON模式)
#       一并完成"约束输出结构"和"解析"，不再需要单独的 parser
from typing import TypedDict


class Person(TypedDict):
    name: str
    age: int
    skills: list[str]


structured_model = model.with_structured_output(Person)
r2 = structured_model.invoke(f"提取人物信息：{RAW}")

print("【方式二】with_structured_output(TypedDict)")
print("返回类型：", type(r2).__name__)
print("内容：", r2)
print("取值：r2['name'] =", r2["name"])
print("ℹ️ TypedDict 版返回 dict；想拿带方法的对象请用 Pydantic（见方式三）")
print("=" * 70)


# ========== 方式三：结构化输出 + Pydantic 校验（工程姿势） ==========
# 思路：结构化输出的同时，用 Pydantic 声明字段类型/范围，拿到带类型的对象，
#       再补业务规则校验——这是真实项目入库前的标准动作
from pydantic import BaseModel, Field


class PersonStrict(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(ge=0, le=150, description="年龄")
    skills: list[str] = Field(default_factory=list, description="技能列表")


structured_model3 = model.with_structured_output(PersonStrict)
r3 = structured_model3.invoke(f"提取人物信息：{RAW}")

print("【方式三】with_structured_output(Pydantic) + 业务校验")
print("返回类型：", type(r3).__name__)
print("内容：", r3)
print("点语法取值：r3.name =", r3.name, "| r3.age =", r3.age)

# 业务规则校验（相当于教程说的"入库前清洗"）
if 18 <= r3.age <= 60:
    print("业务校验：年龄符合招聘要求(18~60)，允许入库 ✓")
else:
    print("业务校验：年龄不符，拒绝入库 ✗")

print("=" * 70)
print(
    "总结：三种方式目标一致——让输出进入程序系统。\n"
    "  方式一：后处理，parser 在模型输出后解析（灵活，但无类型保证）\n"
    "  方式二：前约束+自动解析，一步到位（现代模型首选）\n"
    "  方式三：在二之上补类型与业务校验（生产推荐）"
)
