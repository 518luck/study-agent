"""
【案例】函数链：用 RunnableLambda 将普通 Python 函数接入 LCEL 链

对应教程章节：第 15 章 - LCEL 与链式调用 → 4.5 RunnableLambda（函数链）

知识点速览：
- `RunnableLambda` 的作用，是把普通 Python 函数变成 Runnable 节点，方便插入到 LCEL 链中。
- 它特别适合做轻量逻辑，例如打印中间结果、字段映射、输入输出结构适配；如果逻辑已经很重，就不建议继续塞在函数链里。
- 除了显式写 `RunnableLambda(函数)`，也可以直接把函数放在 `|` 之间，LangChain 会自动完成包装。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from loguru import logger

load_dotenv(encoding="utf-8")

model = init_chat_model(
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


def debug_print(x):
    """打印中间结果，并把文本包装成 chain2 所需的 {"input": 文本} 结构。"""
    logger.info(f"中间结果:{x}")
    return {"input": x}


# 子链 1：中文介绍某主题，输出 str
prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个知识渊博的计算机专家，请用中文简短回答"),
        ("human", "请简短介绍什么是{topic}"),
    ]
)
parser1 = StrOutputParser()
chain1 = prompt1 | model | parser1

# 子链 2：将 input 翻译成英文
prompt2 = ChatPromptTemplate.from_messages(
    [("system", "你是一个翻译助手，将用户输入内容翻译成英文"), ("human", "{input}")]
)
parser2 = StrOutputParser()
chain2 = prompt2 | model | parser2

# 方式一：直接把函数放在 | 之间，LCEL 会自动包装成 Runnable
full_chain = chain1 | debug_print | chain2
result1 = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果111:{result1}")

# 方式二：显式使用 RunnableLambda(函数)，效果相同
debug_node = RunnableLambda(debug_print)
full_chain = chain1 | debug_node | chain2
result2 = full_chain.invoke({"topic": "langchain"})
logger.info(f"最终结果222:{result2}")

"""
【输出示例】
2026-03-06 10:30:31.657 | INFO     | __main__:debug_print:34 - 中间结果:LangChain 是一个开源框架，用于构建基于大语言模型（LLM）的应用程序。它提供模块化组件（如链、代理、记忆、工具集成等），帮助开发者轻松连接 LLM 与外部数据源（如数据库、API）、实现对话状态管理、支持多步推理与工具调用，从而快速开发智能应用（如问答系统、聊天机器人、自动化工作流）。核心理念是“编排”——将 LLM 作为通用接口，协同其他能力完成复杂任务。
2026-03-06 10:30:34.964 | INFO     | __main__:<module>:57 - 最终结果111:LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides modular components—such as chains, agents, memory, and tool integrations—that enable developers to easily connect LLMs with external data sources (e.g., databases, APIs), manage conversational state, support multi-step reasoning and tool invocation, and rapidly develop intelligent applications—including question-answering systems, chatbots, and automated workflows. Its core philosophy centers on “orchestration”: treating the LLM as a universal interface that coordinates with other capabilities to accomplish complex tasks.
2026-03-06 10:30:38.152 | INFO     | __main__:debug_print:34 - 中间结果:LangChain 是一个开源框架，用于构建基于大语言模型（LLM）的应用程序。它提供模块化组件（如链、代理、记忆、工具集成等），帮助开发者轻松连接 LLM 与外部数据源（如数据库、API）、实现对话状态管理、支持多步推理与工具调用，从而快速开发智能应用（如问答系统、聊天机器人、自动化工作流）。核心理念是“编排”——将 LLM 作为通用接口，协同其他能力完成复杂任务。
2026-03-06 10:30:41.184 | INFO     | __main__:<module>:63 - 最终结果222:LangChain is an open-source framework for building applications powered by large language models (LLMs). It provides modular components—such as chains, agents, memory, and tool integrations—that enable developers to easily connect LLMs with external data sources (e.g., databases, APIs), manage conversational state, support multi-step reasoning and tool invocation, and rapidly develop intelligent applications—including question-answering systems, chatbots, and automated workflows. Its core philosophy centers on “orchestration”: treating the LLM as a universal interface that coordinates with other capabilities to accomplish complex tasks.
"""
