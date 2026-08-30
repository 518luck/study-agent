"""
【案例】用「字典」定义 ChatPromptTemplate 的消息

对应教程章节：第 13 章 - 提示词与消息模板 → 7、对话提示词模板（ChatPromptTemplate）

知识点速览：
- 字典写法最贴近 OpenAI 风格的消息结构：`{"role": "...", "content": "..."}`。
- 它特别适合和 JSON 配置、接口透传数据、日志回放等场景对齐。
- 从教学角度看，字典、元组、Message 类三种写法都成立；本文件重点让你看懂“数据结构长什么样”。
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------- 方式一：from_messages ----------
chat_prompt = ChatPromptTemplate.from_messages(
    [
        {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
        {"role": "user", "content": "请问：{question}"},
    ]
)
message = chat_prompt.format_messages(name="小问", question="什么是LangChain")
print("from_messages:", message)

# ---------- 方式二：构造函数（传入同样字典列表，效果一致）----------
chat_prompt2 = ChatPromptTemplate(
    [
        {"role": "system", "content": "你是AI助手，你的名字叫{name}。"},
        {"role": "user", "content": "请问：{question}"},
    ]
)
message2 = chat_prompt2.format_messages(name="小问", question="什么是LangChain")
print("构造函数:", message2)

"""
【输出示例】
[SystemMessage(content='你是AI助手，你的名字叫小问。', additional_kwargs={}, response_metadata={}), HumanMessage(content='请问：什么是LangChain', additional_kwargs={}, response_metadata={})]
"""
