"""
【案例】PromptTemplate 的 invoke() 方法

对应教程章节：第 13 章 - 提示词与消息模板 → 6、文本提示词模板（PromptTemplate）

知识点速览：
- `invoke(...)` 返回的不是普通字符串，而是 `PromptValue`，更适合衔接 LangChain 的链式调用。
- 对 `PromptTemplate` 来说，返回值通常是 `StringPromptValue`；你可以继续 `.to_string()`，也可以 `.to_messages()`。
- 其中 `.to_messages()` 常见会转成单条 `HumanMessage`，它是把文本 Prompt 包装成消息的一种统一方式。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建模板 ----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. 用 invoke 传入变量（字典），得到 PromptValue 对象 ----------
prompt = template.invoke({"role": "python开发", "question": "冒泡排序怎么写？"})
print(prompt)
print(type(prompt))
print()

# ---------- 3. 从 PromptValue 取内容：to_string() 得到整段字符串 ----------
print(prompt.to_string())
print(type(prompt.to_string()))
print()

# ---------- 4. to_messages()：转成「消息列表」，可接入需要多角色消息的链 ----------
print(prompt.to_messages())
print(type(prompt.to_messages()))


"""
【输出示例】
text='你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写？'
<class 'langchain_core.prompt_values.StringPromptValue'>

你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写？
<class 'str'>

[HumanMessage(content='你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：冒泡排序怎么写？', additional_kwargs={}, response_metadata={})]
<class 'list'>
"""
