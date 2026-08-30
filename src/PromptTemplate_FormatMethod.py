"""
【案例】PromptTemplate 的 format() 方法

对应教程章节：第 13 章 - 提示词与消息模板 → 6、文本提示词模板（PromptTemplate）

知识点速览：
- `format(...)` 是 `PromptTemplate` 最常用的方法：填入变量后直接得到一条字符串。
- 如果模板中还有未传入的占位符，`format(...)` 会报错，因此它也能帮助你尽早发现参数遗漏。
- 这类字符串既可以直接发给模型，也可以继续作为后续 Prompt 组合的一部分。
"""

from langchain_core.prompts import PromptTemplate

# ---------- 1. 创建模板（from_template 自动推断 {role}、{question}）----------
template = PromptTemplate.from_template(
    "你是一个专业的{role}工程师，请回答我的问题给出回答，我的问题是：{question}"
)

# ---------- 2. format 填入变量，得到「最终一条提示词字符串」----------
prompt = template.format(role="python开发", question="二分查找算法怎么写？")
print(prompt)
# 类型是 str，可直接传给 model.invoke(prompt)（若模型支持）
print(type(prompt))

"""
【输出示例】
你是一个专业的python开发工程师，请回答我的问题给出回答，我的问题是：二分查找算法怎么写？
<class 'str'>
"""
