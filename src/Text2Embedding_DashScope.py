"""
【案例】LangChain DashScope 封装：单条与批量文本向量化

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4.5 案例：用 LangChain 的统一接口做单条与批量向量化

知识点速览：
- 这是最贴近后续 LangChain 检索器、向量库、RAG 用法的 Embedding 案例，因为它使用的是 LangChain 统一接口。
- embed_query(text)：更偏“查询阶段”，常用于把用户问题转成向量。
- embed_documents(texts)：更偏“索引阶段”，常用于把文档片段批量转成向量。
- 返回值分别是“单个向量”和“向量列表”；向量维度由当前模型决定，建索引和查询时应保持模型一致。

模型文档链接：https://bailian.console.aliyun.com/cn-beijing/?tab=api#/api/?type=model&url=2587654
"""

# pip install langchain-community dashscope
import os

from dotenv import load_dotenv
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()

# 使用项目统一的 aliQwen-api；DashScopeEmbeddings 默认只读 DASHSCOPE_API_KEY，故显式传入
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("aliQwen-api"),
)

text = "This is a test document."

# 单条文本 → 一个向量（列表）；这类写法更贴近“把用户问题转成查询向量”
query_result = embeddings.embed_query(text)
# sep=""：print 多个参数时用空字符串连接，默认是空格；这里让「文本向量长度：」和数字紧挨着输出，中间不留空
print("文本向量长度：", len(query_result), sep="")

# 多条文本 → 多个向量（列表的列表）；这类写法更贴近“批量建索引”
doc_results = embeddings.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!",
    ]
)
print(doc_results)
# sep=""：多个参数之间不加空格，输出如「文本向量数量：5，文本向量长度：1024」
print(
    "文本向量数量：", len(doc_results), "，文本向量长度：", len(doc_results[0]), sep=""
)
