"""
【案例】将 Document 列表向量化并写入 Redis（LangChain 统一接口 + 硅基流动）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 6.1 案例：把 Document 列表写入 Redis，再用检索器取回结果

知识点速览：
- 本文件是教程 6.1 案例的改造版：把「DashScopeEmbeddings（langchain-community）」换成
  「init_embeddings + 硅基流动」。Embedding 模型换成 BAAI/bge-m3（1024 维）；
  langchain-community 已被官方 sunset，新项目建议用独立集成包或本例这种统一初始化入口。
- 这是本章最贴近“向量库实战入口”的案例，演示的是：先准备 Document，再向量化，再写入 Redis，最后按相似度检索。
- Redis.from_documents() 会自动读取每个 Document 的 page_content，调用 embedding 做向量化，并把原文、向量、metadata 一起写入 Redis。
- as_retriever() 得到的是检索器；invoke(查询文本) 时，LangChain 会先把查询文本转成向量，再去库里找最相关的 Document。
- 这个案例是 RAG 的底层能力演示，不包含文档加载器、文本分割器和“检索后交给大模型生成答案”的完整流程。
- 环境要求（两个硬前提）：
  1）Redis 必须带 RediSearch 模块（Redis Stack）。普通 redis 镜像没有该模块，写入会直接报错；
     本项目 compose.yaml 里配好了 redis-stack-server 服务（见 compose 注释），docker compose up -d 即可。
  2）索引名、embedding 模型要与查询端一致——尤其换过 embedding 模型后，旧索引里的向量已不可比，要换新索引名重建。
- 依赖版本兼容：langchain_community 的 Redis 向量库写死了 redis 7.x 的旧模块路径，redis 8.1 改名后会误报
  “Could not import redis python package”，文件里用一个模块别名垫片解决（详见下方代码注释）。
- 本脚本可重复运行：每次运行会先删掉同名索引再重建。因为 from_documents 不查重，直接重跑会往同一索引里
  再写一遍，检索结果就出现同一句话的多份副本（实测跑两遍后文档数 3→6，k=2 返回的是两条相同内容）。
"""

import os
import sys

# 兼容垫片（shim）：langchain_community 的 Redis 向量库还在用 redis 7.x 的旧模块路径
# redis.commands.search.indexDefinition，而项目装的 redis 8.1 已把它改名成下划线风格
# index_definition（redisvl 用的就是新路径）。这里把新模块挂到旧路径上，让旧代码继续可用。
# 若将来 langchain-community 修复了该导入，或者你改用 langchain_redis.RedisVectorStore，可删掉这两行。
import redis
import redis.commands.search.index_definition as _redis_index_definition
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_community.vectorstores import Redis
from langchain_core.documents import Document

sys.modules.setdefault("redis.commands.search.indexDefinition", _redis_index_definition)

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

# 提前拦一道，避免拿着空 key 去请求，只看到一句含义不明的 401
if not api_key:
    raise SystemExit(
        "❌ 未读取到 SILICONFLOW_API_KEY，请在项目根目录的 .env 中补充：\n"
        "   SILICONFLOW_API_KEY=你的密钥\n"
        "   申请地址：https://cloud.siliconflow.cn/me/account/ak"
    )

# 1. 初始化嵌入模型：硅基流动走 OpenAI 兼容协议，provider 仍填 openai
embeddings = init_embeddings(
    "BAAI/bge-m3",
    provider="openai",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    # 直接发原始文本；不设 False 会把文本转成 token 数组，平台用错分词器解读，向量静默出错（实测余弦仅 0.33）
    check_embedding_ctx_length=False,
)

# 2. 构造 Document 列表：page_content 是正文，metadata 是附加信息
# 在完整 RAG 中，这些 Document 往往来自“加载器 + 分割器”；本案例先用手写数据聚焦理解向量库存取流程
texts = [
    "通义千问是阿里巴巴研发的大语言模型。",
    "Redis 是一个高性能的键值存储系统，支持向量检索。",
    "LangChain 可以轻松集成各种大模型和向量数据库。",
]

documents = [
    Document(page_content=text, metadata={"source": "manual"}) for text in texts
]

# 3. 一次性写入 Redis：内部会对每个 Document 的 page_content 做向量化，并建立可检索索引
# 索引名用 siliconflow_demo_index——教程示例的 my_index11 若已用旧模型建过，向量空间不同，不能混用
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:26379")
INDEX_NAME = "siliconflow_demo_index"

# 先删掉同名索引（连文档一起删）再写入：from_documents 不查重，直接重跑会往同一个索引里再写一遍，
# 检索结果就会出现同一句话的多份副本（实测跑两遍后文档数 3→6，k=2 拿到两条一模一样的内容）。
# 这里每次重建，保证脚本可以反复运行；生产环境应改成按唯一键 upsert，而不是无脑重写。
_client = redis.Redis.from_url(REDIS_URL)
try:
    # DD = 连文档一起删；索引不存在时会抛 ResponseError("Unknown Index name")，首次运行时走 except
    _client.execute_command("FT.DROPINDEX", INDEX_NAME, "DD")
except redis.ResponseError:
    pass
finally:
    _client.close()

vector_store = Redis.from_documents(
    documents=documents,
    embedding=embeddings,
    redis_url=REDIS_URL,
    index_name=INDEX_NAME,
)

# 4. 得到检索器：当你 invoke 查询文本时，LangChain 会先把问题向量化，再在库中做相似度检索
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
results = retriever.invoke("LangChain 和 Redis 怎么结合？")
print("检索结果（k=3，按相似度从高到低）：")
print("results:", results)
for res in results:
    print(f"  - {res.page_content}    metadata={res.metadata}")

"""
【输出示例】以下为实跑的真实输出（id 是写库时生成的 Redis 键，每次重建都会变）

检索结果（k=3，按相似度从高到低）：
  - Redis 是一个高性能的键值存储系统，支持向量检索。    metadata={'id': 'doc:siliconflow_demo_index:9128a4998c76…', 'source': 'manual'}
  - LangChain 可以轻松集成各种大模型和向量数据库。    metadata={'id': 'doc:siliconflow_demo_index:de0691814cb8…', 'source': 'manual'}
  - 通义千问是阿里巴巴研发的大语言模型。    metadata={'id': 'doc:siliconflow_demo_index:7bdd80a6fcdd…', 'source': 'manual'}
"""
