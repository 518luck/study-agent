"""
【案例】在 Redis 向量库中做相似性检索（similarity_search_with_score）—— 硅基流动版

对应教程章节：第 19 章 - RAG 检索增强生成 → 2.1.3 再往后一步：检索案例和它们是什么关系；也可与第 18 章相似检索案例对照阅读

知识点速览：
- 这个案例对应的是 RAG 的检索阶段：前提是索引已经建好，现在要做的是“把相关内容查出来”。
- 相似性检索的核心流程是：查询文本先向量化，再到向量库中找到与查询向量最接近的若干条记录。
- `similarity_search_with_score(query, k)` 返回 `(Document, score)` 列表；很多实现里 score 更接近“距离”，通常越小越相似。
- 代码里把 score 换算成 1 - score，主要是为了更符合初学者直觉；真实项目里应以具体向量库和距离度量定义为准。
- 运行前需确保 Redis 中已有数据，例如先执行同目录下的 RedisVectorStore.py；`index_name`、`redis_url` 也必须保持一致。
- 在完整 RAG 里，这一步通常不会直接把结果打印完就结束，而是会把查到的 `Document` 进一步组织进 Prompt，再交给 LLM 生成答案。
- 本文件是教程案例的硅基流动改造版：把「langchain_community 的 DashScopeEmbeddings」换成
  「init_embeddings + 硅基流动 bge-m3」。注意检索端的 embedding 必须与写入端一致：
  不同模型产出的向量不在同一空间，换模型就得换索引名重建（见文件末尾注释）。
"""

import os

from dotenv import load_dotenv
from langchain.embeddings import init_embeddings
from langchain_redis import RedisConfig, RedisVectorStore

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

# 提前拦一道，避免拿着空 key 去请求，只看到一句含义不明的 401
if not api_key:
    raise SystemExit(
        "❌ 未读取到 SILICONFLOW_API_KEY，请在项目根目录的 .env 中补充：\n"
        "   SILICONFLOW_API_KEY=你的密钥\n"
        "   申请地址：https://cloud.siliconflow.cn/me/account/ak"
    )

# 1. 嵌入模型（必须与写入时一致，保证向量空间一致）
embeddingsModel = init_embeddings(
    "BAAI/bge-m3",
    provider="openai",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    # 直接发原始文本；不设 False 会把文本转成 token 数组，平台用错分词器解读，向量静默出错（实测余弦仅 0.33）
    check_embedding_ctx_length=False,
)

# 2. 连接已有索引（index_name、redis_url 必须与 RedisVectorStore.py 一致）
vector_store = RedisVectorStore(
    embeddingsModel,
    config=RedisConfig(
        index_name="newsgroups",
        redis_url=os.getenv("REDIS_URL", "redis://localhost:26379"),
    ),
)

# 3. 查询文本 → 向量化 → 在库中做相似度检索；这里取前 3 条结果
query = "我喜欢用什么手机"
results = vector_store.similarity_search_with_score(query, k=3)

print("=== 查询结果 ===")
for i, (doc, score) in enumerate(results, 1):
    # 这里把“距离”近似换算成“相似度”只是为了展示更直观；工程里请以具体返回定义为准
    # 本索引建的是 COSINE 距离，score = 1 - 余弦相似度，所以 1 - score 就是余弦相似度本身
    similarity = 1 - score
    print(f"结果 {i}:")
    print(f"内容: {doc.page_content}")
    print(f"元数据: {doc.metadata}")
    print(f"相似度: {similarity:.4f}")

"""
【输出示例】以下为实跑的真实输出

=== 查询结果 ===
结果 1:
内容: 我喜欢用苹果手机
元数据: {'segment_id': '3'}
相似度: 0.8673
结果 2:
内容: 我喜欢用苹果手机
元数据: {'segment_id': '3'}
相似度: 0.8673
结果 3:
内容: 我喜欢吃苹果
元数据: {'segment_id': '1'}
相似度: 0.6626
"""

# ============================================================================
# 【实操补充】三个与检索端有关的点（均实测）
#
# 1）结果里出现两条一模一样的文本，不是检索出错：写入端 RedisVectorStore.py 会写两批数据
#    （3 条自动 ULID + 3 条固定 key），所以每句话在库里有 2 份副本，k 大于 3 时就会看到重复。
#    想看干净结果可以先清索引再只跑一次写入端：
#      docker compose exec redis-stack redis-cli FT.DROPINDEX newsgroups DD
#
# 2）score 的含义由索引的距离度量决定，必须以实际配置为准：本索引是 COSINE，
#    RediSearch 的 COSINE 距离 = 1 - 余弦相似度，所以 score 越小越相似，1 - score 就是余弦相似度。
#    换成 L2 等其他度量后，这个换算关系就不成立了。
#
# 3）检索端和写入端必须是同一个 embedding 模型 + 同一个索引名。
#    实测混用不同维度的模型时，写入阶段不报错，但检索会直接失败：
#      RedisSearchError: query vector blob size (2048) does not match index's expected size (4096)
# ============================================================================
