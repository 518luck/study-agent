"""
【案例】使用 langchain_redis 将文本写入 Redis 向量库（add_texts）—— 硅基流动版

对应教程章节：第 19 章 - RAG 检索增强生成 → 2.1.1 from_documents 与 add_texts；也可与第 18 章向量库写入案例对照阅读

知识点速览：
- 这个案例展示的是纯文本流驱动的入库路线：先创建 `RedisVectorStore`，再通过 `add_texts()` 把字符串列表写入向量库。
- `add_texts(texts, metadata)` 会在内部调用 `embed_documents(texts)` 做批量向量化，然后把文本、向量和 metadata 一起写入 Redis。
- 这条路线和 `from_documents(...)` 并不冲突：前者更适合你手里已经是纯文本列表，后者更适合你已经有 `Document` 列表。
- 本例里额外手动执行了一次 `embed_documents`，目的是先观察“向量长什么样、维度是多少”；真正做存储时，这一步不是必须的。
- 返回的 ids 可用于后续更新、删除或追踪；index_name 需要和后续检索端保持一致。
- 本文件是教程案例的硅基流动改造版：把「langchain_community 的 DashScopeEmbeddings」换成
  「init_embeddings + 硅基流动 bge-m3」，并顺手补了三处教程没写、但实操会踩到的点（见文件末尾注释）：
  1）langchain_redis 与 langchain_community 的 Redis 向量库存储格式完全不同，两套数据不能混用同一个索引；
  2）RedisConfig 的 redis_url 默认是 6379，本项目向量库在 26379，必须显式传；
  3）add_texts 的 keys 参数可实现“幂等写入”，是防止重复入库的正规做法（示例里有演示）。
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

# 1. 初始化嵌入模型：硅基流动走 OpenAI 兼容协议，provider 仍填 openai
embeddingsModel = init_embeddings(
    "BAAI/bge-m3",
    provider="openai",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    # 直接发原始文本；不设 False 会把文本转成 token 数组，平台用错分词器解读，向量静默出错（实测余弦仅 0.33）
    check_embedding_ctx_length=False,
)

# 2. 待写入的文本及（可选）元数据
texts = [
    "我喜欢吃苹果",
    "苹果是我最喜欢吃的水果",
    "我喜欢用苹果手机",
]


# 批量转成向量：这里只是为了先观察向量维度和内容；真正写入时 add_texts 内部会再次完成向量化
embeddings = embeddingsModel.embed_documents(texts)
for i, vec in enumerate(embeddings, 1):
    print(f"文本 {i}: {texts[i - 1]}")
    print(f"向量长度: {len(vec)}")
    print(f"前5个向量值: {vec[:10]}\n")

# 定义每条文本对应的元数据信息
# metadata = [{"segment_id": "1"}, {"segment_id": "2"}, {"segment_id": "3"}]

# 定义每条文本对应的元数据信息；真实 RAG 中这些 metadata 往往来自 Document.metadata，也可作为来源展示或过滤条件
metadata = [{"segment_id": str(i)} for i in range(1, len(texts) + 1)]

# 3. Redis 连接与索引名（需与检索案例一致）
# redis_url 必须显式传：RedisConfig 的默认值是 redis://localhost:6379，而本项目向量库跑在 26379 的 Redis Stack 上
config = RedisConfig(
    index_name="newsgroups",
    redis_url=os.getenv("REDIS_URL", "redis://localhost:26379"),
)

# 创建 Redis 向量存储实例：此时只是“连上库 + 指定索引配置”，还没真正写入文本；真正写入发生在 add_texts()
vector_store = RedisVectorStore(embeddingsModel, config=config)

# 4. 将文本与元数据写入向量库（add_texts 内部会调 embed_documents，无需先算向量）
# 不传 keys 时，langchain_redis 会为每条文本生成 ULID 作为 id（前 10 位是写入时刻的时间戳，所以天然按时间递增）
# 注意：这种自动 id 每次都是新的，所以本步骤是“追加”——脚本每跑一次，索引里就多 3 条重复文本
ids = vector_store.add_texts(texts, metadata)

# 打印前5个存储记录的ID
print(ids[0:5])

# 5.（可选）幂等写入：keys 传“裸 id”，用同一批 id 再写就是覆盖而不是追加
# 不传 keys 的话每次都生成新 ULID，脚本多跑几遍库里就会积累重复文本（社区版案例踩过这个坑）
FIXED_KEYS = ["apple-1", "apple-2", "apple-3"]
print("\n用固定 id 连续写入两次，观察返回的 id 是否稳定：")
for round_no in (1, 2):
    ids_fixed = vector_store.add_texts(texts, metadata, keys=FIXED_KEYS)
    print(f"  第 {round_no} 次 -> {ids_fixed}")

"""
【输出示例】以下为实跑的真实输出（向量值截断了后面几位）

文本 1: 我喜欢吃苹果
向量长度: 1024
前5个向量值: [-0.016413753852248192, 0.025761496275663376, -0.040335144847631454, 0.017223400995135307, -0.02046198956668377, ...]

文本 2: 苹果是我最喜欢吃的水果
向量长度: 1024
前5个向量值: [-0.0006920063169673085, 0.027276962995529175, -0.04340850189328194, 0.01891789399087429, -0.01598488725721836, ...]

文本 3: 我喜欢用苹果手机
向量长度: 1024
前5个向量值: [-0.013315897434949875, 0.008071367628872395, -0.08569783717393875, -0.012795163318514824, 0.005095748230814934, ...]

['newsgroups:01M28N6P4Z243MJ7VJPE2S7AP0', 'newsgroups:01M28N6P4Z243MJ7VJPE2S7AP1', 'newsgroups:01M28N6P4Z243MJ7VJPE2S7AP2']

用固定 id 连续写入两次，观察返回的 id 是否稳定：
  第 1 次 -> ['newsgroups:apple-1', 'newsgroups:apple-2', 'newsgroups:apple-3']
  第 2 次 -> ['newsgroups:apple-1', 'newsgroups:apple-2', 'newsgroups:apple-3']
"""

# ============================================================================
# 【实操补充】教程没写、但换到 langchain_redis 后必须知道的四个点（均实测）
#
# 1）存储格式与 langchain_community 的 Redis 向量库完全不同，两套数据不能混用同一个索引名：
#      langchain_redis      ：text（正文）、embedding（向量）、_metadata_json（元数据 JSON）+ 展开的元数据字段
#      langchain_community  ：content（正文）、content_vector（向量）、元数据拆成独立字段
#    另外键名也不同：langchain_redis 是「索引名:ULID」，社区版是「doc:索引名:uuid4」。
#    好处是 langchain_redis 的元数据存在固定字段里，换个进程重新连接也能直接取回，不需要像社区版那样声明 schema。
#
# 2）RedisConfig 的 redis_url 默认值是 redis://localhost:6379（不是本项目的 26379）。
#    忘传这个参数时，报错会指向一个根本没在跑的 Redis，很容易看懵。
#
# 3）索引名要与后续检索端一致，且换过 embedding 模型后必须换新索引名重建：
#    不同模型产出的向量不在同一个空间，混在同一个索引里比较没有意义。
#
# 4）自动 id 是“追加”，固定 key 才是“覆盖”——这决定脚本能不能反复运行：
#    第 4 步不传 keys，每条都生成新 ULID，所以每跑一次脚本索引里就多 3 条重复文本（实测跑 3 次得到 9 条）；
#    第 5 步用同一批 keys，连续写两次索引里始终只有 3 条（实测验证）。
#    所以想反复运行要么像上面那样传固定 keys，要么先清索引：
#      docker compose exec redis-stack redis-cli FT.DROPINDEX newsgroups DD
# ============================================================================
