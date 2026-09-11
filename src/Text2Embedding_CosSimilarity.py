"""
【案例】通过向量计算语义相似度：余弦相似度（LangChain + 硅基流动版）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 5.4 案例：把多句话转成向量，再两两比较

知识点速览：
- 本文件是教程 5.4 案例的改造版：把「DashScope 原生接口」换成「LangChain 统一 Embeddings 接口 + 硅基流动」。
  LangChain 的 Embeddings 接口只处理文本（embed_query / embed_documents），所以这里用常规文本模型 BAAI/bge-m3；
  这也正对应教程 5.4 自己的说明——相似度计算不是必须用多模态模型，常规文本 Embedding 模型一样能完成。
- 本案例的重点不是模型，而是“向量一旦拿到手，就可以做数学比较”：余弦相似度 cos(θ) = (A·B) / (|A||B|)，
  值域 [-1, 1]，越接近 1 一般表示越相似。
- LangChain 侧两个要点：
  1）init_embeddings 是 init_chat_model 的 Embedding 对应物，同样一行完成厂商切换；
  2）embed_documents(texts) 一次把多句话批量转成向量，不用自己写 for 循环。
- 接非 OpenAI 官方网关有个必须注意的坑：LangChain 默认会先用 tiktoken 把文本切成 token 数组再发送
  （check_embedding_ctx_length=True）。硅基流动收到这种入参**不会报错**，但它用自己的分词器去解读这串 token，
  等于把“另一串文本”拿去向量化——实测两种模式算出的向量余弦只有 0.33，几乎不相关。
  所以这里必须显式设成 False 直接发原始文本，否则拿到的是静默错误的结果。
- 相似度比较常用于检索排序、文本去重、聚类、推荐。
"""

import os

import numpy as np
from dotenv import load_dotenv
from langchain.embeddings import init_embeddings

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

# 提前拦一道，避免拿着空 key 去请求，只看到一句含义不明的 401
if not api_key:
    raise SystemExit(
        "❌ 未读取到 SILICONFLOW_API_KEY，请在项目根目录的 .env 中补充：\n"
        "   SILICONFLOW_API_KEY=你的密钥\n"
        "   申请地址：https://cloud.siliconflow.cn/me/account/ak"
    )

# 准备多句文本，用于观察“语义越接近，相似度通常越高”
texts = ["我喜欢吃苹果", "苹果是我最喜欢吃的水果", "我喜欢用苹果手机"]

# 硅基流动走 OpenAI 兼容协议：provider 仍填 openai，只把 base_url / api_key / model 三处换成硅基流动的
embeddings = init_embeddings(
    "BAAI/bge-m3",
    provider="openai",
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
    # 关键参数：直接发原始文本，不把文本转成 token 数组。
    # 不设成 False 时不会报错，但平台会用错分词器解读 token 数组，向量静默出错（实测两者余弦仅 0.33）
    check_embedding_ctx_length=False,
    # 硅基流动单次请求最多 32 条文本，把分片大小对齐到它的上限，文本变多时也不会被平台拒绝
    chunk_size=32,
)

# 批量向量化：一次调用拿到全部向量（bge-m3 每条 1024 维）
vectors = embeddings.embed_documents(texts)
print(f"向量：{vectors}")
print(f"向量数量：{len(vectors)}，每个向量维度：{len(vectors[0])}")
print()


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度：点积 / (模长之积)，结果越接近 1 一般越相似"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


print("文本相似度比较结果:")
print("=" * 60)

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        similarity = cosine_similarity(vectors[i], vectors[j])
        print(f"文本{i + 1} vs 文本{j + 1}:")
        print(f"  文本{i + 1}: {texts[i]}")
        print(f"  文本{j + 1}: {texts[j]}")
        print(f"  余弦相似度: {similarity:.4f}")
        print("-" * 40)

"""
【输出示例】以下为 BAAI/bge-m3 实跑的真实输出

向量数量：3，每个向量维度：1024
文本相似度比较结果:
============================================================
文本1 vs 文本2:
  文本1: 我喜欢吃苹果
  文本2: 苹果是我最喜欢吃的水果
  余弦相似度: 0.8839
----------------------------------------
文本1 vs 文本3:
  文本1: 我喜欢吃苹果
  文本3: 我喜欢用苹果手机
  余弦相似度: 0.8636
----------------------------------------
文本2 vs 文本3:
  文本2: 苹果是我最喜欢吃的水果
  文本3: 我喜欢用苹果手机
  余弦相似度: 0.7743
----------------------------------------

观察：排序符合直觉（1-2 最像、2-3 最不像），但三句都在 0.77 以上，因为三句共有“苹果”“喜欢”这些字面重合，
说明短文本相似度对字面重合很敏感——“吃苹果”和“用苹果手机”主题完全不同，分数却高达 0.8636。
真实检索里遇到这种“字面像、意思不同”的情况，通常要靠重排序（rerank）或换更强的模型再区分。

注：向量在不同次调用间有极小波动（实测余弦 0.99997），所以重跑时上面的末位数字可能差千分之几（如 0.7743 / 0.7751），
属平台侧正常现象，不影响结论。
"""
