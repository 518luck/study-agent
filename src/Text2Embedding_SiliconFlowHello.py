"""
【案例】硅基流动（SiliconFlow）OpenAI 兼容接口：单句文本向量化（Hello 级）

对应教程章节：第 18 章 - 向量数据库与 Embedding 实战 → 4.4 案例：OpenAI 兼容写法，理解“同一能力，不同接法”
（本文件是同章 4.3 案例 Text2Embedding_DashScopeHello.py 的硅基流动版本，目标同样是先看到“向量长什么样”）

知识点速览：
- 硅基流动没有自己的原生 SDK，对外提供的就是 OpenAI 兼容接口，所以接法和教程 4.4 完全一致：
  调用代码仍是 OpenAI SDK，只改三处配置——base_url、api_key、model。
- 与 DashScope 原生调用的两个直观差异：
  1）请求地址从厂商私有网关换成 https://api.siliconflow.cn/v1（embeddings 路径由 SDK 自动拼接）；
  2）OpenAI SDK 调用失败会直接抛异常，不再需要自己判断 status_code == 200。
- 本案例默认用 BAAI/bge-m3：1024 维、单条最长 8192 token，中文表现好且目前处于免费/极低价档，适合作为入门模型。
- input 可传单个字符串或字符串列表（文档写单次最多 32 条，但实测 33/50/100 条也能过，该上限没被强制执行）；
  向量在 data[0].embedding，向量长度即模型维度。
- 换模型只需改 model：Qwen/Qwen3-Embedding-0.6B（默认 1024 维）/ 4B（2560 维）/ 8B（4096 维），
  Qwen3 系列额外支持 dimensions 参数自定义输出维度（如 dimensions=1024），bge 系列不支持该参数。

接口文档：https://docs.siliconflow.cn/cn/api-reference/embeddings/create-embeddings
API Key 申请：https://cloud.siliconflow.cn/me/account/ak
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("SILICONFLOW_API_KEY")

# 提前拦一道，避免拿着空 key 去请求，只看到一句含义不明的 401
if not api_key:
    raise SystemExit(
        "❌ 未读取到 SILICONFLOW_API_KEY，请在项目根目录的 .env 中补充：\n"
        "   SILICONFLOW_API_KEY=你的密钥\n"
        "   申请地址：https://cloud.siliconflow.cn/me/account/ak"
    )

# 待向量化的单句文本（与 DashScope 版保持一致，方便对照两个平台的输出）
input_text = "衣服的质量杠杠的"

# 硅基流动走 OpenAI 兼容协议：仍用 OpenAI SDK，只是把网关地址和密钥换成硅基流动的
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1",
)

# 调用方式与 OpenAI Embedding 完全一致：model 为硅基流动的模型名，input 为待向量化文本
completion = client.embeddings.create(
    model="BAAI/bge-m3",
    input=input_text,
)

# 先打印完整响应，是为了观察响应结构；这里和 DashScope 原生返回的字段名大体一致
print(completion.model_dump_json())

# 单独把向量取出来看看它“长什么样”：长度即维度，前几个数字是本条文本落在各维上的取值
embedding = completion.data[0].embedding
print(f"\n向量维度：{len(embedding)}")
print(f"前 5 个数字：{embedding[:5]}")

"""
【输出示例】以下为 BAAI/bge-m3 实跑的真实输出（响应里的向量过长，此处折叠显示；脚本运行时终端会完整打印）

{"data":[{"embedding":[……此处省略 1024 个浮点数……],"index":0,"object":"embedding"}],"model":"BAAI/bge-m3","object":"list","usage":{"prompt_tokens":9,"total_tokens":9,"completion_tokens":0}}

向量维度：1024
前 5 个数字：[-0.013220297172665596, -0.012605398893356323, -0.07378770411014557, 0.03335819020867348, -0.008454840630292892]

注：同一句话在不同时间调用，向量可能有极小波动（同一进程内多次调用逐位相同；隔约十分钟的两次调用实测余弦 0.99997、
单点最大差 0.002），属平台侧正常现象，对相似度检索没有影响。所以上面的数值只要量级和趋势一致即可，不必逐位比对。
"""
