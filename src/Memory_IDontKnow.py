"""
【案例】「我不知道」演示：无记忆时两轮请求相互独立，模型无法利用上一轮内容

对应教程章节：第 16 章 - 记忆与对话历史 → 3、「我不知道」演示：无记忆时的行为

知识点速览：
一、核心结论：模型本身是「无状态」的
  - 一次 invoke 就是一次独立的模型调用；服务端不替你保留任何上下文。
  - 「记住上一轮」不是模型参数被改写，而是应用层每次都把历史重新拼进 Prompt 再发过去。
  - JS/TS 类比：相当于一个无状态后端接口，每次都要求带上全量 token/session；前端不带，服务端就“不认识你”。

二、本案例要验证什么
  - 只用「Prompt + Model + Parser」且不保存历史时：
    第一轮问「我叫张三，你叫什么?」→ 模型能答出「张三」，因为这句话就在本次请求里；
    第二轮问「你知道我是谁吗?」→ 模型答「我不知道」，因为第二轮请求里根本没有第一轮的内容。
  - 网页版聊天能记住多轮，是因为前端或后端实现了历史记忆（读历史 → 拼入提示 → 写回历史）；
    本章后续案例用 RunnableWithMessageHistory + 记忆组件实现该能力。

三、第二轮真正发出去的请求体（关键对照）
  只有：  [ HumanMessage(content='请回答我的问题：你知道我是谁吗?') ]
  里面没有「我叫张三」四个字 —— 模型不是“忘了”，而是压根没收到。
"""

from dotenv import load_dotenv

# 读取项目根目录 .env，把里面的键值对写进环境变量（等同于 Node 里 dotenv config()）
load_dotenv(encoding="utf-8")

import os  # 用来读环境变量，等价于 JS 的 process.env

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# ---------- 1. 大模型与简单链：仅「提示模板 → 模型 → 解析器」，无任何记忆组件 ----------
# llm = init_chat_model(
#     model="qwen-plus",
#     model_provider="openai",
#     api_key=os.getenv("aliQwen-api"),
#     temperature=0.0,
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 变量名要与 .env 里配置的一致，否则取到 None 会导致调用失败
    base_url="https://api.deepseek.com",
    temperature=0.7,  # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_completion_tokens=256,  # 可选：限制单次回复长度（新版 langchain-openai 已把 max_tokens 改名为 max_completion_tokens）
    # 注：本案例没有结构化输出，不需要 extra_body={"thinking": {"type": "disabled"}} 去关思考模式
)

# 文本提示模板：from_template 把带 {占位符} 的字符串变成可复用的模板
# 类比 JS 的模板字符串 `请回答我的问题：${question}`，只是变量在 invoke 时才注入，且键名必须完全一致
prompt = PromptTemplate.from_template("请回答我的问题：{question}")

# 字符串解析器：把模型返回的 AIMessage 取出 content 字段，转成纯 str
# 类比：把响应对象 response.data.content 拍平成字符串
parser = StrOutputParser()

# ---------- 2. LCEL 管道组合 ----------
# `|` 把前一个组件的输出喂给后一个组件，类似 Unix 管道，也类似 pipe(prompt, llm, parser)
# 数据流向：{"question": ...} → prompt 渲染成 PromptValue → llm 返回 AIMessage → parser 取出 content 转 str
# 重点：这条链是「无状态」的 —— 每个组件只负责「当次输入 → 当次输出」，没有任何地方保存历史
chain = prompt | llm | parser

# ---------- 3. 第一轮：告诉模型「我叫张三」 ----------
# .invoke 的入参是字典，等价于 JS 里 fn({ question: "..." }) 传 options 对象；
# 字典的键必须覆盖模板里的全部占位符，否则报 KeyError（正好对应 JS 取到 undefined 的场景）
# 这一轮请求体：请回答我的问题：我叫张三，你叫什么?
print(chain.invoke({"question": "我叫张三，你叫什么?"}))

# ---------- 4. 第二轮：问「你知道我是谁吗？」——模型无法看到上一轮，会回答「我不知道」 ----------
# 这一轮请求体只有：请回答我的问题：你知道我是谁吗?
# 两次 invoke 之间没有任何共享状态：换成两个进程分别执行，结果完全一样
print(chain.invoke({"question": "你知道我是谁吗?"}))

"""
【输出示例】
（以下为早期用 qwen-plus 跑出的记录；换成 deepseek-v4-flash 措辞会不同，但「第二轮答不知道」的现象一致）

第一轮（模型当然答得出张三，因为这句话就在本次请求里）：
你好，张三！我叫通义千问（Qwen），是阿里云研发的超大规模语言模型。很高兴认识你！😊

第二轮（模型答不出，因为请求里没有「我叫张三」）：
我不知道你是谁。我是一个AI助手，没有能力识别或获取用户的身份信息。如果你有任何问题需要帮助，我会很乐意为你提供支持！
"""

"""
【补充说明】
我们刚刚在本地程序，前一轮对话告诉大语言模型的信息，下一轮就被“遗忘了”。
但如果我们直接使用网页版聊天工具，它之所以能记住多轮内容，是因为应用层实现了历史记忆功能，
而不是模型参数在本地被改写。
"""

# ---------- 5. 怎么让它记住？（本章后续案例的两条路线） ----------
# 路线一：手工维护历史 —— 自己把上一轮的 HumanMessage / AIMessage 追加进 messages 列表，
#         下一轮连同整个列表一起发过去（等价于自己在前端维护一份聊天记录数组）。
# 路线二：交给 LangChain —— RunnableWithMessageHistory + 记忆组件，
#         自动完成「读历史 → 拼提示 → 写回历史」，无需手写拼接逻辑。
