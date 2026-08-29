"""
【案例】LangChain 0.x 写法：ChatOpenAI + DeepSeek 官方 API

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld

知识点速览：
- 0.x 写法从各厂商包直接导入具体类（如 ChatOpenAI），通过 base_url 接 DeepSeek 官方 OpenAI 兼容接口。
- 配置方式：.env + load_dotenv（推荐），避免 API Key 写进代码/进 Git。
- invoke 同步调用、response.content 取回复正文。

补充说明（2026-08 最新有效信息）：
- DeepSeek 官方 API 当前模型：deepseek-v4-flash（高速版） / deepseek-v4-pro（旗舰版）。
- 旧模型名 deepseek-chat / deepseek-reasoner 已于 2026-07-24 停止使用，切勿再写。
- 若改用 langchain-deepseek 包的 ChatDeepSeek（本项目已安装），写法基本相同，但无需 base_url。
- 运行前请在项目根目录 .env 中粘贴 DEEPSEEK_API_KEY（申请地址：https://platform.deepseek.com/api_keys）。
"""

import os

from dotenv import load_dotenv  # 从 .env 文件加载环境变量，避免把 API Key 写进代码
from langchain_openai import (
    ChatOpenAI,
)  # OpenAI 兼容的聊天模型封装，可配合 base_url 接 DeepSeek 等兼容接口
from pydantic import SecretStr

# ========== 1. 大模型客户端初始化（.env + 环境变量，推荐写法） ==========

load_dotenv(encoding="utf-8")  # encoding 指定 utf-8，避免 .env 中中文注释乱码

llm = ChatOpenAI(
    model="deepseek-v4-flash",  # DeepSeek 官方当前模型；换成 deepseek-v4-pro 即为旗舰版
    api_key=SecretStr(
        os.getenv("DEEPSEEK_API_KEY") or ""
    ),  # 从 .env 读取（自己粘贴的密钥）
    base_url="https://api.deepseek.com",  # DeepSeek 官方 OpenAI 兼容接口地址
)

# 另一种写法（用 DeepSeek 官方集成包，效果相同）：
# from langchain_deepseek import ChatDeepSeek
# llm = ChatDeepSeek(model="deepseek-v4-flash", api_key=os.getenv("DEEPSEEK_API_KEY"))

# ========== 2. 调用大模型并打印结果 ==========
# invoke：同步调用，传入用户问题字符串，返回 AIMessage 等消息对象
response = llm.invoke("你是谁")

print(response)  # 打印完整对象（含 token 用量、finish_reason 等元数据，便于调试）
print()
print(response.content)  # 只取「正文」文本，即模型回复内容
