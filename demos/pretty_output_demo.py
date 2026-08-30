"""
【Demo】让终端输出更舒服：美化打印模型响应的三件套
对应教程：第 14 章 JsonOutputParser 案例（美化版）

"密密麻麻"的两个根源和对策：
  1. 打印了整个对象 → 只挑关键字段（回复正文 / 模型名 / token 数）
  2. 没有排版       → rich：彩色、缩进、表格、面板

零依赖替代方案（不装 rich 时）：
  print(json.dumps(data, ensure_ascii=False, indent=2))   # JSON 缩进
  from pprint import pprint; pprint(data)                  # 标准库美化

运行（study-agent 目录下）：
    .venv/bin/python demos/pretty_output_demo.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

from rich import print as rprint          # 彩色 print：dict/list 自动缩进高亮
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------- 模型（自动选择 provider） ----------
if os.getenv("aliQwen-api"):
    model = init_chat_model(
        model="qwen-plus", model_provider="openai",
        api_key=os.getenv("aliQwen-api"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
elif os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek-api"):
    model = init_chat_model(
        model="deepseek-chat", model_provider="deepseek",
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("deepseek-api"),
    )
else:
    raise SystemExit("请先在 .env 里配置 aliQwen-api 或 DEEPSEEK_API_KEY")

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，请简短回答我提出的问题，"
               "结果返回json格式，q字段表示问题，a字段表示答案。"),
    ("human", "请回答:{question}"),
])
prompt = chat_prompt.invoke(
    {"role": "AI助手", "question": "什么是LangChain，简洁回答100字以内"}
)

# ---------- 1. 消息列表：一行一条，只看角色和内容 ----------
console.rule("[bold cyan]① 发给模型的消息")
for i, m in enumerate(prompt.messages, 1):
    console.print(f"[magenta]{i}. [{m.type}][/magenta] {m.content}")

# ---------- 2. 模型响应：只挑关键字段，不打印整个对象 ----------
result = model.invoke(prompt)

console.rule("[bold cyan]② 模型响应（只看关键信息）")
info = Table(show_header=False, box=None, pad_edge=False)
info.add_column(style="bold green", width=14)
info.add_column()
info.add_row("回复正文", result.content)
info.add_row("模型", str(result.response_metadata.get("model_name", "-")))
usage = getattr(result, "usage_metadata", None) or {}
info.add_row("输入 tokens", str(usage.get("input_tokens", "-")))
info.add_row("输出 tokens", str(usage.get("output_tokens", "-")))
console.print(info)

# ---------- 3. 解析结果：rich 彩色 JSON ----------
parser = JsonOutputParser()
data = parser.invoke(result)

console.print(Panel("③ JsonOutputParser 解析结果（彩色自动缩进）"))
rprint(data)          # ← 就这一行：dict 自动变成彩色缩进的 JSON 样子

# ---------- 4. 再进一步：把 q / a 摆成表格 ----------
console.print(Panel("④ 用表格展示解析结果"))
t = Table(show_header=True, header_style="bold magenta")
t.add_column("q（问题）", overflow="fold")
t.add_column("a（答案）", overflow="fold")
t.add_row(data.get("q", "-"), data.get("a", "-"))
console.print(t)
