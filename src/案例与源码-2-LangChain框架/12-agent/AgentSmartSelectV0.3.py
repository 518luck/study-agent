"""
【案例】多工具并行调用与聚合回答（V0.3：Agent + AgentExecutor）

对应教程章节：第 21 章 - Agent 智能体 → 3、Agent 工作原理（V0.3）

知识点速览：
- Tool 与 Agent 关系：Tool 提供能力（如查天气），Agent 负责决策「何时用、用哪个、如何聚合结果」。
  本案例中一次问题「北京和上海哪个更热」触发多次工具调用，再由 Agent 汇总比较。
- 天气工具不再写在本文件里，改为从公用工具文件 `src/tools/weather.py` 引入：
  `from tools import get_weather`（`tools` 文件夹的 `__init__.py` 负责把它转出来）。
  工具定义和 Agent 组装分开，后续示例要用天气工具时不用再抄一遍，直接引入即可。
- V0.3 流程：模型 + 工具 + 提示模板 → create_tool_calling_agent 得到 Agent → 用 AgentExecutor 执行，
  对应教程「3、Agent 工作原理（V0.3 视角）」：Agent 只做决策，Executor 负责真正调用工具并把结果传回 Agent。
- 关键组件：ChatPromptTemplate 定义对话结构（含 `agent_scratchpad` 占位符）、AgentExecutor 驱动循环。
- `agent_scratchpad` 可以理解成 Agent 的“草稿区 / 中间步骤区”，没有它，classic 路线下的多步推理就很难成立。
- `AgentExecutor(verbose=True)` 很适合教学和排查，它相当于一个轻量级的执行日志窗口；新版教程里补充的
  `stream()` / LangSmith 则是更偏 1.x 和工程化的观察手段。
- 这个文件的核心价值不是“天气查询”，而是帮助你看清 classic Agent 是如何围绕一次问题完成多次工具调用的。
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

# 天气工具统一放在 src/tools/weather.py，后续示例直接这样引入即可
from tools import get_weather

# 初始化大模型，用于理解用户问题并决定是否调用工具、如何组合结果

llm = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 变量名要与 .env 里配置的一致，否则取到 None 会导致调用失败
    base_url="https://api.deepseek.com",
    temperature=0.7,  # 0～1，越高越随机；此处略高便于看到多次输出差异
    # max_completion_tokens=256,  # 可选：限制单次回复长度（新版 langchain-openai 已把 max_tokens 改名为 max_completion_tokens）
)

# 定义 Agent 的对话结构：system 定角色，human 为用户输入，
# placeholder 供 Executor 填入中间推理与工具调用记录
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息"),
        ("human", "{input}"),
        (
            "placeholder",
            "{agent_scratchpad}",
        ),  # V0.3 必备：Agent 的「草稿本」，记录多轮推理与工具输出
    ]
)

tools = [get_weather]

# 将 LLM、工具列表、提示模板组装成「可做工具调用决策」的 Agent（尚未执行）
agent = create_tool_calling_agent(llm, tools, prompt)

# AgentExecutor 负责循环：调用 Agent → 执行其选中的工具 →
# 把结果写回 agent_scratchpad → 再交给 Agent，直到结束
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 一次问题触发多工具调用（北京、上海天气）并聚合回答
result = agent_executor.invoke(
    {"input": "请问今天北京和上海的天气怎么样，哪个城市更热？"}
)

print(result)

"""
【输出示例】（工具改用 src/tools 下的和风天气版后的实跑结果，JSON 已截断）
> Entering new AgentExecutor chain...

Invoking: `get_weather` with `{'loc': '北京'}`

{"condition": {"text": "晴", "code": "100"}, "temperature": {"value": 30.3, "unit": "°C"}, "feelsLike": {"value": 26.72, "unit": "°C"}, "humidity": 0.16, ..., "location": {"name": "北京", "adm1": "北京市", "id": "101010100", "lat": 39.9, "lon": 116.41}}
Invoking: `get_weather` with `{'loc': '上海'}`

{"condition": {"text": "多云", "code": "101"}, "temperature": {"value": 29.2, "unit": "°C"}, ..., "location": {"name": "上海", "adm1": "上海市", "id": "101020100", "lat": 31.23, "lon": 121.47}}

> Finished chain.
{'input': '请问今天北京和上海的天气怎么样，哪个城市更热？', 'output': '## 今日天气对比（北京 vs 上海）\n\n| 项目 | 北京 | 上海 |\n|---|---|---|\n| 天气现象 | 晴 | 多云 |\n| 气温 | **30.3 °C** | 29.2 °C |\n| 体感温度 | 26.7 °C | **29.7 °C** |\n| 相对湿度 | 16%（很干燥） | 48%（较湿润） |\n...\n\n### 哪个更热？\n- **看气温：北京更热**，30.3 °C 比上海高约 1 °C。\n- **看体感：上海更热**，体感 29.7 °C 明显高于北京的 26.7 °C。\n\n原因在于**湿度差异**：北京空气非常干燥... 上海湿度接近 50%，闷热感更强。\n\n*数据来源：和风天气实时接口*'}

说明：同一道问题触发了两次 get_weather（北京、上海），模型再把两份 JSON 汇总成对比结论，
这正是「Agent 决策 + 工具提供实时数据」的分工。
"""
