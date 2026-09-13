"""
【案例】多工具并行调用与聚合回答（V1.0：create_agent 一步创建 + 结构化输出）

对应教程章节：第 21 章 - Agent 智能体 → 4、Agent 工作原理（V1.0）

知识点速览：
- V1.0 与 V0.3 对比：不再手写 PromptTemplate、create_tool_calling_agent、AgentExecutor，改为
  create_agent(model, tools, system_prompt, response_format=...) 一步得到可调用的 Agent，对应教程「4、Agent 工作原理（V1.0）」。
- 结构化输出：通过 response_format 指定 TypedDict（如 WeatherCompareOutput），Agent 的返回中会包含
  structured_response 字段，便于程序化处理（如比温度、写结论），而不必从自然语言里再解析。
- 本文件重点演示 `create_agent` 最常见的 4 个输入：`model / tools / system_prompt / response_format`。
  教程里还补充了 `checkpointer / middleware` 这两个更偏工程化的扩展点，但这里不作为主线展开。
- 调用方式：当前示例用 `agent.invoke(...)` 直接看最终结果；如果真实项目里想看中间进展，通常还会配合
  `stream()`，如果想做短期记忆，则会进一步引入 `checkpointer + thread_id`。
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from typing_extensions import (
    TypedDict,
)  # Python < 3.12 下 Pydantic 要求用 typing_extensions.TypedDict

from tools import get_weather

# .env 在项目根目录，从任意子目录运行脚本时都从根目录加载
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


# @tool
# def get_weather(loc: str) -> str:
#     """
#     查询即时天气函数
#     :param loc: 城市英文名，如 Beijing、Shanghai。
#     :return: OpenWeather API 返回的天气信息（JSON 字符串）。
#     """
#     url = "https://api.openweathermap.org/data/2.5/weather"
#     params = {
#         "q": loc,
#         "appid": os.getenv("OPENWEATHER_API_KEY"),
#         "units": "metric",
#         "lang": "zh_cn",
#     }
#     response = httpx.get(url, params=params, timeout=30)
#     data = response.json()
#     return json.dumps(data, ensure_ascii=False)


# 定义结构化输出：Agent 最终回答会按此结构填充，便于代码中直接取字段
class WeatherCompareOutput(TypedDict):
    beijing_temp: float
    shanghai_temp: float
    hotter_city: str
    summary: str


# model = ChatOpenAI(
#     model="qwen-plus",
#     api_key=os.getenv("aliQwen-api"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )


model = init_chat_model(
    model="deepseek-v4-flash",
    model_provider="openai",
    api_key=os.getenv(
        "DEEPSEEK_API_KEY"
    ),  # 变量名要与 .env 里配置的一致，否则取到 None 会导致调用失败
    base_url="https://api.deepseek.com",
    temperature=0.7,  # 0～1，越高越随机；此处略高便于看到多次输出差异
    # 必需：DeepSeek 默认开启思考模式，而本示例的结构化输出（response_format）会让
    # LangChain 强制 tool_choice，思考模式不支持这个组合，必须禁用，否则报
    # "Thinking mode does not support this tool_choice"（V0.3 示例没强制选工具，所以不需要）
    extra_body={"thinking": {"type": "disabled"}},
    # max_completion_tokens=256,  # 可选：限制单次回复长度（新版 langchain-openai 已把 max_tokens 改名为 max_completion_tokens）
)


# V1.0 一步创建 Agent：模型、工具、系统提示、输出格式一次传入
# 如果后面还要扩展短期记忆或拦截控制，通常会继续给 create_agent 传 checkpointer / middleware
agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt=(
        "你是天气助手。"
        "当用户询问多个城市天气时，"
        "你需要分别调用工具获取数据，并进行比较分析。"
    ),
    response_format=WeatherCompareOutput,
)


# 调用 Agent：注意 V1.0 的入参是 messages（消息列表），而不是 V0.3 的 {"input": ...}
# 返回结果中包含 messages 与 structured_response（若指定了 response_format）
# 这里先用 invoke 看最终结果；如需观察中间步骤，可在工程里改为 stream()
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "请问今天北京和上海的天气怎么样，哪个城市更热？",
            }
        ]
    }
)
print(result)
print()
print(json.dumps(result["structured_response"], ensure_ascii=False, indent=2))


"""
【输出示例】（改用 DeepSeek + src/tools 的和风天气工具后的实跑结果，JSON 已截断）
{'messages': [HumanMessage(content='请问今天北京和上海的天气怎么样，哪个城市更热？'),
  AIMessage(tool_calls=[{'name': 'get_weather', 'args': {'loc': '北京'}}, ...]),   ← 并行发起两地查询
  ToolMessage('{"condition": {"text": "晴"}, "temperature": {"value": 29.98, ...}}'),
  ToolMessage('{"condition": {"text": "多云"}, "temperature": {"value": 28.4, ...}}'),
  AIMessage(tool_calls=[{'name': 'WeatherCompareOutput', 'args': {...}}]),        ← 结构化输出也是"一次工具调用"
  ToolMessage("Returning structured response: {...}")],
 'structured_response': {'beijing_temp': 29.98, 'shanghai_temp': 28.4, 'hotter_city': '北京', 'summary': '...'}}

{
  "beijing_temp": 29.98,
  "shanghai_temp": 28.4,
  "hotter_city": "北京",
  "summary": "今天北京晴，气温约30.0°C..."
}

说明：V1.0 与 V0.3 的业务完全相同，区别在输出——这里最后拿到的是可直接取字段的
structured_response（而不是一段自然语言），它本质上是模型对 WeatherCompareOutput
这个 schema 发起的一次"工具调用"。
"""
