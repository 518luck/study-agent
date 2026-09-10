"""
【案例】加法工具（Pydantic 版）：用 args_schema 绑定参数模型，让模型看到规范参数说明

对应教程章节：第 17 章 - Tools 工具调用 → 4、参数 schema：为什么要配合 Pydantic → 4.4 加法工具的 Pydantic 版

知识点速览：
- `args_schema` 的价值，不只是“参数能校验”，更重要的是把工具参数的语义说明显式暴露给模型，提升参数生成正确率。
- `tool.args` 主要反映参数 schema；而工具的整体用途说明，仍然应该优先写在函数 docstring 中，不建议完全依赖参数模型的类注释。
- `return_direct` 等参数更偏 Agent 场景；本例的重点放在“如何让工具定义更像一份清晰的小接口文档”。
"""

from langchain_core.tools import tool
from loguru import logger
from pydantic import BaseModel, Field


# Pydantic 模型：定义“工具参数接口”，字段 description 会进入工具参数 schema
class FieldInfo(BaseModel):
    """定义加法运算所需的参数结构"""

    a: int = Field(description="第1个参数")
    b: int = Field(description="第2个参数")


# args_schema=FieldInfo：把参数模型绑定到工具，模型会更清楚看到 a、b 的类型与说明
@tool(args_schema=FieldInfo)
def add_number(a: int, b: int) -> int:
    """计算两个整数之和"""
    return a + b


# 打印工具属性：带 args_schema 时，args 中会包含 Field 的 description
logger.info(f"name = {add_number.name}")
logger.info(f"args = {add_number.args}")
logger.info(f"description = {add_number.description}")
logger.info(f"return_direct = {add_number.return_direct}")

# 调用工具：传入字典，Pydantic 会做类型校验与转换
res = add_number.invoke({"a": 1, "b": 2})
logger.info(res)

"""
【输出示例】
name = add_number
args = {'a': {'description': '第1个参数', 'title': 'A', 'type': 'integer'}, 'b': {'description': '第2个参数', 'title': 'B', 'type': 'integer'}}
description = 计算两个整数之和
return_direct = False
3
"""
