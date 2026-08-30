"""
【案例】Python 入门：Pydantic + Annotated 实现带范围的运行时校验

对应教程章节：第 14 章 - 输出解析器 → 3、结构化输出（TypedDict / Pydantic / Annotated）

知识点速览：
一、和 AnnotatedTypedDict 的区别？
  - AnnotatedTypedDict.py：用 typing.Annotated 只加了「描述」元数据，**运行时不会**按「0–150」校验。
  - 本文件：用 **Pydantic** 的 Field(ge=0, le=150) 放在 Annotated 里，即 Annotated[int, Field(ge=0, le=150, ...)]，Pydantic 会在**运行时**校验数值是否在 0–150，超出则抛 ValidationError。

二、适用场景：需要「强类型 + 取值范围校验」时，用 Pydantic 模型 + Field 约束；LangChain 的结构化输出也可用这类模型做解析与校验。
"""

from typing import Annotated

from pydantic import BaseModel, Field, ValidationError

# 用 Annotated 结合 Pydantic 的 Field：ge=0, le=150 会在运行时校验，不在范围内会触发 ValidationError
Age = Annotated[int, Field(ge=0, le=150, description="年龄，范围0-150")]


class Person(BaseModel):
    name: str
    age: int
    age2: Age  # 这里 age2 会被 Pydantic 按 Field(ge=0, le=150) 校验


try:
    p = Person(name="z3", age=11, age2=128)  # age2=188 超出 0–150，会抛 ValidationError
    print(p)
except ValidationError as e:
    print("数据校验失败：")
    print(e)

"""
【输出示例】
数据校验失败：
1 validation error for Person
age2
  Input should be less than or equal to 150 [type=less_than_equal, input_value=188, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/less_than_equal
"""
