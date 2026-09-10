"""
【案例】Pydantic 入门：类型校验、自动转换与 ValidationError

对应教程章节：第 17 章 - Tools 工具调用 → 4、参数 schema：为什么要配合 Pydantic → 4.2 Pydantic 定义 / 4.3 入门案例

知识点速览：
- 本例先单独演示 Pydantic，本身不直接定义 Tool；它的作用是帮助你理解后面为什么 `args_schema` 能提升工具参数的清晰度与安全性。
- Pydantic 基于类型注解在「实例化时」做校验与转换：合法则自动转，不合法则抛 `ValidationError`。
- `StrictInt` 这类严格类型会拒绝自动转换，仅接受真实 `int`，适合在工具参数需要更严格约束时使用。
- 两个容易混的点：
  ① 静态检查（pyright）与运行校验（Pydantic）是两层：字面量写错，pyright 在编辑器里就能发现；
     而「外部来的数据」（如模型返回的 JSON）pyright 看不到，只能靠 Pydantic 在运行时兜住 —— 这正是它存在的价值。
  ② 报错文案随版本变化：下方输出示例是 pydantic V2 的格式（V1 的旧文案长这样：type_error.integer）。
"""

import json

from pydantic import BaseModel, StrictInt, ValidationError


# 继承 BaseModel：实例化时按类型注解校验，不合规则抛出 ValidationError
class User(BaseModel):
    # id: int  # 普通 int 时，传入 "41" 会被自动转成 41
    id: StrictInt  # 严格整数：不接受字符串等，必须已是 int，否则报错
    name: str
    age: int = 0  # 可选字段，默认 0；传入值会被校验并转换


# ---------- 1. 合法输入：实例化成功 ----------
# 用 try/except/else：else 里只放「没有异常时才执行」的代码。
# 这样既保留了「合法 → 成功」的对照结构，也不会触发 pyright 的
# "u is possibly unbound"（u 只在 try 里赋值，抛异常时它确实可能没定义）
try:
    u = User(id=42, name="z3")
except ValidationError as e:
    print(e)
else:
    print(u.id, type(u.id))  # 42 <class 'int'>

print()
print()


# ---------- 2. 非法输入：StrictInt 不做模糊转换 ----------
# 关键：非法值来自「外部数据」（这里用 json.loads 模拟模型/接口送来的 JSON 字符串），
# 而不再写成字面量 User(id="abc", ...)。
# 两者运行时结果一样，但前者更贴近真实场景：
# 字面量写错的话 pyright 在编辑器里就报错了（静态检查）；
# 而外部数据的值静态检查器看不见，只能靠 Pydantic 在运行时拦住 —— 这就是它的价值。
try:
    external_data = json.loads(
        '{"id": "abc", "name": "Bob"}'
    )  # 类型为 Any，pyright 不逐字段检查
    User(**external_data)
except ValidationError as e:
    print(e)

"""
【输出示例】（pydantic 2.13 实测输出）
42 <class 'int'>


1 validation error for User
id
  Input should be a valid integer [type=int_type, input_value='abc', input_type=str]
    For further information visit https://errors.pydantic.dev/2.13/v/int_type

注：如果把 StrictInt 换回普通 int，传入 "41" 会被自动转换成 41 —— 这就是上面注释里说的「自动转换」。
"""
