"""
【案例】基础加法工具：使用 @tool 装饰器将普通函数转为 LangChain Tool

对应教程章节：第 17 章 - Tools 工具调用 → 3、自定义 Tool：先从最简单的工具开始 → 3.1 使用 @tool 装饰器 / 3.2 基础案例：加法工具

知识点速览：
一、@tool 是「装饰器」，做的事和你前端写的 debounce/memo 是同一类：
    add_number = tool(add_number)   ← @tool 这行语法糖完全等价于这一句。
    函数进、新对象出，只是出来的不再是函数，而是 StructuredTool 对象。

二、装饰后调用方式变了（最容易踩的坑，实测）：
    add_number(1, 12)              → TypeError: 'StructuredTool' object is not callable
    add_number.invoke({"a": 1, "b": 12})   → 13   ✅ 必须走 invoke
    原因：名字没变，但绑定的对象被换掉了，普通函数调用语法不再适用。

三、三样元信息就是「给模型看的说明书」（本例最后一行打印的就是它们）：
    .name        工具名         ← 默认取函数名，可显式传入 @tool("other_name")
    .description 工具描述       ← 默认取函数的 docstring，模型靠它判断「什么时候该调用我」
    .args        参数 JSON Schema ← 从函数的**类型注解**自动生成（int → "type": "integer"）
    所以：docstring 别省；参数要求复杂时再用 args_schema 补充约束。

四、本案例的执行者是「你自己」，不是模型：
    tool.invoke(...) 是程序侧直接执行工具，用来先看清 Tool 本身长什么样；
    「让模型自己决定调用哪个工具」要靠后面的 bind_tools + 工具调用循环，那是另一个层次。

五、工具也是 Runnable：invoke / ainvoke / batch / stream 一应俱全（第 15 章学的接口在这里继续生效）。
"""

from langchain_core.tools import tool


# @tool 装饰器：不写参数时，工具名默认为函数名 add_number，description 取自下方 docstring
@tool
def add_number(a: int, b: int) -> int:
    """两个整数相加"""
    return a + b


# 上面三行的等价展开（装饰器机制，不写 @ 时就是这样）：
#     def add_number(a, b): ...
#     add_number = tool(add_number)   # 函数被替换成 StructuredTool 对象


# 直接执行工具：invoke 接收参数字典，键为参数名，值为参数值（与函数签名对应）
# 注意不能写成 add_number(1, 12)：装饰后它已经不是可调用的函数了（实测报 TypeError）
result = add_number.invoke({"a": 1, "b": 12})
print(result)

print()

# 查看工具元信息：这些内容正是模型后续理解工具时会重点参考的部分
# f"{x=}" 是 f-string 的「调试简写」(Python 3.8+)，会自动打印成 变量名=值，
# 等价于 JS 里 console.log({ x }) 的简写效果，省得手写 "x=" + repr(x)
print(f"{add_number.name=}\n{add_number.description=}\n{add_number.args=}")

# 观察输出可知：
# - .description 就是那句 docstring；
# - .args 的 JSON Schema 是从 a: int / b: int 推断出来的（int → "type": "integer"）；
# - 这三样合起来，就是模型判断「该不该调、怎么传参」的全部依据。
#
# 想看更完整的 schema（含 title/必填项等）可以打印：add_number.args_schema.model_json_schema()

"""
【输出示例】
13

add_number.name='add_number'
add_number.description='两个整数相加'
add_number.args={'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}
"""
