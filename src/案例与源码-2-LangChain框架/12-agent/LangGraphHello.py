"""
【案例】LangGraph 最简 HelloWorld：用「State + Nodes + Edges + Graph」构建一张最小线性图（START → greeting → add_emoji → END），体会 Graph API 的入门写法和图可视化方式

对应教程章节：第 22 章 - LangGraph 概述与快速入门 → 2、HelloWorld 快速入门

知识点速览：
- State：用 TypedDict 定义状态字段（如 name、greeting），表示这张图运行过程中会保存哪些数据。
- Nodes：每个节点本质上都是一个函数，接收当前 state，返回“本节点要更新的字段字典”，不需要手动拼完整状态。
- Edges：add_edge 定义执行顺序；START / END 为虚拟起止节点。
- Graph API 入门主流程：定义 State → 定义节点函数 → StateGraph(State) → add_node / add_edge → compile() → invoke(initial_state)。
- 可视化：compile() 之后可通过 get_graph().print_ascii() 和 draw_mermaid() 查看图结构；输出里的 __start__、__end__ 是 LangGraph 内置虚拟节点名，不要自定义同名节点。
- 本案例虽然是线性流程，但它已经是后续“分支、循环、多节点 LLM 图”的最小雏形；第 23 章会继续展开 State Schema 和 Reducer。
"""

import uuid
from typing import TypedDict, cast

from langgraph.graph import END, START, StateGraph


# 1. 定义 State（状态）：声明图中要传递的字段及类型（可选）
class HelloState(TypedDict):
    name: str
    greeting: str


# 2. 定义节点函数 Node：接收当前 state，返回对 state 的「部分更新」字典
# 注意：参数名必须叫 state（不能写成 helloState）——LangGraph 的节点函数协议就要求这个名字，
# 叫别的名字运行没问题，但 pyright 类型检查会报 Parameter name mismatch。
# 返回类型写成 dict[str, str] 而不是裸 dict：明确告诉类型检查器返回的字段类型。
def greet(state: HelloState) -> dict[str, str]:
    name = state["name"]
    return {"greeting": f"你好,{name}"}


def add_emoji(state: HelloState) -> dict[str, str]:
    greeting = state["greeting"]
    return {"greeting": greeting + "  。。。😄"}


# 3. 构建图 Graph：初始化 StateGraph，添加节点与边
graph = StateGraph(HelloState)
graph.add_node("greeting", greet)
graph.add_node("add_emoji", add_emoji)
graph.add_edge(START, "greeting")
graph.add_edge("greeting", "add_emoji")
graph.add_edge("add_emoji", END)


# 4. 编译图，得到可执行的 app
app = graph.compile()

# 5. 运行：invoke 只接收一个核心参数——初始状态字典
# 这里用 cast 断言成 HelloState：运行时只提供 name 字段（greeting 由节点补上），
# 但类型检查器按"完整状态"要求所有字段，cast 是告诉它"这里按 HelloState 对待"的标准做法。
# 用法与 Agent2Agent.py 里 cast(StructuredTool, ...) 完全一致。
initial_state = cast(HelloState, {"name": "z3"})
result = app.invoke(initial_state)
print(result)
print(result["greeting"])

# 6. 可视化：ASCII 和 Mermaid 两种方式最适合入门阶段快速看图结构
print(app.get_graph().print_ascii())
print("=" * 50)
print(app.get_graph().draw_mermaid())
print("=" * 50)

# 可选：生成 PNG 图片（依赖 mermaid.ink 或 Pyppeteer，易受网络影响）
png_bytes = app.get_graph().draw_mermaid_png(max_retries=2, retry_delay=2.0)
output_path = "langgraph" + str(uuid.uuid4())[:8] + ".png"
with open(output_path, "wb") as f:
    f.write(png_bytes)
print(f"图片已生成：{output_path}")

"""
【输出示例】
{'name': 'z3', 'greeting': '你好,z3  。。。😄'}
你好,z3  。。。😄

+-----------+
| __start__ |
+-----------+
      *
      *
      *
+----------+
| greeting |
+----------+
      *
      *
      *
+-----------+
| add_emoji |
+-----------+
      *
      *
      *
 +---------+
 | __end__ |
 +---------+
None                      ← print_ascii() 自己就打印了，外面那层 print() 打印的是它的返回值 None
==================================================
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        greeting(greeting)
        add_emoji(add_emoji)
        __end__([<p>__end__</p>]):::last
        __start__ --> greeting;
        greeting --> add_emoji;
        add_emoji --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
