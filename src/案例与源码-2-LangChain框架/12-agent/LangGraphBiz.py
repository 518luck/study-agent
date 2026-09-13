"""
【案例】不接入大模型的业务图：用自定义加法/减法节点演示“同一个状态字段 x 如何沿着节点逐步更新”，帮助理解 Graph 不一定每个节点都要调用 LLM

对应教程章节：第 22 章 - LangGraph 概述与快速入门 → 2、HelloWorld 快速入门

知识点速览：
- 用 TypedDict 声明状态字段（本文件是 BizState：一个 int 字段 x），字段清晰、类型检查器也能校验；
  教程原版这里用的是 StateGraph(dict)，运行没问题，但 pyright 会报类型不匹配（LangGraph 的类型协议
  要求状态是 TypedDict 这类结构化类型），所以本项目改成了 TypedDict 写法。
- 节点函数接收 state，返回要更新的键值对（如 {"x": state["x"] + 1}），LangGraph 会按默认 Reducer 合并；未显式指定 Reducer 时，通常就是“新值覆盖旧值”。
- add_edge 串联 START → addition → subtraction → END，形成固定执行顺序。
- graph.edges / graph.nodes 可在 compile() 前查看当前图已注册的边与节点，适合排查“节点没加进去”或“边连错了”的问题。
- 本例是一条固定线性业务流；如果后面出现“根据 x 的值决定走不同分支”或“某一步失败后回退重跑”，就更能体现 LangGraph 相比单纯 LCEL Chain 的价值。
"""

from typing import TypedDict

from langgraph.constants import END, START
from langgraph.graph import StateGraph


# 定义状态结构：用 TypedDict 明确声明"这张图里有 x 这个字段、类型是 int"。
# x 是必填字段：节点一进来就要读 state["x"]，所以初始状态必须提供它（invoke 时给 {"x": 5}）。
# 教程原版用的是 StateGraph(dict)，运行没问题，但 pyright 会报"dict 不满足状态类型协议"。
class BizState(TypedDict):
    x: int


def addition(state: BizState) -> dict[str, int]:
    """加法节点：将 state 中的 x 加 1。"""
    print(f"加法节点收到的初始值:{state}")
    return {"x": state["x"] + 1}


def subtraction(state: BizState) -> dict[str, int]:
    """减法节点：将 state 中的 x 减 2。"""
    print(f"减法节点收到的初始值:{state}")
    return {"x": state["x"] - 2}


# 创建图纸：声明状态类型为 BizState（原教程用 StateGraph(dict) 也能跑，
# 但 pyright 检查不过——LangGraph 的类型协议要求状态是 TypedDict 这类"结构化"类型）
graph = StateGraph(BizState)
graph.add_node("addition", addition)
graph.add_node("subtraction", subtraction)

# 定义执行顺序：START → addition → subtraction → END
graph.add_edge(START, "addition")
graph.add_edge("addition", "subtraction")
graph.add_edge("subtraction", END)

# 查看图的边与节点（调试用）
print(graph.edges)
print(graph.nodes)

# 编译图构建器，得到可执行的图应用对象
app = graph.compile()
# invoke() 的核心输入是一整个状态字典，这里给 x 一个初始值 5
initial_state: BizState = {"x": 5}
# invoke 只接收一个核心参数：初始状态字典
result = app.invoke(initial_state)
print(f"最后的结果是:{result}")

# 打印图的可视化结构
print(app.get_graph().print_ascii())
print()
# 打印图的可视化结构，生成更加美观的Mermaid 代码，通过processon 编辑器查看
print(app.get_graph().draw_mermaid())

"""
【输出示例】
{('subtraction', '__end__'), ('addition', 'subtraction'), ('__start__', 'addition')}
{'addition': StateNodeSpec(runnable=addition(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class 'dict'>, retry_policy=None, cache_policy=None, ends=(), defer=False), 'subtraction': StateNodeSpec(runnable=subtraction(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class 'dict'>, retry_policy=None, cache_policy=None, ends=(), defer=False)}
加法节点收到的初始值:{'x': 5}
减法节点收到的初始值:{'x': 6}
最后的结果是:{'x': 4}
 +-----------+
 | __start__ |
 +-----------+
        *
        *
        *
  +----------+
  | addition |
  +----------+
        *
        *
        *
+-------------+
| subtraction |
+-------------+
        *
        *
        *
  +---------+
  | __end__ |
  +---------+
None

---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        addition(addition)
        subtraction(subtraction)
        __end__([<p>__end__</p>]):::last
        __start__ --> addition;
        addition --> subtraction;
        subtraction --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
"""
