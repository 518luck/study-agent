"""
【案例】节点定义方式与可选参数：普通节点、带额外参数的节点（用 partial 绑定）、以及 add_node 时传入 RetryPolicy 配置重试策略。

对应教程章节：第 24 章 - LangGraph API：节点、边与进阶 → 1、Graph API 之 Node（节点）

知识点速览：
- Node 本质上是被图调度的 Python 函数；本例重点不是业务逻辑，而是理解“节点如何被 add_node 注册进图”。
- 节点常见返回值是对 State 的局部更新 dict，而不是整份完整状态；若节点需要额外参数，可用 functools.partial 预先绑定，再传给 add_node。
- add_node(name, node_func, retry_policy=RetryPolicy(...)) 说明节点除了函数本身，还能挂执行策略；本例顺手演示了 retry_policy 的挂法。
- 节点函数的 state 参数要注解成图的 State 类型（GraphState），不要写成裸 dict：add_node 会按 StateNode 协议做类型检查，注解不一致时 pyright 会报 partial 无法赋值给 action。
- 初始状态同理，需满足 GraphState 声明的形状；这层校验只发生在类型检查期（pyright），运行时传入错误类型不会报错。
"""

from functools import partial
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from requests import RequestException, Timeout


class GraphState(TypedDict):
    process_data: dict


def input_node(state: GraphState) -> dict:
    print(f"input_node 收到的初始值:{state}")
    return {"process_data": {"input": "input_value"}}


# 节点可带额外参数，用 partial 绑定后传给 add_node
# 注意：state 的注解要和图的 State 保持一致，写成裸 dict 会让 partial 无法通过 add_node 的类型检查
def process_node(state: GraphState, param1: int, param2: str) -> dict:
    print(state, param1, param2)
    return {"process_data": {"process": "process_value"}}


# 重试策略：仅对 RequestException、Timeout 重试，最多尝试 3 次（含首次，即最多重试 2 次）
# 等待间隔公式：initial_interval * backoff_factor ** (已重试次数)，再用 max_interval 截断上限（本例未设，用默认 128 秒）
retry_policy = RetryPolicy(
    max_attempts=3,  # 最多尝试次数，含首次调用；默认 3
    initial_interval=1,  # 第一次重试前的等待秒数；默认 0.5
    jitter=True,  # 在等待时间上加 0~1 秒随机抖动，避免多个任务同时重试打成尖峰；默认 True
    backoff_factor=2,  # 每重试一次等待时间乘该系数，即 1s → 2s → 4s…；默认 2.0
    retry_on=[
        RequestException,
        Timeout,
    ],  # 只有这些异常触发重试，其余异常直接抛出；默认只重试网络类错误（default_retry_on）
)

stateGraph = StateGraph(GraphState)
stateGraph.add_node("input", input_node)
process_with_params = partial(process_node, param1=100, param2="test")
stateGraph.add_node("process", process_with_params, retry_policy=retry_policy)
stateGraph.add_edge(START, "input")
stateGraph.add_edge("input", "process")
stateGraph.add_edge("process", END)

graph = stateGraph.compile()

print(stateGraph.edges)  # edges 是 set，打印顺序每次运行都可能不同
print(stateGraph.nodes)
print(graph.get_graph().print_ascii())
print()

# 初始状态必须是符合 GraphState 声明的 dict（注解给了 pyright 检查的依据，运行时仍不做校验）
initial_state: GraphState = {"process_data": {"seed": "initial_value"}}
result = graph.invoke(initial_state)
print(f"最后的结果是:{result}")

"""
【输出示例】
{('process', '__end__'), ('__start__', 'input'), ('input', 'process')}
{'input': StateNodeSpec(runnable=input(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class '__main__.GraphState'>, retry_policy=None, cache_policy=None, is_error_handler=False, error_handler_node=None, ends=(), defer=False, timeout=None, trace_policy=None), 'process': StateNodeSpec(runnable=process(tags=None, recurse=True, explode_args=False, func_accepts={}), metadata=None, input_schema=<class '__main__.GraphState'>, retry_policy=RetryPolicy(initial_interval=1, backoff_factor=2, max_interval=128.0, max_attempts=3, jitter=True, retry_on=[<class 'requests.exceptions.RequestException'>, <class 'requests.exceptions.Timeout'>]), cache_policy=None, is_error_handler=False, error_handler_node=None, ends=(), defer=False, timeout=None, trace_policy=None)}
+-----------+
| __start__ |
+-----------+
      *
      *
      *
  +-------+
  | input |
  +-------+
      *
      *
      *
 +---------+
 | process |
 +---------+
      *
      *
      *
 +---------+
 | __end__ |
 +---------+
None

input_node 收到的初始值:{'process_data': {'seed': 'initial_value'}}
{'process_data': {'input': 'input_value'}} 100 test
最后的结果是:{'process_data': {'process': 'process_value'}}
"""
