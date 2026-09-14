"""
【实验】Runtime Context 的六个关键事实 —— 配合同目录 RuntimeContextDemo.py 一起看

对应教程章节：第 24 章 - LangGraph API：节点、边与进阶 → 3.6 Runtime：配置与状态分开 / 3.7 Runtime 的基本用法

这里的图故意做得很小（只有一个计数器 + 一个日志列表），为的是把注意力全放在 context 上：
A. 同一张图，换 context 就换环境（模型/数据库/密钥），state 完全不用改 —— 这才是配置与状态分离的价值
B. context 永远不会混进 state，也不会出现在图的返回值里
C. 节点第二个参数「必须叫 runtime」：注入靠参数名，不是靠类型标注
D. 不传 context 时 runtime.context 是 None，读属性会 AttributeError
E. context 位置传 dict 会被自动转成 EnvContext 实例（_coerce_context 的兜底）
F. 带 checkpointer 时 context 不会被持久化：同一条 thread 再次 invoke 要重新传

运行：uv run python -u "src/案例与源码-3-LangGraph框架/04-node/05-edge/RuntimeContextDemo_实验.py"
"""

from dataclasses import dataclass
from typing import Annotated

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict


# ---------- 1. State：业务状态，会随图运行变化，节点之间互相流转 ----------
class CounterState(TypedDict):
    # Annotated 第二个参数是规约函数：节点返回 {"calls": 1} 时不是覆盖，而是累加（相当于 JS 里的 reducer）
    calls: Annotated[int, lambda x, y: x + y]
    log: Annotated[list, lambda x, y: x + y]


# ---------- 2. Context：运行配置，一次运行期间固定，节点只读 ----------
@dataclass  # 也可以写成 TypedDict 或 pydantic BaseModel，LangGraph 三种都支持
class EnvContext:
    model_name: str
    db_connection: str
    api_key: str


# ---------- 3. 节点：第二个参数 runtime 由 LangGraph 注入 ----------
def call_model(state: CounterState, runtime: Runtime[EnvContext]) -> dict:
    ctx = runtime.context  # 就是 invoke(..., context=...) 传进来的那个对象
    print(f"    [节点] 模型={ctx.model_name} 库={ctx.db_connection} 密钥={ctx.api_key[:5]}***")
    return {"calls": 1, "log": [f"用 {ctx.model_name} 处理了一次请求"]}


def build_graph(checkpointer=None):
    # 把 context_schema 挂在图上：声明「这张图需要什么配置」，运行时按声明校验/补全
    builder = StateGraph(CounterState, context_schema=EnvContext)
    builder.add_node("call_model", call_model)
    builder.add_edge(START, "call_model")
    builder.add_edge("call_model", END)
    return builder.compile(checkpointer=checkpointer)


def main():
    graph = build_graph()

    dev = EnvContext("deepseek-v4-flash", "postgresql://localhost/dev_db", "sk-dev-123456")
    prod = EnvContext("deepseek-v4-pro", "postgresql://prod-host/prod_db", "sk-prod-987654")

    # ---------- A. 换 context 就换环境 ----------
    print("A. 同一张图，换 context 就换环境")
    print("  dev 环境：")
    r_dev = graph.invoke({"calls": 0, "log": []}, context=dev)
    print("  prod 环境：")
    r_prod = graph.invoke({"calls": 0, "log": []}, context=prod)
    print(f"  两次 state 的字段完全一样 -> {list(r_dev) == list(r_prod) == ['calls', 'log']}")
    print(f"  prod 的日志：{r_prod['log']}")

    # ---------- B. context 不进 state ----------
    print("\nB. context 不进 state")
    leaked = [k for k in ("model_name", "db_connection", "api_key") if k in r_prod]
    print(f"  返回值里出现的配置字段：{leaked or '无'}")
    print(f"  图返回值的全部内容：{r_prod}")

    # ---------- C. 第二个参数必须叫 runtime ----------
    print("\nC. 第二个参数必须叫 runtime（靠参数名注入，类型标注只给 pyright 看）")

    def wrong_name_node(state: CounterState, ctx: EnvContext) -> dict:
        # 名字不叫 runtime，LangGraph 不会注入任何东西，调用时直接缺参数
        _ = ctx
        return {"calls": 1}

    g = StateGraph(CounterState, context_schema=EnvContext)
    g.add_node("wrong", wrong_name_node)  # type: ignore[arg-type]
    g.add_edge(START, "wrong")
    g.add_edge("wrong", END)
    try:
        g.compile().invoke({"calls": 0, "log": []}, context=dev)
        print("  没报错？那说明这个版本改变了注入规则")
    except TypeError as e:
        print(f"  参数名写成 ctx -> TypeError: {e}")

    # ---------- D. 不传 context ----------
    print("\nD. 不传 context")
    try:
        graph.invoke({"calls": 0, "log": []})
    except AttributeError as e:
        print(f"  runtime.context 是 None -> AttributeError: {e}")

    # ---------- E. context 传 dict 也能用 ----------
    print("\nE. context 传 dict（context_schema 是 dataclass 时会自动构造实例）")
    r = graph.invoke(
        {"calls": 0, "log": []},
        # 运行时允许传 dict，但类型签名要求 EnvContext，所以这里 pyright 会报错 -> 用 ignore 演示「运行时宽容、类型不宽容」
        context={"model_name": "m", "db_connection": "d", "api_key": "k-123456"},  # type: ignore[arg-type]
    )
    print(f"  结果正常 -> {r['log']}")

    # ---------- F. checkpointer 不保存 context ----------
    print("\nF. 带 checkpointer 时，context 不会被持久化")
    graph_with_ckpt = build_graph(checkpointer=InMemorySaver())

    # 加 RunnableConfig 类型注解：否则字面量被推断成 dict[str, dict[str, str]]，pyright 对不上 invoke 的参数类型
    cfg: RunnableConfig = {"configurable": {"thread_id": "thread-1"}}
    graph_with_ckpt.invoke({"calls": 0, "log": []}, context=dev, config=cfg)
    values = graph_with_ckpt.get_state(cfg).values
    print(f"  checkpoint 里保存的 state：{list(values)}")
    print(f"  里面能读到配置吗 -> {'model_name' in values}")
    print("  所以恢复同一条 thread 时，context 要重新传（它是「这次运行」的配置，不是业务数据）")

    # ---------- 附：context_schema 能导出 JSON Schema ----------
    print("\n附：ctx_schema 导出 JSON Schema（相当于 zod schema -> JSON Schema）")
    schema = graph.get_context_jsonschema()
    print(f"  必填字段：{schema['required'] if schema else None}")


if __name__ == "__main__":
    main()
