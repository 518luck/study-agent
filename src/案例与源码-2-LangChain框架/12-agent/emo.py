"""
【案例】LangGraph 最小示例：用「状态 + 节点 + 边」画出第一张可执行图

对应教程章节：第 22 章 - LangGraph 概述与快速入门（教程源码：案例与源码-3-LangGraph框架/01-helloworld）
运行方式（项目根目录）：uv run python -u "src/案例与源码-2-LangChain框架/12-agent/emo.py"

知识点速览（这个文件只有 20 多行，但把 LangGraph 的核心概念全用上了）：
- 状态（State）：MyState，图流转时携带的"数据包"（本质就是一个字典）。
- 节点（Node）：node_a / node_b，就是普通函数，约定是"吃进当前状态，返回要更新的字段"。
- 边（Edge）：add_edge 决定"这一步做完下一步去哪"；START / END 是图的入口和出口标记（虚拟的，不是真实节点）。
- 编译与运行：compile() 把画好的图变成可执行对象；invoke() 传入初始状态，跑完返回最终状态。
- 图可视化：app.get_graph() 取出图结构，可导出 ASCII / Mermaid 文本 / PNG 图片（见文件末尾）。

一句话：LangGraph 不负责"调模型"，它只负责"谁先谁后、状态怎么在步骤之间流动"；
调模型、执行工具这些具体动作，都写在各个节点函数里面，节点里想干什么就干什么。

【输出示例】（实测）
{'value': 'start AA BB'}

【和 Agent 的关系】
你在 AgentSmartSelectV1.0.py 里用 stream() 看到的「节点=model / 节点=tools」，
就是 create_agent 在内部搭的同一类图（model → tools → model 的循环，靠条件边决定要不要回头）。
本文件是"手工画图"的最小版本，节点里换成了拼字符串——结构完全相同，只是没有分支和循环。
"""

# TypedDict：给字典定义"字段表"的类型标注，作用约等于 TS 里的 interface。
# 注意它只是类型提示（给编辑器 / pyright 看的），运行时并不校验；图跑起来状态就是个普通 dict。
from typing import TypedDict

# StateGraph 是"图的图纸 / 建造器"：用它 add_node / add_edge 画出流程，最后 compile 成可执行对象。
# START / END 代表图的入口和出口，相当于流程图的"开始 / 结束"符号。
from langgraph.graph import END, START, StateGraph


# 定义状态：这张图流转时携带的数据形状。
# 这里只有一个字段 value（字符串）；运行时它就是一个 {"value": "..."} 这样的字典。
class MyState(TypedDict):
    value: str


# 节点 1：普通函数。约定是「收进当前状态，返回要更新的字段」——
# 不用返回完整状态，只返回发生变化的字段，LangGraph 会把它合并回去。
def node_a(state: MyState):
    return {"value": state["value"] + " AA"}


# 节点 2：同样的约定。
# 关键点：它拿到的 state 里，value 已经是 node_a 更新过的结果（"start AA"），
# 这就是"状态在节点之间流动"——后一个节点看到的是前一个节点的产物。
def node_b(state: MyState):
    return {"value": state["value"] + " BB"}


# 创建图纸：声明这张图使用 MyState 作为状态类型
builder = StateGraph(MyState)

# 注册节点：第一个参数是节点名（图内部用的标识，也是流式输出里看到的 "节点=xxx"），
# 第二个参数是实际执行的函数。
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)

# 连边：决定执行顺序，add_edge(从哪, 到哪)。这三行连起来就是：
#   START → node_a → node_b → END
# 注意 add_edge 是"无条件跳转"（顺序固定）；要做"看结果决定去哪"的分支，
# 用 add_conditional_edges（教程后续章节会展开）。
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

# 编译：把图纸变成可执行对象，之后才能运行
app = builder.compile()

# 运行：invoke 的参数是「初始状态」，返回值是「跑完后的完整状态」。
# 实测执行过程：
#   node_a 收到 {'value': 'start'}     → 更新为 'start AA'
#   node_b 收到 {'value': 'start AA'}  → 更新为 'start AA BB'
result = app.invoke({"value": "start"})

# 最终输出 {'value': 'start AA BB'}
print(result)
print()


# ===================== 图可视化：把画好的图"看"出来 =====================
# get_graph() 从编译好的图里取出"图结构"（有哪些节点、连了哪些边），三个画法都基于它。
# 任何 LangGraph 应用都能这么画，包括 create_agent 建出来的 Agent——把 V1.0 示例里的
# agent 拿来 get_graph()，就能看到 model / tools 节点和那两条循环的边。
graph = app.get_graph()

# 方式 1：ASCII 画法，直接在终端里画出方框和箭头，不需要任何额外依赖、不用联网
print("=== ASCII 图 ===")
print(graph.draw_ascii())
print()

# 方式 2：Mermaid 文本，把输出复制到 https://mermaid.live 就能渲染成好看的流程图
# 写笔记 / 教程文档想配图时用这个最省事
print("=== Mermaid 文本（可粘贴到 mermaid.live 渲染）===")
print(graph.draw_mermaid())
print()

# 方式 3：直接导出 PNG 图片（默认走 mermaid.ink 在线渲染，需要联网）
# output_file_path 指定保存路径（相对当前运行目录，即项目根目录）；不传则返回图片字节流。
# 如果不想每次运行都生成图片，把下面两行注释掉即可。
png_path = "graph.png"
graph.draw_mermaid_png(output_file_path=png_path)
print(f"PNG 已导出：{png_path}")

"""
【输出示例】（实测）
{'value': 'start AA BB'}

=== ASCII 图 ===
+-----------+
| __start__ |
+-----------+
      *
+--------+
| node_a |
+--------+
      *
+--------+
| node_b |
+--------+
      *
+---------+
| __end__ |
+---------+

=== Mermaid 文本（可粘贴到 mermaid.live 渲染）===
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_a(node_a)
	node_b(node_b)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_a;
	node_a --> node_b;
	node_b --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

PNG 已导出：graph.png

【三种可视化方式的选用建议】
- draw_ascii()：最轻量，跑脚本时随手看一眼结构，不装东西、不联网
- draw_mermaid()：想贴到 Markdown 笔记 / 教程文档里时用，粘到 mermaid.live 即可渲染
- draw_mermaid_png()：要生成图片文件（写文章配图）时用；默认调用 mermaid.ink 在线服务，
  需要联网；也可以传 draw_method="pyppeteer" 用本地浏览器渲染（需额外安装 pyppeteer）
"""
