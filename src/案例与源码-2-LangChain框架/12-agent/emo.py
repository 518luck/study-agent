from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class MyState(TypedDict):
    value: str


def node_a(state: MyState):
    return {"value": state["value"] + " A"}


def node_b(state: MyState):
    return {"value": state["value"] + " B"}


builder = StateGraph(MyState)
builder.add_node("node_a", node_a)
builder.add_node("node_b", node_b)
builder.add_edge(START, "node_a")
builder.add_edge("node_a", "node_b")
builder.add_edge("node_b", END)

app = builder.compile()
result = app.invoke({"value": "start"})
print(result)
