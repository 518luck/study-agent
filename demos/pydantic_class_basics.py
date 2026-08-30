"""
【Demo】Pydantic class 语法拆解：从普通 class 到教程里的写法
对应问题：class Person(BaseModel) 的 time: str = Field(description=...) 是什么意思

运行（study-agent 目录下）：
    .venv/bin/python demos/pydantic_class_basics.py
"""

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel

console = Console()

# ========== 第 1 步：普通 class（无校验） ==========
console.rule("[bold cyan]第 1 步：普通 class + __init__")


class PersonOld:
    def __init__(self, time, person, event):   # 相当于 JS 的 constructor
        self.time = time       # self ≈ JS 的 this
        self.person = person
        self.event = event


p1 = PersonOld(time=123, person="亮仔", event="发布会")   # time 故意传数字
rprint(f"普通 class：time 传了数字 123 也照单全收 → p1.time = {p1.time!r}")
rprint("[dim]（类型注解只是注释，没有任何强制力）[/dim]")

# ========== 第 2 步：Pydantic class（声明式字段 + 自动校验） ==========
console.rule("[bold cyan]第 2 步：Pydantic —— 声明字段，自动生成 __init__ 和校验")
from pydantic import BaseModel


class Person(BaseModel):     # (BaseModel) = 继承，自动获得校验能力
    time: str                # 声明：字段名 : 类型
    person: str
    event: str


p2 = Person(time="昨天", person="亮仔", event="发布会")   # 不用写 __init__ 就能传参
rprint(f"正常创建：p2.person = {p2.person!r}")

# 故意传错类型，看校验触发
try:
    Person(time=123, person="亮仔", event="发布会")   # time 传了数字
except Exception as e:
    console.print(Panel(
        f"{type(e).__name__}:\n{e}",
        title="[red]类型传错 → 当场报错[/red]",
        border_style="red",
    ))

# ========== 第 3 步：Field(description=...) —— 字段说明书 ==========
console.rule("[bold cyan]第 3 步：Field(description=...) 的用途")
from pydantic import Field


class News(BaseModel):
    """定义一条「新闻」的结构：时间、人物、事件。（教程原代码）"""

    time: str = Field(description="时间")
    person: str = Field(description="人物")
    event: str = Field(description="事件")


n = News(time="昨天下午", person="亮仔", event="发布了新产品")
rprint(f"实例化：n.person = {n.person!r}")

# 关键实验：把这份 class 转成 JSON Schema —— 这就是"发给 AI 的字段说明"
import json

schema = News.model_json_schema()
console.print(Panel("News.model_json_schema() —— class 变成 JSON Schema"))
rprint(schema)
console.print(
    "[bold]看到了吗？[/bold]你在 Field(description=...) 里写的"
    "「时间」「人物」「事件」[bold]原样出现在 schema 的 description 里[/bold]"
    " —— 这份 schema 就是告诉 AI 每个字段该填什么的说明书。"
)
