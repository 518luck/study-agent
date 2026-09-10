"""
【基础】看懂 LangChain 代码的 3 个 Python 前置知识

对应教程章节：贯穿全书的基础语法（不属于某一章）

知识点速览：
一、类与实例：`ClassName(...)` 就等于 JS 的 `new ClassName(...)`，Python 没有 new 关键字。
二、函数参数的四种形态：位置参数 / 关键字参数 / 默认值 / **kwargs。
   —— 这是 JS 里没有的机制，也是看 LangChain 代码最容易卡住的地方。
三、lambda：没有名字的短函数，等价 JS 的箭头函数。

看完这个文件，再回头看 `RunnableWithMessageHistory(chain, get_session_history=..., ...)`
那四行，就只是"造一个对象，给它四个零件"而已。
"""

# ---------- 1. 类与实例 ----------
# JS 写法：
#   class History { constructor() { this.messages = []; } }
#   const h = new History();


class History:
    def __init__(self):  # ≈ constructor（构造时自动执行）
        self.messages = []  # ≈ this.messages

    def add(self, text):  # self 要显式写在第一个参数；JS 的 this 是隐式的
        self.messages.append(text)


h = History()  # ≈ new History()：造一个实例
h.add("我叫张三")
print("1) 实例里的数据:", h.messages)

h2 = History()  # 再调一次类名，就是又造一个新对象，两者互不影响
print("   另一个实例:", h2.messages)
print("   所以：一个实例 = 一段独立的对话历史")


# ---------- 2. 函数参数的四种形态（重点） ----------
def demo(a, b, c="默认值", **rest):
    print(f"   a={a}  b={b}  c={c}  rest={rest}")


print("2) 四种传参方式：")
demo(1, 2)  # ① 位置参数：按顺序对应，和 JS 一样
demo(b=2, a=1)  # ② 关键字参数：写「名字=值」，顺序随意 —— JS 没有这个，只能传对象
demo(1, 2, c="传入的")  # ③ 默认值：不传就用默认值
demo(1, 4, "第三个", d=44)  # ④ **rest：多余的键值被打包成一个 dict（≈ JS 的 {...rest}）
demo(1, 2, c="abc", any_name="随便写", whatever=[1, 2])
print()


# 所以 LangChain 里那四行长这样：
#     RunnableWithMessageHistory(
#         chain,                        ← 位置参数：没写名字，按顺序坐进第一个位置
#         get_session_history=...,      ← 关键字参数：靠「名字」对上
#         input_messages_key="input",   ← 关键字参数
#     )
# 两者的区别只是「按顺序」还是「按名字」，和你在脚本里用的变量名叫什么无关。


# ---------- 3. lambda：没有名字的函数 ----------
# JS:  const double = (x) => x * 2;
double = lambda x: x * 2  # 只能写一个表达式，多行逻辑要用 def
print("3) lambda 直接调用:", double(21))


def call_it(fn, value):  # 函数也能像普通值一样当参数传（JS 里很常见）
    return fn(value)


print("   lambda 当参数传:", call_it(lambda x: x + 1, 41))
print("   get_session_history=lambda session_id: history 就是把它当参数传给对方")

# 多行逻辑的等价写法（lambda 写不下时用 def）：
#     def double(x):
#         return x * 2

"""
【输出示例】
1) 实例里的数据: ['我叫张三']
   另一个实例: []
   所以：一个实例 = 一段独立的对话历史
2) 四种传参方式：
   a=1  b=2  c=默认值  rest={}
   a=1  b=2  c=默认值  rest={}
   a=1  b=2  c=传入的  rest={}
   a=1  b=4  c=第三个  rest={'d': 44}
   a=1  b=2  c=abc  rest={'any_name': '随便写', 'whatever': [1, 2]}

3) lambda 直接调用: 42
   lambda 当参数传: 42
   get_session_history=lambda session_id: history 就是把它当参数传给对方
"""
