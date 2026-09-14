"""
【案例】节点重试策略（RetryPolicy）：默认重试、自定义 retry_on 仅对特定异常重试、以及「不可重试异常」直接失败，演示 add_node(..., retry_policy=RetryPolicy(...)) 的用法。

对应教程章节：第 24 章 - LangGraph API：节点、边与进阶 → 1、Graph API 之 Node（节点）

知识点速览：
- RetryPolicy 不只是“重试几次”，而是两层策略组合：一层是时间策略（重试次数、间隔、退避），一层是异常策略（哪些错误值得重试）。
- RetryPolicy(max_attempts=5) 适合先观察默认行为；RetryPolicy(..., retry_on=custom_retry_on) 则更贴近真实项目里的精细控制。
- 本例最值得关注的是：不是所有异常都应该重试，像 ValueError 这类逻辑/参数错误通常应直接失败。
- 重试的粒度是「整个节点函数重新跑一遍」（输入状态相同），不是从断点续跑；所以节点要幂等，否则副作用会重复发生。
- 失败的那次尝试不会留下状态写入（状态只在 return 时写入），但 print、调用外部 API 这类副作用会真实重复。
- max_attempts 含首次调用（max_attempts=5 → 1 次首次 + 最多 4 次重试）；重试等待 = initial_interval * backoff_factor ** (重试序号 - 1)，默认再叠加 0~1 秒 jitter。
- 不传 retry_policy 的节点默认不重试，异常直接抛给 invoke 的调用方。
"""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict


# 定义状态类型
class DiliState(TypedDict):
    result: str  # 只有一个键：节点成功时把结果写进来


# 全局计数器：记录API尝试次数
# 说明：函数内「重新赋值」模块级变量必须写 global（只读取、或改内容如 list.append 都不需要）。
# 这是 Python 与 JS 的一个差异——JS 里外层 let 直接赋值即可，Python 要显式声明。
attempt_counter = 0


# 工具函数：把「单节点图」的建图样板收起来，三次测试只换节点函数和重试策略
# 返回 compile() 后的可执行图（CompiledStateGraph），可以直接 invoke
def build_retry_graph(node_name: str, node_func, retry_policy: RetryPolicy):
    builder = StateGraph(DiliState)
    # 为节点添加重试策略，需要在add_node中设置retry_policy参数。
    # retry_policy参数接受一个RetryPolicy命名元组对象。
    # 默认情况下，retry_on 参数使用 default_retry_on 函数：它重试「网络/外部抖动」类异常，
    # 但明确排除一批「代码写错」类异常（ValueError、TypeError、RuntimeError、OSError 等）。
    # 准确说法是：除那批代码 bug 之外的异常都会重试，而不是「任何异常都重试」。
    builder.add_node(
        node_name, node_func, retry_policy=retry_policy
    )  # 重试是节点级配置
    builder.add_edge(START, node_name)  # 入口：START → 该节点
    builder.add_edge(node_name, END)  # 终点：该节点 → END（单节点图，一条直线）
    return builder.compile()


# 模拟不稳定的API调用，使用全局变量跟踪尝试次数
# 注意：节点被重试时会完整重跑，所以「打印 + 计数器递增」这类副作用会真实发生多次；
#      真实项目里写数据库、发消息等操作必须做幂等设计，否则重试会造成重复
def unstable_api_call(state: DiliState) -> dict[str, Any]:
    """模拟不稳定API：前2次失败，第3次成功（全局计数器记录尝试次数）"""
    global attempt_counter  # 需要重新赋值，所以必须声明 global
    attempt_counter += 1
    # 纯文本打印尝试次数
    print(f"尝试调用API，这是第 {attempt_counter} 次尝试")

    # 模拟失败/成功逻辑：前2次抛异常，第3次返回结果
    # 抛异常后，langgraph 先问 retry_on「该不该重试」，再问 max_attempts「还有没有次数」
    if attempt_counter < 3:
        raise Exception(f"模拟API调用失败abcd (尝试 {attempt_counter})")
    # 只有 return 才会写入状态；抛异常的那几次不会留下任何状态改动
    return {"result": f"API调用成功，经过 {attempt_counter} 次尝试"}


# 自定义重试条件判断函数：这是 retry_on 的第二种形态（谓词函数）
#   形态一：retry_on=[Timeout, ConnectionError]  ← 异常类列表（DefNode.py 用的就是这种）
#   形态二：retry_on=custom_retry_on            ← 收 Exception、返回 bool 的判断函数（本例）
# langgraph 在节点每次抛异常后都会调用它一次，返回 True 才进入重试流程
def custom_retry_on(exception: Exception) -> bool:
    """自定义重试规则：只对包含「模拟API调用失败」的异常重试"""
    print("########################:  " + str(exception))
    err_msg = str(exception)
    if "模拟API调用失败" in err_msg:
        print(f"捕获到可重试异常: {err_msg}")
        return True
    print(f"捕获到不可重试异常: {err_msg}")
    return False


# 模拟抛出 ValueError 的节点
# ValueError 属于 default_retry_on 明确排除的「代码 bug 类」异常 → 一次都不重试，直接失败
def value_error_call(state: DiliState) -> dict[str, Any]:
    """模拟抛出ValueError：默认重试策略对这类异常不重试"""
    print("调用会抛出 ValueError 的节点")
    raise ValueError("模拟 ValueError 异常")  # 参数/数据写错应立即暴露，重试无意义


# 测试方法1：默认重试策略
def test_default_retry():
    global attempt_counter
    print("1. 使用默认重试策略:")
    print("   默认策略会对除特定异常外的所有异常进行重试")
    print("   不会重试的异常包括: ValueError, TypeError, ArithmeticError, ImportError,")
    print("                     LookupError, NameError, SyntaxError, RuntimeError,")
    print(
        "                     ReferenceError, StopIteration, StopAsyncIteration, OSError\n"
    )
    # ^ 这份名单就是 default_retry_on 里明确返回 False 的那批；不在名单里的异常（含普通 Exception）都会重试

    print("测试默认重试策略:")
    attempt_counter = 0  # 重置计数器：重试会让节点跑多次，每次测试前都要归零
    default_graph = build_retry_graph(
        node_name="unstable_api",
        node_func=unstable_api_call,
        retry_policy=RetryPolicy(max_attempts=5),  # 最多5次尝试（含首次），足够重试成功
        # 未指定的字段走默认值：initial_interval=0.5、backoff_factor=2、jitter=True
        # → 第一次重试前等 0.5 秒 + 抖动，第二次等 1.0 秒 + 抖动
    )
    try:
        result = default_graph.invoke({"result": ""})
        print(
            f"最终结果: {result}\n"
        )  # 第 3 次尝试成功，返回 {'result': 'API调用成功，经过 3 次尝试'}
    except Exception as e:
        # 次数耗尽或异常不可重试时，异常会传播到这里；类型不会被包装（仍是原始异常类）
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 测试方法2：自定义重试策略（输出完全匹配要求）
def test_custom_retry():
    global attempt_counter
    print("2. 使用自定义重试策略:")
    print("   自定义策略只对特定错误进行重试\n")
    print("测试自定义重试策略:")
    attempt_counter = 0  # 重置计数器
    custom_graph = build_retry_graph(
        node_name="custom_retry_api",
        node_func=unstable_api_call,
        # 换成谓词函数后，重试与否由异常消息决定（含「模拟API调用失败」才重试）
        # 这一句让每次异常都多出 custom_retry_on 的两行打印，可与测试 1 对照观察
        retry_policy=RetryPolicy(max_attempts=5, retry_on=custom_retry_on),
    )
    try:
        result = custom_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 测试方法3：不可重试异常演示,测试 ValueError（默认策略不会重试）
def test_no_retry_exception():
    print("3. 测试不会重试的异常类型:")
    print("测试 ValueError（默认策略不会重试）:")
    # 这里的 max_attempts=3 其实用不上：ValueError 一次就失败，根本没有第二次尝试
    no_retry_graph = build_retry_graph(
        node_name="value_error_node",
        node_func=value_error_call,
        retry_policy=RetryPolicy(max_attempts=3),
    )
    try:
        result = no_retry_graph.invoke({"result": ""})
        print(f"最终结果: {result}\n")
    except Exception as e:
        print(f"最终失败: {type(e).__name__}: {e}\n")


# 主演示函数
def run_demo():
    print("=== LangGraph 节点重试策略完整演示===")
    print("-" * 80 + "\n")
    # 前两个测试默认注释掉，所以本文件默认只演示第 3 项（与文末的输出示例一致）。
    # 取消下面两行的注释即可看到完整演示：测试1 走默认策略重试 2 次后成功；测试2 换成自定义
    # 谓词后同样成功，但过程中多出 custom_retry_on 的打印。三个测试全开实测约 5 秒
    # （等待时间来自重试退避 0.5 + 1.0 秒及 jitter，不是代码慢）。
    # test_default_retry()
    # test_custom_retry()
    test_no_retry_exception()
    print("-" * 80)
    print("=== 演示结束 ===")


# 程序入口
if __name__ == "__main__":
    run_demo()


"""
【输出示例】
（注：前两个测试默认被注释，所以下面只有第 3 项的输出）
=== LangGraph 节点重试策略完整演示===
--------------------------------------------------------------------------------

3. 测试不会重试的异常类型:
测试 ValueError（默认策略不会重试）:
调用会抛出 ValueError 的节点
最终失败: ValueError: 模拟 ValueError 异常

--------------------------------------------------------------------------------
=== 演示结束 ===
"""
