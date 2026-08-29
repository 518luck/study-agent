"""
【案例】标准/工程化写法：用 LangChain 调用大模型（invoke + stream）

对应教程章节：第 10 章 - LangChain 快速上手与 HelloWorld → 6、实战：企业级封装与流式输出

本案例演示从零到一的完整工程化写法：
- 用 DeepSeek 官方 API（OpenAI 兼容接口）通过 LangChain 发问，掌握 invoke（一次性返回）与 stream（流式返回）两种调用方式。
- 将「初始化模型」封装成函数便于复用；用 .env 存密钥、logging 打日志、try/except 区分错误，符合正式项目习惯。
- 运行前在项目根目录 .env 里配置 DEEPSEEK_API_KEY（在 platform.deepseek.com 创建），执行：uv run python src/StandardDesc.py

补充说明：
- 为了让工程化示例更直观，这里继续使用很多同学在旧资料里更常见的 `ChatOpenAI` 写法；若你想看 1.x 统一入口，请对照同目录下的 `LangChainV1.0.py`。
- 当前脚本直连 DeepSeek 官方 API（https://api.deepseek.com，OpenAI 兼容端点），重点仍然是学习工程化写法，而不是限定某一个具体模型。
"""

# ========== 1. 导入与环境 ==========
# 下面每一行都是「把别人写好的功能拿进来」，后面才能用。

import os  # Python 自带：用来读「环境变量」（如 API 密钥）

# load_dotenv：从项目根目录的 .env 文件里，把变量加载到「环境」里，之后用 os.getenv("变量名") 就能读到。
# 把密钥写在 .env 里而不是代码里，既安全（不把密钥提交到 Git），又方便换环境（开发/生产用不同 .env）。
from dotenv import load_dotenv

# LangChainException：LangChain 在调用模型失败时会抛出的异常类型。
# 在 main() 里用 except LangChainException 单独接住这类错误，就能打出「模型调用失败」的日志，和配置错误、其他未知错误区分开。
from langchain_core.exceptions import LangChainException
from langchain_openai import (
    ChatOpenAI,
)  # 用 OpenAI 兼容接口和模型对话（DeepSeek 官方 API 也兼容这个接口）

# SecretStr：pydantic 的「保密字符串」类型，打印/记日志时只显示 **********，防止密钥意外泄露。
# ChatOpenAI 的 api_key 参数要求传入这个类型（普通 str 会被 IDE 标黄），所以下面包了一层。
from pydantic import SecretStr

# 真正执行「从 .env 加载到环境」；encoding='utf-8' 避免 .env 里有中文时乱码。
load_dotenv(encoding="utf-8")

# ----- 日志配置 -----
# logging 是 Python 自带的日志库，不用 pip 安装。用 logger.info() / logger.error() 代替 print，方便区分「普通信息」和「错误」，且可统一格式、写文件等。
# 通过环境变量 LOG_LEVEL 控制输出多少：开发时用 INFO（看得到调试信息），生产时在 .env 里设 LOG_LEVEL=WARNING，就只打警告和错误，减少刷屏。
import logging

_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _log_level, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)  # 当前模块的 logger，后面用 logger.info(...) 即可


# ========== 2. LLM 客户端初始化（封装为函数，便于多处复用） ==========
# 「LLM」= 大语言模型（如 DeepSeek、通义千问）。这里把「创建可对话的客户端」封装成一个函数，以后在别处也能直接调 init_llm_client()，不用重复写一长串配置。


def init_llm_client() -> ChatOpenAI:
    """
    初始化 LLM 客户端（封装成函数，提高复用性）。

    Returns:
        ChatOpenAI: 初始化好的「对话客户端」，可以对其调用 .invoke(问题) 或 .stream(问题)。
    """

    # 从环境变量里拿 API 密钥；没配置的话直接报错，提示去检查 .env。
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("环境变量 DEEPSEEK_API_KEY 未配置，请检查 .env 文件")

    # 创建客户端：指定用哪个模型、密钥、接口地址，以及「回复风格」相关参数。
    llm = ChatOpenAI(
        # 模型名称，与 DeepSeek 官方文档一致：deepseek-v4-flash（快、便宜）/ deepseek-v4-pro（更强、更贵）。
        # 旧名称 deepseek-chat / deepseek-reasoner 已于 2026-07-24 停用，不要再填。
        model="deepseek-v4-flash",
        api_key=SecretStr(
            api_key
        ),  # 包成「保密字符串」：既满足参数类型要求，打印模型对象时也不会泄露密钥
        base_url="https://api.deepseek.com",  # DeepSeek 官方地址，兼容 OpenAI 接口
        temperature=0.7,  # 控制「随机程度」：0 更确定、重复性高；1 更随机、更有创意。一般 0.5～0.8 即可。
        # 注意：新版 langchain-openai 已把 max_tokens 改名为 max_ completion _tokens（旧名已弃用，
        # IDE 会提示「没有名为 max_tokens 的参数」），含义不变：单次回复最多生成多少个 token，防止回复过长或超限。
        max_completion_tokens=2048,
    )
    return llm


# ========== 3. 主逻辑：invoke（一次性） + stream（流式）两种调用方式 ==========
# 这里把「问问题、拿回答、打日志」都放在 main() 里，并用 try/except 把可能出现的错误分开处理，避免程序一报错就崩掉、且能打出清晰错误信息。


def main():
    """主函数：封装核心逻辑，符合 Python 工程化规范。"""
    try:
        # 先拿到「可对话的客户端」
        llm = init_llm_client()
        logger.info("LLM客户端初始化成功")

        # ----- 方式一：invoke（一次性拿完整回复） -----
        # 发一个问题，程序会等模型全部答完，再一次性把 response 给你。适合短问答。
        question = "你是谁"
        response = llm.invoke(question)
        logger.info(f"问题：{question}")
        logger.info(f"回答：{response.content}")  # .content 里是模型的纯文字回复

        # ----- 方式二：stream（流式，边生成边输出） -----
        # 模型边想边返回，每次返回一小段（chunk），用 for 循环一段段打印，就像打字机效果。适合长文或需要「实时看到输出」的场景。
        print("==================== 以下是流式输出（另一种调用方式）")
        print("*" * 50)
        response_stream = llm.stream("介绍下 langchain，300字以内")
        for chunk in response_stream:
            print(chunk.content, end="")  # end="" 表示不换行，紧挨着打
        print()  # 流式结束后补一个换行，避免和后续输出粘在一起

    # ----- 异常处理：根据错误类型打不同日志，方便排查 -----
    # try 里面的代码一旦报错，会跳到下面某个 except；若都不匹配，再往上抛。
    except ValueError as e:
        # 例如：.env 里没配 DEEPSEEK_API_KEY，init_llm_client 里会 raise ValueError
        logger.error(f"配置错误：{e!s}")
    except LangChainException as e:
        # 例如：网络失败、API 限流、模型返回异常等，LangChain 会抛出 LangChainException
        logger.error(f"模型调用失败：{e!s}")
    except Exception as e:
        # 其他没预料到的错误都归到这里，避免程序静默崩溃
        logger.error(f"未知错误：{e!s}")


# ========== 脚本入口 ==========
# __name__ 是 Python 给每个模块自动设置的内置变量，表示「当前模块的名字」：
#   - 直接运行本文件时（如 python StandardDesc.py），Python 会把 __name__ 设为字符串 "__main__"（前后各两个下划线），
#     于是下面的条件为真，会执行 main()；
#   - 被别的文件 import 时，__name__ 是模块名（如 "01_helloworld.StandardDesc"），不等于 "__main__"，不会执行 main()，
#     避免一导入就自动跑一遍问问题。
#
# 注意：必须写 "__main__" 不能写成 "main"。Python 规定「主程序」的 __name__ 就是 "__main__"，
# 若写成 if __name__ == "main": 则条件永远为假（因为 __name__ 实际是 "__main__"），直接运行脚本时 main() 也不会被调用。
#
# 这里直接写 main() 即可，因为本文件的 main 是普通函数（def main），调用即执行。
# 若 main 是异步函数（async def main），则必须写 asyncio.run(main())，否则协程不会真正运行。
if __name__ == "__main__":
    main()

"""
【输出示例】
2026-08-28 21:41:40,715 - INFO - LLM客户端初始化成功
2026-08-28 21:41:40,928 - INFO - HTTP Request: POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"
2026-08-28 21:41:41,765 - INFO - 问题：你是谁
2026-08-28 21:41:41,765 - INFO - 回答：我是 DeepSeek，由深度求索公司创造的 AI 助手。我可以帮你解答问题、提供建议、处理信息等。有什么我可以帮你的吗？
==================== 以下是流式输出（另一种调用方式）
**************************************************
LangChain 是一个用于构建大语言模型（LLM）应用的开发框架，提供模块化组件和标准化接口。它支持模型调用、提示词管理、记忆存储、外部工具集成（如搜索、数据库、API）以及文档加载与向量检索，常用于开发聊天机器人、知识库问答、智能代理等场景。其核心特性包括链（Chain）组合逻辑、Agent 自主决策，以及丰富的生态集成。通过 LangChain，开发者可以快速将 LLM 与外部数据、工具连接，构建复杂工作流，同时支持 LangSmith 进行调试监控，LangServe 部署服务。整体上，它降低了 LLM 应用开发门槛，提升开发效率。
"""
