"""
【案例】Redis 环境校验：确认 redis 包已安装，且 Redis 服务可连通

对应教程章节：第 16 章 - 记忆与对话历史 → 6、案例代码 → 6.2 持久化：Redis 存储 → 环境验证

知识点速览：
- 使用 RedisChatMessageHistory 前，至少要先确认两件事：Python 能否导入 redis 包、当前 REDIS_URL 指向的 Redis 服务是否真的可达。
- 默认按原生 Redis 地址 redis://localhost:6379 检查；如果你用的是 Redis Stack 的 Docker 映射端口（如 -p 26379:6379），可先设置环境变量 REDIS_URL=redis://localhost:26379。
- 这个脚本不依赖 LangChain，本质上是在排查“Python 依赖”和“Redis 服务”这两个基础环境问题。
"""

import os

try:
    import redis
except ModuleNotFoundError:
    print("❌ 未找到 redis 包，请先执行：pip install -r requirements.txt")
    raise SystemExit(1)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

print("✅ redis 包导入成功！")
print(f"✅ redis 包版本：{redis.__version__}")
print(f"正在连接 Redis：{REDIS_URL}")

client = None
try:
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    print(f"✅ Redis 连接成功，PING -> {client.ping()}")
except redis.RedisError as e:
    # RedisError 是 redis 所有异常的基类（已涵盖 ConnectionError / TimeoutError / ResponseError 等）
    print("❌ Redis 连接失败")
    print(f"   REDIS_URL = {REDIS_URL}")
    print(f"   错误信息 = {e}")
    print("   如果你使用的是 Redis Stack 的 Docker 端口映射，可尝试：")
    print("   export REDIS_URL=redis://localhost:26379")
    raise SystemExit(1)
except ValueError as e:
    # URL 写法有问题（如写成 "garbage"、端口越界）时抛的是 ValueError，不是 RedisError
    print(f"❌ REDIS_URL 格式不正确：{e}")
    raise SystemExit(1)
finally:
    if client is not None:
        client.close()

"""
【输出示例】
✅ redis 包导入成功！
✅ redis 包版本：5.3.1
正在连接 Redis：redis://localhost:6379
✅ Redis 连接成功，PING -> True
"""
