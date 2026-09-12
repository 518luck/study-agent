"""
【案例】本地 MCP 天气服务端（极简实现，无 FastMCP 依赖）

对应教程章节：第 20 章 - MCP 模型上下文协议 → 6、案例实战：本地 MCP 天气服务与客户端

知识点速览：
- 本案例是「教学版极简 MCP 服务端」：它重点演示 @mcp.tool() 背后的注册思想，让读者先理解
  “服务端负责暴露能力”这件事，再去看 FastMCP 的正式写法。
- 这里的 MCPWeatherServer 只模拟了“工具注册表 + 服务进程存活”两件事，并没有完整实现真实 MCP
  通信中的 JSON-RPC、握手、能力发现、标准传输层，所以它更适合拿来建立概念，不适合当成生产级 MCP 服务。
- 与第 17 章 Tool 的区别：Tool 更像单进程里的能力封装；MCP 则是在 Tool 之上增加一层标准协议，
  让同一套能力更容易被不同宿主、不同 AI 应用复用。
- 数据源使用和风天气（QWeather），请求实现与第 17 章 QueryWeatherTool.py 保持一致：
  和风只认经纬度不认城市名，因此工具内部要先调 GeoAPI 解析地点、再查实时天气——
  这种“一个工具内部发多次请求”对模型和 MCP 客户端都是完全透明的，是 MCP 封装能力的一个直观例子。
- 仓库里保留了 transport="sse" 这类写法，主要是为了和本章 mcp.json、网络化演示案例保持一致；
  如果从当前官方主线理解，初学者更应该先把 stdio 和 HTTP/Streamable HTTP 当作重点。
"""

import json
import os

import httpx
from dotenv import load_dotenv
from loguru import logger

load_dotenv(encoding="utf-8")

# 和风天气把「请求地址」和「凭据」拆成了两样东西，都需要在项目根目录 .env 中配置：
#   QWEATHER_API_HOST=你的API Host   # 控制台「设置」页分配，形如 abcxyz.qweatherapi.com，不带 https://
#   QWEATHER_API_KEY=你的API KEY     # 控制台「项目和凭据」页创建
API_HOST_ENV = "QWEATHER_API_HOST"
API_KEY_ENV = "QWEATHER_API_KEY"

# 旧版错误码的释义（官方文档：错误码 v1），用来把 code 翻译成人能看懂的原因
V1_CODE_HINTS = {
    "400": "请求参数错误或缺少必选参数",
    "401": "认证失败，通常是 KEY 不对、凭据类型不匹配，或 API Host 填错",
    "402": "额度不足或超出访问次数",
    "403": "无访问权限，或使用了错误的 API Host",
    "404": "查询的地区不存在",
    "429": "超过 QPM 限制（每分钟请求次数）",
    "500": "接口服务异常，可稍后重试",
}


def _qweather_get(path: str, params: dict) -> dict:
    """按和风天气的规则发一次 GET 请求，返回解析后的 JSON 字典；失败时抛出 RuntimeError。

    - 请求地址由「API Host + 接口路径」拼成，官方只支持 HTTPS；
    - 凭据放在 X-QW-Api-Key 请求头里（API KEY 认证方式）；
    - 接口返回都是 Gzip 压缩的，httpx 会自动解压，不需要手动处理。

    和风天气目前新老两版错误码并存，官方要求迁移期做兼容，所以两种都要判断：
    - 旧版（v1）：HTTP 状态码恒为 200，错误只藏在响应体的 code 字段，如 {"code": "401"}
    - 新版（v2）：HTTP 状态码与错误码一致，错误体形如 {"error": {"title": "Unauthorized", ...}}
    """
    api_host = os.getenv(API_HOST_ENV)
    api_key = os.getenv(API_KEY_ENV)
    if not api_host or not api_key:
        raise RuntimeError(
            f"缺少环境变量 {API_HOST_ENV} 或 {API_KEY_ENV}，请先在项目根目录 .env 中配置"
        )
    # API Host 一定是域名（形如 abcxyz.qweatherapi.com）。少了这个校验的话，
    # 填错值会以 DNS / SSL 握手失败的形式报出来，很难联想到是配置写错了——比如误填成凭据ID
    if "." not in api_host:
        raise RuntimeError(
            f"{API_HOST_ENV} 不像是一个域名：API Host 需要在控制台「设置」页查看，"
            f"形如 abcxyz.qweatherapi.com，注意不要填成凭据ID或API KEY"
        )

    response = httpx.get(
        f"https://{api_host}{path}",
        params=params,
        headers={"X-QW-Api-Key": api_key},  # 凭据走请求头，勿将 Key 写死在代码中
        timeout=30,  # 超时保护，避免长时间阻塞
    )

    try:
        data = response.json()
    except ValueError:
        data = {}  # 官方文档说明 404 / 405 不返回响应体

    # 新版错误码：HTTP 状态码本身就是错误码，细节在 error 字段里
    if response.status_code != 200:
        error = data.get("error") or {}
        detail = " ".join(filter(None, [error.get("title"), error.get("detail")]))
        raise RuntimeError(
            f"和风天气请求失败：HTTP {response.status_code} {detail}".strip()
        )

    # 旧版错误码：HTTP 是 200，错误只在 code 字段里；204 表示请求成功但该地区暂无数据
    code = data.get("code")
    if code is not None and str(code) not in ("200", "204"):
        hint = V1_CODE_HINTS.get(str(code), "未知错误")
        raise RuntimeError(f"和风天气返回错误码 code={code}：{hint}")

    return data


# ---------------------- 极简版 MCP 服务类（无 FastMCP 依赖，纯手写）----------------------
# 若使用 FastMCP，则不需要下面这一整段 class，直接 mcp = FastMCP("名") + @mcp.tool() + mcp.run() 即可，见 McpServerWeatherByFastMCP.py
class MCPWeatherServer:
    """极简版教学服务类：只保留“注册工具”和“维持进程”两层概念。"""

    def __init__(self, name: str, host: str, port: int):
        # 保留原实例化参数，与原代码配置对齐
        self.name = name
        self.host = host
        self.port = port
        # 存储已注册的工具函数；本仓库里的同进程客户端会直接读取这个注册表做教学演示
        self._tools = {}

    def tool(self):
        """实现 @mcp.tool() 装饰器：把普通函数登记到工具注册表中。"""

        def decorator(func):
            self._tools[func.__name__] = func  # 注册工具函数，key 为函数名
            return func

        return decorator

    def run(self, transport: str):
        """模拟 run() 入口；这里只打印监听信息并保持进程存活，不提供完整网络服务。"""
        if transport != "sse":
            logger.warning(f"不支持的传输协议 {transport}，默认使用 SSE")
        logger.info(f"启动 MCP SSE 天气服务器，监听 http://{self.host}:{self.port}/sse")
        self._keep_alive()

    def _keep_alive(self):
        """简单保持进程运行，便于从日志层面观察“服务端已启动”的状态。"""
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("MCP 天气服务器已停止")


# ---------------------- 创建 MCP 实例并注册工具 ----------------------
# 对应教程：MCP 架构中的「MCP 服务器」角色，为客户端提供可暴露的能力
# 若改用 FastMCP：构造函数只接受服务名，不能写 FastMCP(..., host=..., port=...)；
# host/port 在 run() 时传，如 mcp.run(transport="sse", host="127.0.0.1", port=8000)。参见 McpServerWeatherByFastMCP.py
mcp = MCPWeatherServer("WeatherServerSSE", host="127.0.0.1", port=8000)


# @mcp.tool() 将 get_weather 注册为 MCP 工具；教学版客户端会直接从注册表里取出它。
# 注册之后，函数的 docstring 就是模型看到的「工具说明」，所以参数规则要写清楚
@mcp.tool()
def get_weather(loc: str) -> str:
    """
    查询指定城市的实时天气。

    参数:
        loc: 查询地点，以下三种写法都支持：城市名（如 北京、Shanghai）、
             和风天气 LocationID（如 101010100）、或「经度,纬度」坐标（如 116.41,39.92）。
             遇到同名地点时（如「朝阳」），建议把城市名写得更完整，或改用 LocationID / 坐标。

    返回:
        和风天气实时天气接口返回的 JSON 字符串，包含天气现象、气温、体感温度、湿度、
        风向风速、降水、气压、能见度、露点、云量、紫外线指数等信息，以及实际命中的地点。
        注意 humidity 和 cloudCover 是 0~1 的小数（不是百分比），uvIndex 取值 0~15。
    """
    try:
        # Step 1. 城市名 → 经纬度：和风天气的天气接口只认坐标，先用 GeoAPI 做一次地点解析
        geo = _qweather_get("/geo/v2/city/lookup", {"location": loc, "lang": "zh"})

        # 同名地点可能匹配到多条，接口已按匹配度（rank）排序，这里取第一条
        places = geo.get("location") or []
        if not places:
            return json.dumps(
                {
                    "error": f"未找到地点「{loc}」，可换用更完整的城市名，"
                    f"或直接传坐标，如 116.41,39.92"
                },
                ensure_ascii=False,
            )
        place = places[0]

        # 天气接口要求经纬度最多两位小数，GeoAPI 返回的是五位数，先四舍五入再拼进 URL
        lat = round(float(place["lat"]), 2)
        lon = round(float(place["lon"]), 2)

        # Step 2. 拿坐标查实时天气（实时天气 v1，1 公里分辨率，全球覆盖）
        data = _qweather_get(f"/weather/v1/current/{lat}/{lon}", {"lang": "zh"})
    except RuntimeError as exc:
        # 凭据错误、额度不足、请求超限等统一转成 JSON 错误说明返回给调用方，
        # 让模型能把原因转述给用户，而不是让整条调用链直接抛异常中断
        logger.warning(f"查询 {loc} 天气失败：{exc}")
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # Step 3. 把解析到的城市信息一并返回：同名城市不止一个，模型需要知道实际查的是哪一个
    data["location"] = {
        "name": place["name"],
        "adm1": place["adm1"],
        "id": place["id"],
        "lat": lat,
        "lon": lon,
    }

    # 用 .get 兜底：code=204（请求成功但该地区暂无数据）时响应里没有 condition 字段
    condition = (data.get("condition") or {}).get("text", "暂无数据")
    logger.info(f"查询 {place['name']} 天气成功，天气现象：{condition}")

    # Step 4. 序列化为 JSON 字符串返回，供后续链继续处理；ensure_ascii=False 让中文保持可读
    return json.dumps(data, ensure_ascii=False)


if __name__ == "__main__":
    logger.info("启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse")
    mcp.run(transport="sse")

"""
【输出示例】服务端启动后日志（实测）
2026-09-12 21:22:46.102 | INFO     | __main__:<module>:209 - 启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse
2026-09-12 21:22:46.102 | INFO     | __main__:run:127 - 启动 MCP SSE 天气服务器，监听 http://127.0.0.1:8000/sse

【本地验证工具函数】服务端启动后会一直阻塞在保活循环里，所以单独验证 get_weather 本身即可：
$ uv run python -c "import sys; sys.path.insert(0, 'src/案例与源码-2-LangChain框架/11-mcp'); import McpServer; print(McpServer.get_weather('北京'))"
2026-09-12 21:22:45.983 | INFO     | McpServer:get_weather:202 - 查询 北京 天气成功，天气现象：晴
{"metadata": {"tag": "..."}, "condition": {"text": "晴", "code": "100"}, "temperature": {"value": 21.77, "unit": "°C"}, ..., "location": {"name": "北京", "adm1": "北京市", "id": "101010100", "lat": 39.9, "lon": 116.41}}

【查不到地点时】不抛异常，返回错误说明，便于模型转述给用户：
{"error": "和风天气请求失败：HTTP 400 No Such Location Cannot find the location of the query, please try another location."}
"""
