"""
【案例】天气查询工具：用 @tool 定义可被 LLM 调用的天气接口，请求和风天气 API 并返回 JSON

对应教程章节：第 17 章 - Tools 工具调用 → 5、天气助手实战：把 Tool 跑成业务闭环 → 5.2 定义天气查询工具

知识点速览：
- 本例沿用教程 5.2 的天气助手案例，但数据源换成了和风天气（QWeather）：接口地址、认证方式、
  返回结构都变了，而「把第三方 HTTP API 封装成 Tool」这条主线不变。
- 工具 docstring 要尽量写清调用场景和关键参数规则；像 `loc` 既能传城市名又能传坐标这种约束，
  最好直接写在工具说明里，而不是留给模型猜。
- 和风天气的天气接口只接收经纬度、不认城市名，所以工具内部是两步调用：先调 GeoAPI 把城市名
  解析成坐标，再用坐标查实时天气。这种“工具内部发多次请求”对模型是完全透明的。
- 认证用 API KEY，放在 `X-QW-Api-Key` 请求头；官方推荐的是 JWT（`Authorization: Bearer <token>`），
  并宣布从 2027 年 2 月 1 日起逐步限制 API KEY 的每日请求数量。JWT 需要额外生成 Ed25519 密钥对
  并上传公钥，本例从简，先用 API KEY 跑通。
- 返回值这里使用 JSON 字符串，是为了方便后续链路继续处理；真实项目里也可以返回更稳定的结构化对象，
  但要注意和后续消费方式保持一致。
"""

import json
import os

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool

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


# @tool 装饰器：函数名 get_weather 即工具名，下方 docstring 会成为模型理解工具的重要依据
@tool
def get_weather(loc: str) -> str:
    """
    查询指定城市的实时天气。

    参数:
        loc: 查询地点，以下三种写法都支持：城市名（如 北京、Shanghai）、
             和风天气 LocationID（如 101010100）、或「经度,纬度」坐标（如 116.41,39.92）。
             遇到同名地点时（如「朝阳」），建议把城市名写得更完整，或改用 LocationID / 坐标。

    返回:
        和风天气实时天气接口返回的 JSON 字符串，包含天气现象、气温、体感温度、湿度、
        风向风速、降水、气压、能见度、露点、云量、紫外线指数等信息。
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
        # 凭据错误、额度不足、请求超限等统一转成 JSON 错误说明返回给模型，
        # 让模型能把原因转述给用户，而不是让整条调用链直接抛异常中断
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # Step 3. 把解析到的城市信息一并返回：同名城市不止一个，模型需要知道实际查的是哪一个
    data["location"] = {
        "name": place["name"],
        "adm1": place["adm1"],
        "id": place["id"],
        "lat": lat,
        "lon": lon,
    }

    # Step 4. 序列化为 JSON 字符串返回，供后续链继续处理；ensure_ascii=False 让中文保持可读
    return json.dumps(data, ensure_ascii=False)


# 本地测试：单参数工具可直接传值；若和更通用的工具调用风格保持一致，也可传 {"loc": "..."}
# 运行前需在项目根目录 .env 中配好 QWEATHER_API_HOST 与 QWEATHER_API_KEY
#
# 必须加 __main__ 守卫：本模块会被 LLMQueryWeatherDemo.py 导入，而 Python 导入模块时
# 会执行模块的全部顶层代码。不加守卫的话，别人一 import 就会白白发一次天气请求，
# 还会在真正的逻辑开始前多打印一大段 JSON。
if __name__ == "__main__":
    result = get_weather.invoke("北京")
    print(result)

"""
【输出示例】
{"metadata": {"tag": "8928552abbe4823476a10adbee40da9244d12cea0ee0951a3001e415fcaa0d6d", "attributions": ["https://developer.qweather.com/attribution.html"]}, "condition": {"text": "晴", "code": "100"}, "temperature": {"value": 21.97, "unit": "°C"}, "feelsLike": {"value": 20.17, "unit": "°C"}, "humidity": 0.42, "wind": {"direction": {"degree": 201, "compass": "ssw"}, "speed": {"value": 3.39, "unit": "m/s"}, "scale": 2}, "windGust": {"value": 7.15, "unit": "m/s"}, "precipitation": {"amount": {"value": 0, "unit": "mm"}, "intensity": {"value": 0, "unit": "mm/h"}, "type": "none"}, "pressure": {"value": 1021.21, "unit": "hPa"}, "visibility": {"value": 30120, "unit": "m"}, "dewPoint": {"value": 8.54, "unit": "°C"}, "cloudCover": 0, "uvIndex": 0, "location": {"name": "北京", "adm1": "北京市", "id": "101010100", "lat": 39.9, "lon": 116.41}}
"""
