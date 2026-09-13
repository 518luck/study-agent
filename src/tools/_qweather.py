"""
和风天气的请求部分（内部使用）

这里不是「工具」，只放和风天气共用的请求逻辑：拼地址、带凭据、把错误转成人话。
天气工具本身在 weather.py，需要用到请求时从本文件引入。

函数名开头的下划线表示「内部使用」：外部只该用 get_weather，不用关心这个函数。

和风天气需要两个环境变量，配置在项目根目录的 .env 里：
    QWEATHER_API_HOST=你的APIHost      # 控制台「设置」页分配，形如 abcxyz.qweatherapi.com，不带 https://
    QWEATHER_API_KEY=你的APIKEY        # 控制台「项目和凭据」页创建
"""

import os

import httpx
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# 旧版错误码的释义（官方文档：错误码 v1），用来把 code 翻译成人能看懂的原因
CODE_HINTS = {
    "400": "请求参数错误或缺少必选参数",
    "401": "认证失败，通常是 KEY 不对、凭据类型不匹配，或 API Host 填错",
    "402": "额度不足或超出访问次数",
    "403": "无访问权限，或使用了错误的 API Host",
    "404": "查询的地区不存在",
    "429": "超过 QPM 限制（每分钟请求次数）",
    "500": "接口服务异常，可稍后重试",
}


def _qweather_get(path: str, params: dict) -> dict:
    """按和风天气的规则发一次 GET 请求，返回解析后的 JSON；失败时抛出 RuntimeError。"""
    api_host = os.getenv("QWEATHER_API_HOST")
    api_key = os.getenv("QWEATHER_API_KEY")
    if not api_host or not api_key:
        raise RuntimeError(
            "缺少环境变量 QWEATHER_API_HOST 或 QWEATHER_API_KEY，"
            "请先在本项目根目录的 .env 中配置"
        )

    # 请求地址由「API Host + 接口路径」拼成；凭据放在请求头里，不要写死在代码中
    response = httpx.get(
        f"https://{api_host}{path}",
        params=params,
        headers={"X-QW-Api-Key": api_key},
        timeout=30,  # 超时保护，避免长时间卡住
    )

    try:
        data = response.json()
    except ValueError:
        data = {}  # 官方文档说明 404 / 405 不返回响应体

    # 和风目前新旧两套错误码并存，两种都要判断：
    # 新版：HTTP 状态码本身就是错误码，细节在 error 字段里
    if response.status_code != 200:
        error = data.get("error") or {}
        detail = " ".join(filter(None, [error.get("title"), error.get("detail")]))
        raise RuntimeError(
            f"和风天气请求失败：HTTP {response.status_code} {detail}".strip()
        )

    # 旧版：HTTP 状态码恒为 200，错误只藏在响应体的 code 字段里；204 表示该地区暂无数据
    code = data.get("code")
    if code is not None and str(code) not in ("200", "204"):
        raise RuntimeError(
            f"和风天气返回错误码 code={code}：{CODE_HINTS.get(str(code), '未知错误')}"
        )

    return data
