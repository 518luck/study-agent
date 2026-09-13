"""
【工具】天气查询（和风天气）：查某个城市的实时天气

这是整个项目公用的天气工具，放在 src/tools/ 下，其他章节的示例想查天气时直接引入即可：

    from tools import get_weather          # 推荐
    from tools.weather import get_weather  # 直接指明文件，也可以

    result = get_weather.invoke("北京")   # 单独调用
    tools = [get_weather]                # 交给 Agent / 链

和风的请求部分（拼地址、带凭据、错误处理）在 _qweather.py 里，本文件只写工具本身。
两个环境变量配置在项目根目录的 .env 里：QWEATHER_API_HOST 和 QWEATHER_API_KEY。
"""

import json

from dotenv import load_dotenv
from langchain_core.tools import tool

from tools._qweather import _qweather_get

load_dotenv(encoding="utf-8")


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
        风向风速、降水、气压、能见度、露点、云量、紫外线指数等信息，以及实际命中的地点。
        注意 humidity 和 cloudCover 是 0~1 的小数（不是百分比），uvIndex 取值 0~15。
    """
    try:
        # 第 1 步：城市名 → 经纬度。和风天气的天气接口只认坐标，先用 GeoAPI 查一次地点
        geo = _qweather_get("/geo/v2/city/lookup", {"location": loc, "lang": "zh"})

        # 同名地点可能匹配到多条，接口已按匹配度排序，取第一条
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

        # 天气接口要求经纬度最多两位小数，GeoAPI 返回的是五位数，先四舍五入再拼进地址
        lat = round(float(place["lat"]), 2)
        lon = round(float(place["lon"]), 2)

        # 第 2 步：拿坐标查实时天气
        data = _qweather_get(f"/weather/v1/current/{lat}/{lon}", {"lang": "zh"})
    except RuntimeError as exc:
        # KEY 不对、额度不足、请求超限等，都转成 JSON 错误说明返回，
        # 让模型能把原因转述给用户，而不是让整条链直接报错中断
        return json.dumps({"error": str(exc)}, ensure_ascii=False)

    # 第 3 步：把解析到的城市信息一起返回——同名城市不止一个，模型需要知道实际查的是哪一个
    data["location"] = {
        "name": place["name"],
        "adm1": place["adm1"],
        "id": place["id"],
        "lat": lat,
        "lon": lon,
    }

    # 第 4 步：转成 JSON 字符串返回；ensure_ascii=False 让中文保持可读
    return json.dumps(data, ensure_ascii=False)


# 单独运行本文件时做一次真实调用，确认 .env 和接口都没问题
if __name__ == "__main__":
    print(get_weather.invoke("北京"))
