"""
tools 文件夹的入口文件

它的作用只有一个：让工具能用更短的写法被引入。

    from tools import get_weather          # 走这个文件（推荐）
    from tools.weather import get_weather  # 直接指明文件，也可以

以后新增工具文件时，在这里补一行 import，就同样能用短写法了。
"""

from tools.weather import get_weather

__all__ = ["get_weather"]
