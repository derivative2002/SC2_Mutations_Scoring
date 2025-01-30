"""API文档配置."""

from fastapi.openapi.models import OpenAPI, Info, License, Contact

description = """
SC2 Mutations API 允许您：

* 生成突变组合
* 获取可用地图列表
* 获取可用指挥官列表
* 获取可用突变因子列表

## 突变组合生成

您可以：

* 指定目标难度（1.0-5.0）
* 指定游戏模式（solo/duo）
* 指定地图（可选）
* 指定指挥官（可选）
* 指定难度容忍度（可选）

生成器会：

* 考虑突变因子之间的互斥和依赖关系
* 优化组合以达到目标难度
* 返回相关的规则说明

## 数据接口

您可以获取：

* 所有可用地图的列表
* 所有可用指挥官的列表
* 所有可用突变因子的列表
"""

tags_metadata = [
    {
        "name": "mutations",
        "description": "突变组合生成相关接口"
    }
]

def get_openapi_schema() -> OpenAPI:
    """获取OpenAPI文档配置."""
    return {
        "openapi": "3.0.2",
        "info": {
            "title": "SC2 Mutations API",
            "description": description,
            "version": "0.1.0",
            "contact": {
                "name": "BTL",
                "email": "your.email@example.com",
                "url": "https://github.com/your-username/sc2-mutations"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "tags": tags_metadata
    } 