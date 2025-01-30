"""FastAPI应用主入口."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .api import router
from .cache import CacheMiddleware
from .logger import logger
from .exceptions import RandomizerError

# 创建FastAPI应用
app = FastAPI(
    title="SC2突变评分API",
    description="""
    # 星际争霸2合作任务突变组合难度评分API

    ## 功能特点

    - 🎯 **突变组合评分**: 基于深度学习模型的突变组合难度预测
    - 📊 **数据验证**: 严格的输入验证，确保数据有效性
    - 🔄 **缓存支持**: 5分钟响应缓存，提升性能
    - 🛡️ **错误处理**: 详细的错误信息和日志记录

    ## 使用说明

    1. 使用 `/api/mutations/maps`、`/api/mutations/commanders` 和 `/api/mutations/mutations` 获取可用选项
    2. 使用 `/api/mutations/rules` 查看突变因子的组合规则
    3. 使用 `/api/mutations/score` 评估突变组合的难度

    ## 注意事项

    - 指挥官数量限制：1-2个
    - 突变因子数量限制：1-8个
    - 部分突变因子组合可能互斥或需要搭配使用
    """,
    version="1.0.0",
    contact={
        "name": "SC2 Mutations Team",
        "url": "https://github.com/your-repo",
        "email": "your-email@example.com",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=False,  # 不需要凭证
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头部
)

# 添加缓存中间件
app.add_middleware(CacheMiddleware, ttl=300)  # 5分钟缓存

# 注册路由
app.include_router(
    router,
    prefix="/api",
    tags=["mutations"],
    responses={
        400: {
            "description": "请求错误",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "输入验证失败"
                    }
                }
            }
        },
        500: {
            "description": "服务器错误",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "内部服务器错误"
                    }
                }
            }
        }
    }
)


def custom_openapi():
    """自定义OpenAPI文档."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # 添加安全模式（如果需要）
    # openapi_schema["components"]["securitySchemes"] = {...}
    
    # 添加标签描述
    openapi_schema["tags"] = [
        {
            "name": "mutations",
            "description": "突变组合相关的操作",
            "externalDocs": {
                "description": "游戏介绍",
                "url": "https://starcraft2.com/zh-cn/"
            }
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.exception_handler(RandomizerError)
async def randomizer_exception_handler(request, exc):
    """处理自定义异常."""
    logger.error(f"请求处理出错: {str(exc)}")
    return {
        "code": "INTERNAL_ERROR",
        "message": str(exc),
        "details": {}
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """处理通用异常."""
    logger.error(f"未处理的异常: {str(exc)}")
    return {
        "code": "INTERNAL_ERROR",
        "message": "内部服务器错误",
        "details": {"error": str(exc)}
    }


@app.on_event("startup")
async def startup_event():
    """应用启动时的处理."""
    logger.info("API服务启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的处理."""
    logger.info("API服务关闭") 