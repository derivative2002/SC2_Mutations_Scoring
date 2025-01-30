"""FastAPI应用."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .config import Config
from .exceptions import RandomizerError
from .logger import logger
from .api.mutations import router as mutations_router
from .api.docs import get_openapi_schema


def create_app() -> FastAPI:
    """创建FastAPI应用.
    
    Returns:
        FastAPI应用实例
    """
    config = Config()
    
    # 创建应用
    app = FastAPI(
        title=config.settings.app.name,
        version=config.settings.app.version,
        description=config.settings.app.description,
        debug=config.settings.app.debug
    )
    
    # 配置CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 在生产环境中应该限制来源
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 异常处理
    @app.exception_handler(RandomizerError)
    async def randomizer_exception_handler(request: Request, exc: RandomizerError):
        """处理自定义异常."""
        logger.error(
            f"Error processing request: {exc.message}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "code": exc.code,
                "details": exc.details
            }
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": exc.code,
                "message": exc.message,
                "details": exc.details
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """处理未捕获的异常."""
        logger.error(
            f"Unhandled error: {str(exc)}",
            extra={
                "path": request.url.path,
                "method": request.method
            },
            exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {"error": str(exc)} if config.settings.app.debug else {}
            }
        )
    
    # 注册路由
    app.include_router(mutations_router, prefix="/api/mutations", tags=["mutations"])
    
    # 配置OpenAPI文档
    app.openapi_schema = get_openapi_schema()
    
    # 健康检查
    @app.get("/health")
    async def health_check():
        """健康检查接口."""
        return {
            "status": "ok",
            "version": config.settings.app.version
        }
    
    return app
 