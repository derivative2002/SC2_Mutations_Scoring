"""缓存模块."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logger import logger


class CacheMiddleware(BaseHTTPMiddleware):
    """缓存中间件."""
    
    def __init__(self, app, ttl: int = 300):
        """初始化缓存中间件.
        
        Args:
            app: FastAPI应用
            ttl: 缓存过期时间（秒）
        """
        super().__init__(app)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def _generate_cache_key(self, request: Request) -> str:
        """生成缓存键.
        
        Args:
            request: 请求对象
            
        Returns:
            缓存键
        """
        # 对于GET请求，使用路径作为键
        if request.method == "GET":
            return f"GET:{request.url.path}"
        
        # 对于POST请求，使用路径和请求体作为键
        if request.method == "POST":
            body = request.scope.get("body", b"").decode()
            return f"POST:{request.url.path}:{body}"
        
        return request.url.path
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """处理请求.
        
        Args:
            request: 请求对象
            call_next: 下一个处理函数
            
        Returns:
            响应对象
        """
        # 只缓存GET和POST请求
        if request.method not in ["GET", "POST"]:
            return await call_next(request)
        
        # 生成缓存键
        cache_key = self._generate_cache_key(request)
        
        # 检查缓存
        cached = self.cache.get(cache_key)
        if cached:
            # 检查是否过期
            if datetime.now() < cached["expires"]:
                logger.debug(f"使用缓存: {cache_key}")
                return Response(
                    content=cached["content"],
                    media_type="application/json"
                )
            else:
                # 删除过期缓存
                del self.cache[cache_key]
        
        # 获取响应
        response = await call_next(request)
        
        # 缓存响应
        if response.status_code == 200:
            content = b""
            async for chunk in response.body_iterator:
                content += chunk
            
            self.cache[cache_key] = {
                "content": content,
                "expires": datetime.now() + timedelta(seconds=self.ttl)
            }
            
            return Response(
                content=content,
                media_type="application/json"
            )
        
        return response 