"""API模型定义."""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class GenerationRequest(BaseModel):
    """突变组合生成请求."""
    
    target_difficulty: float = Field(
        ...,  # 必需字段
        ge=1.0,  # 最小值
        le=5.0,  # 最大值
        description="目标难度 (1.0-5.0)"
    )
    
    map_name: Optional[str] = Field(
        None,
        description="地图名称（可选）"
    )
    
    commanders: Optional[List[str]] = Field(
        None,
        description="指挥官列表（可选）"
    )
    
    mode: str = Field(
        "solo",
        pattern="^(solo|duo)$",
        description="游戏模式 ('solo'/'duo')"
    )
    
    tolerance: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="难度容忍度（可选，0.0-1.0）"
    )
    
    @validator('commanders')
    def validate_commanders(cls, v, values):
        """验证指挥官数量."""
        if v is not None:
            mode = values.get('mode', 'solo')
            expected_count = 2 if mode == 'duo' else 1
            if len(v) != expected_count:
                raise ValueError(
                    f"{mode}模式需要{expected_count}个指挥官"
                )
        return v


class GenerationResponse(BaseModel):
    """突变组合生成响应."""
    
    map_name: str = Field(
        ...,
        description="地图名称"
    )
    
    commanders: List[str] = Field(
        ...,
        description="指挥官列表"
    )
    
    mutations: List[str] = Field(
        ...,
        description="突变因子列表"
    )
    
    difficulty: float = Field(
        ...,
        description="预测难度"
    )
    
    rules: List[str] = Field(
        [],
        description="相关规则说明"
    )


class ErrorResponse(BaseModel):
    """错误响应."""
    
    code: str = Field(
        ...,
        description="错误代码"
    )
    
    message: str = Field(
        ...,
        description="错误信息"
    )
    
    details: Optional[dict] = Field(
        None,
        description="错误详情"
    ) 