"""突变相关的 API 路由。"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..models.scorer import get_scorer
from ..rules.generator import MutationGenerator
from ..config import Config

router = APIRouter()
config = Config()

class GenerateRequest(BaseModel):
    """生成请求模型。"""
    mode: str  # 'solo' 或 'duo'
    target_difficulty: float  # 目标难度 (1-5)
    map_name: Optional[str] = None  # 可选的地图名称
    commanders: Optional[List[str]] = None  # 可选的指挥官列表

class MutationResponse(BaseModel):
    """突变响应模型。"""
    mutations: List[str]  # 突变因子列表
    difficulty: float  # 预测的难度分数
    map_name: str  # 地图名称
    commanders: List[str]  # 指挥官列表

@router.post("/generate", response_model=MutationResponse)
async def generate_mutations(request: GenerateRequest):
    """生成突变组合。
    
    Args:
        request: 生成请求
        
    Returns:
        突变组合响应
    """
    try:
        # 获取生成器
        generator = MutationGenerator(
            mode=request.mode,
            scorer=get_scorer(),
            config=config
        )
        
        # 生成突变组合
        result = generator.generate(
            target_difficulty=request.target_difficulty,
            map_name=request.map_name,
            commanders=request.commanders
        )
        
        return MutationResponse(
            mutations=result.mutations,
            difficulty=result.difficulty,
            map_name=result.map_name,
            commanders=result.commanders
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/maps")
async def get_maps():
    """获取可用地图列表。"""
    try:
        return {
            "maps": config.get_maps()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/commanders")
async def get_commanders():
    """获取可用指挥官列表。"""
    try:
        return {
            "commanders": config.get_commanders()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@router.get("/mutations")
async def get_mutations():
    """获取可用突变因子列表。"""
    try:
        return {
            "mutations": config.get_mutations()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 