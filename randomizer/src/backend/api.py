"""API路由模块."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, validator

from .models.scorer import get_scorer
from .config import Config
from .exceptions import ModelError, ConfigError
from .logger import logger

router = APIRouter()


class ScoreRequest(BaseModel):
    """评分请求模型."""
    map_name: str = Field(
        ...,
        description="地图名称",
        example="虚空降临"
    )
    commanders: List[str] = Field(
        ...,
        min_items=1,
        max_items=2,
        description="指挥官列表",
        example=["雷诺", "凯瑞甘"]
    )
    mutations: List[str] = Field(
        ...,
        min_items=1,
        max_items=8,
        description="突变因子列表",
        example=["丧尸大战", "行尸走肉"]
    )
    ai_type: str = Field(
        default="standard",
        description="AI类型",
        example="standard"
    )

    class Config:
        schema_extra = {
            "example": {
                "map_name": "虚空降临",
                "commanders": ["雷诺", "凯瑞甘"],
                "mutations": ["丧尸大战", "行尸走肉"],
                "ai_type": "standard"
            }
        }

    @validator('commanders')
    def validate_commanders(cls, v):
        """验证指挥官列表."""
        config = Config()
        valid_commanders = set(config.get_commanders())
        invalid_commanders = [c for c in v if c not in valid_commanders]
        if invalid_commanders:
            raise ValueError(f"未知的指挥官: {', '.join(invalid_commanders)}")
        return v

    @validator('map_name')
    def validate_map(cls, v):
        """验证地图名称."""
        config = Config()
        valid_maps = set(config.get_maps())
        if v not in valid_maps:
            raise ValueError(f"未知的地图: {v}")
        return v

    @validator('mutations')
    def validate_mutations(cls, v):
        """验证突变因子列表."""
        config = Config()
        valid_mutations = set(config.get_mutations())
        invalid_mutations = [m for m in v if m not in valid_mutations]
        if invalid_mutations:
            raise ValueError(f"未知的突变因子: {', '.join(invalid_mutations)}")
        
        # 检查互斥规则
        incompatible_pairs = config.get_incompatible_pairs()
        for m1 in v:
            for m2 in v:
                if m1 < m2 and (m1, m2) in incompatible_pairs:
                    reason = config.get_rule_description(m1, m2)
                    raise ValueError(f"突变因子 {m1} 和 {m2} 不能同时使用: {reason}")
        return v


class ScoreResponse(BaseModel):
    """评分响应模型."""
    score: float = Field(
        ...,
        description="难度分数 (1-5)",
        example=3.5,
        ge=1.0,
        le=5.0
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="详细信息",
        example={
            "rules": ["互斥规则：视野影响叠加，会导致游戏体验极差"],
            "num_mutations": 2,
            "commander_count": 2
        }
    )

    class Config:
        schema_extra = {
            "example": {
                "score": 3.5,
                "details": {
                    "rules": ["互斥规则：视野影响叠加，会导致游戏体验极差"],
                    "num_mutations": 2,
                    "commander_count": 2
                }
            }
        }


@router.post(
    "/mutations/score",
    response_model=ScoreResponse,
    summary="评分突变组合",
    description="""
    评估指定的突变组合难度。
    
    - 支持1-2个指挥官
    - 支持1-8个突变因子
    - 自动检查突变因子的互斥和依赖规则
    - 返回1-5的难度分数和详细信息
    """,
    response_description="难度分数和详细信息"
)
async def score_mutations(request: ScoreRequest) -> ScoreResponse:
    """评分突变组合的难度.
    
    Args:
        request: 评分请求
        
    Returns:
        难度分数和详细信息
        
    Raises:
        HTTPException: 如果评分过程出错
    """
    try:
        scorer = get_scorer()
        score = scorer.predict(
            map_name=request.map_name,
            commanders=request.commanders,
            mutations=request.mutations,
            ai_type=request.ai_type
        )
        
        # 获取规则说明
        config = Config()
        rule_descriptions = []
        for i, m1 in enumerate(request.mutations):
            for m2 in request.mutations[i+1:]:
                desc = config.get_rule_description(m1, m2)
                if desc:
                    rule_descriptions.append(desc)
        
        # 获取详细信息
        map_details = config.get_map_details(request.map_name)
        commander_details = [
            config.get_commander_details(commander)
            for commander in request.commanders
        ]
        mutation_details = [
            config.get_mutation_details(mutation)
            for mutation in request.mutations
        ]
        
        return ScoreResponse(
            score=score,
            details={
                "rules": rule_descriptions,
                "num_mutations": len(request.mutations),
                "commander_count": len(request.commanders),
                "map_details": map_details,
                "commander_details": commander_details,
                "mutation_details": mutation_details
            }
        )
    except ModelError as e:
        logger.error(f"评分失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"输入验证失败: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"评分过程出错: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="内部服务器错误"
        )


@router.get(
    "/mutations/maps",
    response_model=List[Dict[str, Any]],
    summary="获取地图列表",
    description="获取所有可用的地图列表",
    response_description="地图名称列表"
)
async def get_maps() -> List[Dict[str, Any]]:
    """获取可用地图列表."""
    try:
        config = Config()
        maps = config.get_maps()
        return [
            {
                "name": map_name,
                **config.get_map_details(map_name)
            }
            for map_name in maps
        ]
    except ConfigError as e:
        logger.error(f"获取地图列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/mutations/commanders",
    response_model=List[Dict[str, Any]],
    summary="获取指挥官列表",
    description="获取所有可用的指挥官列表",
    response_description="指挥官名称列表"
)
async def get_commanders() -> List[Dict[str, Any]]:
    """获取可用指挥官列表."""
    try:
        config = Config()
        commanders = config.get_commanders()
        return [
            {
                "name": commander_name,
                **config.get_commander_details(commander_name)
            }
            for commander_name in commanders
        ]
    except ConfigError as e:
        logger.error(f"获取指挥官列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/mutations/mutations",
    response_model=List[Dict[str, Any]],
    summary="获取突变因子列表",
    description="获取所有可用的突变因子列表",
    response_description="突变因子名称列表"
)
async def get_mutations() -> List[Dict[str, Any]]:
    """获取可用突变因子列表."""
    try:
        config = Config()
        mutations = config.get_mutations()
        return [
            {
                "name": mutation_name,
                **config.get_mutation_details(mutation_name)
            }
            for mutation_name in mutations
        ]
    except ConfigError as e:
        logger.error(f"获取突变因子列表失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get(
    "/mutations/rules",
    response_model=Dict[str, List[Dict[str, str]]],
    summary="获取突变规则",
    description="""
    获取突变因子的组合规则，包括：
    
    - 互斥规则：不能同时使用的突变因子组合
    - 依赖规则：需要搭配使用的突变因子组合
    """,
    response_description="突变规则列表"
)
async def get_rules() -> Dict[str, List[Dict[str, str]]]:
    """获取突变规则."""
    try:
        config = Config()
        return {
            "incompatible_pairs": [
                {
                    "mutation1": m1,
                    "mutation2": m2,
                    "description": config.get_rule_description(m1, m2)
                }
                for m1, m2 in config.get_incompatible_pairs()
            ],
            "required_pairs": [
                {
                    "prerequisite": m1,
                    "dependent": m2,
                    "description": config.get_rule_description(m1, m2)
                }
                for m1, m2 in config.get_required_pairs()
            ]
        }
    except ConfigError as e:
        logger.error(f"获取规则失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        ) 