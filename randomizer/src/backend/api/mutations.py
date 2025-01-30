"""突变组合生成API."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from ..config import Config
from ..models.scorer import get_scorer
from ..rules.generator import MutationGenerator
from ..exceptions import GenerationError
from .models import GenerationRequest, GenerationResponse, ErrorResponse

router = APIRouter()


def get_config() -> Config:
    """获取配置依赖项."""
    return Config()


@router.post(
    "/generate",
    response_model=GenerationResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def generate_mutations(
    request: GenerationRequest,
    config: Config = Depends(get_config)
) -> GenerationResponse:
    """生成突变组合.
    
    Args:
        request: 生成请求
        config: 配置对象
        
    Returns:
        生成结果
        
    Raises:
        HTTPException: 请求处理出错
    """
    try:
        # 获取评分器
        scorer = get_scorer()
        
        # 创建生成器
        generator = MutationGenerator(request.mode, scorer, config)
        
        # 生成突变组合
        result = generator.generate(
            target_difficulty=request.target_difficulty,
            map_name=request.map_name,
            commanders=request.commanders,
            tolerance=request.tolerance
        )
        
        # 获取相关规则说明
        rules = []
        
        # 检查依赖规则
        for prereq, dep in config.get_required_pairs():
            if dep in result.mutations and prereq in result.mutations:
                rules.append(config.get_rule_description(prereq, dep))
        
        # 返回结果
        return GenerationResponse(
            map_name=result.map_name,
            commanders=result.commanders,
            mutations=result.mutations,
            difficulty=result.difficulty,
            rules=rules
        )
        
    except GenerationError as e:
        raise HTTPException(
            status_code=400,
            detail=ErrorResponse(
                code="GENERATION_ERROR",
                message=str(e)
            ).dict()
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="INTERNAL_ERROR",
                message="生成突变组合时出错",
                details={"error": str(e)}
            ).dict()
        )


@router.get("/maps", response_model=List[str])
async def get_maps(config: Config = Depends(get_config)) -> List[str]:
    """获取可用地图列表."""
    return config.get_maps()


@router.get("/commanders", response_model=List[str])
async def get_commanders(config: Config = Depends(get_config)) -> List[str]:
    """获取可用指挥官列表."""
    return config.get_commanders()


@router.get("/mutations", response_model=List[str])
async def get_mutations(config: Config = Depends(get_config)) -> List[str]:
    """获取可用突变因子列表."""
    return config.get_mutations() 