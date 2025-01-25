from fastapi import APIRouter, HTTPException
from ..models.schemas import (
    PredictionRequest,
    PredictionResponse,
    GenerationRequest,
    GenerationResponse
)
from ..services.predictor import PredictionService
from ..services.generator import GeneratorService

router = APIRouter(prefix="/api/v1")
prediction_service = PredictionService()
generator_service = GeneratorService()

@router.post("/predict", response_model=PredictionResponse)
async def predict_difficulty(request: PredictionRequest):
    """预测突变难度"""
    try:
        result = await prediction_service.predict_difficulty(
            map_name=request.map_name,
            commanders=request.commanders,
            mutations=request.mutations,
            enemy_ai=request.enemy_ai
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=GenerationResponse)
async def generate_combination(request: GenerationRequest):
    """生成指定难度的突变组合"""
    try:
        result = await generator_service.generate_combination(
            target_difficulty=request.target_difficulty,
            preferred_commanders=request.preferred_commanders,
            preferred_map=request.preferred_map,
            num_mutations=request.num_mutations
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/options")
async def get_available_options():
    """获取所有可用选项"""
    try:
        return generator_service.get_available_options()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 