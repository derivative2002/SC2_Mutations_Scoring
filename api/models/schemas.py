from typing import List, Optional
from pydantic import BaseModel, Field

class Commander(BaseModel):
    name: str
    race: str
    description: Optional[str] = None

class Mutation(BaseModel):
    name: str
    description: Optional[str] = None
    difficulty_contribution: Optional[float] = None

class Map(BaseModel):
    name: str
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    map_name: str
    commanders: List[str] = Field(..., min_items=2, max_items=2)
    mutations: List[str] = Field(..., max_items=10)
    enemy_ai: str

class PredictionResponse(BaseModel):
    difficulty_score: float
    difficulty_level: int
    analysis: str
    mutation_contributions: List[dict]

class GenerationRequest(BaseModel):
    target_difficulty: float = Field(..., ge=1.0, le=5.0)
    preferred_commanders: Optional[List[str]] = None
    preferred_map: Optional[str] = None
    num_mutations: Optional[int] = Field(default=3, ge=1, le=10)

class GenerationResponse(BaseModel):
    map_name: str
    commanders: List[str]
    mutations: List[str]
    predicted_difficulty: float
    difficulty_level: int
    analysis: str 