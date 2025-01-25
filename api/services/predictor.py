from typing import List, Dict
import logging
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model_loader = ModelLoader()
        logger.info("Prediction service initialized")
    
    async def predict_difficulty(
        self,
        map_name: str,
        commanders: List[str],
        mutations: List[str],
        enemy_ai: str
    ) -> Dict:
        """预测突变难度
        
        Args:
            map_name: 地图名称
            commanders: 指挥官列表
            mutations: 突变因子列表
            enemy_ai: 敌方AI类型
            
        Returns:
            Dict: 预测结果
        """
        try:
            # 验证输入
            if len(commanders) != 2:
                raise ValueError("必须指定两个指挥官")
            if not mutations:
                raise ValueError("必须至少指定一个突变因子")
            if len(mutations) > 10:
                raise ValueError("突变因子数量不能超过10个")
            
            # 进行预测
            result = self.model_loader.predict(
                map_name=map_name,
                commanders=commanders,
                mutations=mutations,
                enemy_ai=enemy_ai
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise 