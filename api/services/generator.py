import random
from typing import List, Dict, Optional
import logging
import itertools
from ..utils.model_loader import ModelLoader

logger = logging.getLogger(__name__)

class GeneratorService:
    def __init__(self):
        self.model_loader = ModelLoader()
        self._load_available_options()
        logger.info("Generator service initialized")
    
    def _load_available_options(self):
        """从模型词表加载可用的选项"""
        vocabs = self.model_loader.vocabs
        self.available_maps = list(vocabs['map'].token2idx.keys())
        self.available_commanders = list(vocabs['commander'].token2idx.keys())
        self.available_mutations = list(vocabs['mutation'].token2idx.keys())
        self.available_ais = list(vocabs['ai'].token2idx.keys())
    
    async def generate_combination(
        self,
        target_difficulty: float,
        preferred_commanders: Optional[List[str]] = None,
        preferred_map: Optional[str] = None,
        num_mutations: int = 3,
        tolerance: float = 0.5,
        max_attempts: int = 100
    ) -> Dict:
        """生成指定难度的突变组合
        
        Args:
            target_difficulty: 目标难度（1-5）
            preferred_commanders: 偏好的指挥官列表
            preferred_map: 偏好的地图
            num_mutations: 突变因子数量
            tolerance: 难度容差
            max_attempts: 最大尝试次数
            
        Returns:
            Dict: 生成的组合
        """
        try:
            best_combination = None
            min_diff = float('inf')
            
            for _ in range(max_attempts):
                # 生成随机组合
                combination = self._generate_random_combination(
                    preferred_commanders,
                    preferred_map,
                    num_mutations
                )
                
                # 预测难度
                result = self.model_loader.predict(
                    map_name=combination['map_name'],
                    commanders=combination['commanders'],
                    mutations=combination['mutations'],
                    enemy_ai=combination['enemy_ai']
                )
                
                # 计算与目标难度的差距
                diff = abs(result['difficulty_score'] - target_difficulty)
                
                # 如果在容差范围内，直接返回
                if diff <= tolerance:
                    return {**combination, **result}
                
                # 更新最佳组合
                if diff < min_diff:
                    min_diff = diff
                    best_combination = {**combination, **result}
            
            if best_combination is None:
                raise ValueError("无法生成符合要求的组合")
            
            return best_combination
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise
    
    def _generate_random_combination(
        self,
        preferred_commanders: Optional[List[str]],
        preferred_map: Optional[str],
        num_mutations: int
    ) -> Dict:
        """生成随机组合"""
        # 选择地图
        map_name = preferred_map if preferred_map else random.choice(self.available_maps)
        
        # 选择指挥官
        if preferred_commanders and len(preferred_commanders) == 2:
            commanders = preferred_commanders
        else:
            commanders = random.sample(self.available_commanders, 2)
        
        # 选择突变因子
        mutations = random.sample(self.available_mutations, num_mutations)
        
        # 选择AI
        enemy_ai = random.choice(self.available_ais)
        
        return {
            'map_name': map_name,
            'commanders': commanders,
            'mutations': mutations,
            'enemy_ai': enemy_ai
        }
    
    def get_available_options(self) -> Dict:
        """获取所有可用选项"""
        return {
            'maps': self.available_maps,
            'commanders': self.available_commanders,
            'mutations': self.available_mutations,
            'ais': self.available_ais
        } 