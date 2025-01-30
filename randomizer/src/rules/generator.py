"""突变生成器模块。"""
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np

from ..config import Config
from ..models.scorer import MutationScorer
from ..logger import logger


@dataclass
class GenerationResult:
    """生成结果。"""
    mutations: List[str]  # 突变因子列表
    difficulty: float  # 难度分数
    map_name: str  # 地图名称
    commanders: List[str]  # 指挥官列表


class MutationGenerator:
    """突变生成器。"""
    
    def __init__(self,
                 mode: str,
                 scorer: MutationScorer,
                 config: Config):
        """初始化生成器。
        
        Args:
            mode: 模式 ('solo' 或 'duo')
            scorer: 难度评分器
            config: 配置对象
        """
        self.mode = mode
        self.scorer = scorer
        self.config = config
        
        # 加载规则配置
        self.max_mutations = 4 if mode == 'solo' else 8
        self.min_mutations = 2 if mode == 'solo' else 4
        
        # 加载资源
        self.maps = config.get_maps()
        self.commanders = config.get_commanders()
        self.mutations = config.get_mutations()
        
        # 加载突变规则
        self.incompatible_pairs = config.get_incompatible_pairs()
        self.required_pairs = config.get_required_pairs()
        
        # 初始化突变因子权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化突变因子权重。
        基于规则和依赖关系设置初始权重。
        """
        self.weights = {mutation: 1.0 for mutation in self.mutations}
        
        # 增加有依赖关系的突变因子的权重
        for prereq, dependent in self.required_pairs:
            self.weights[prereq] *= 1.2
            self.weights[dependent] *= 1.2
        
        # 降低经常互斥的突变因子的权重
        incompatible_count = {}
        for m1, m2 in self.incompatible_pairs:
            incompatible_count[m1] = incompatible_count.get(m1, 0) + 1
            incompatible_count[m2] = incompatible_count.get(m2, 0) + 1
        
        for mutation, count in incompatible_count.items():
            self.weights[mutation] *= (1.0 / (1.0 + 0.1 * count))
    
    def _weighted_sample(self, 
                        available_mutations: List[str], 
                        num_samples: int,
                        current_mutations: Optional[List[str]] = None) -> List[str]:
        """基于权重进行采样。
        
        Args:
            available_mutations: 可选的突变因子列表
            num_samples: 采样数量
            current_mutations: 当前已选的突变因子列表
            
        Returns:
            采样结果
        """
        if not available_mutations or num_samples <= 0:
            return []
            
        current_mutations = current_mutations or []
        
        # 计算调整后的权重
        weights = []
        for mutation in available_mutations:
            weight = self.weights[mutation]
            
            # 如果有依赖关系，调整权重
            for prereq, dependent in self.required_pairs:
                if mutation == dependent and prereq not in current_mutations:
                    weight *= 0.1  # 大幅降低权重
                elif mutation == prereq and dependent in current_mutations:
                    weight *= 2.0  # 提高权重
            
            # 如果有互斥关系，调整权重
            for m1, m2 in self.incompatible_pairs:
                if (mutation == m1 and m2 in current_mutations) or \
                   (mutation == m2 and m1 in current_mutations):
                    weight = 0  # 设置为0以排除
            
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            return []
        
        weights = [w/total_weight for w in weights]
        
        # 进行采样
        try:
            selected_indices = np.random.choice(
                len(available_mutations),
                size=min(num_samples, len(available_mutations)),
                replace=False,
                p=weights
            )
            return [available_mutations[i] for i in selected_indices]
        except ValueError:
            logger.warning("权重采样失败，使用随机采样")
            return random.sample(available_mutations, min(num_samples, len(available_mutations)))
    
    def _optimize_combination(self,
                            base_mutations: List[str],
                            target_difficulty: float,
                            tolerance: float = 0.5,
                            max_iterations: int = 10) -> Tuple[List[str], float]:
        """优化突变组合以接近目标难度。
        
        Args:
            base_mutations: 基础突变组合
            target_difficulty: 目标难度
            tolerance: 可接受的误差范围
            max_iterations: 最大迭代次数
            
        Returns:
            优化后的突变组合和对应的难度分数
        """
        best_mutations = base_mutations.copy()
        best_diff = float('inf')
        best_score = 0
        
        for _ in range(max_iterations):
            # 评估当前组合
            current_score = self.scorer.predict(
                map_name=self.current_map,
                commanders=self.current_commanders,
                mutations=best_mutations
            )
            
            diff = abs(current_score - target_difficulty)
            if diff < best_diff:
                best_diff = diff
                best_score = current_score
            
            # 如果已经足够接近，直接返回
            if diff <= tolerance:
                return best_mutations, best_score
            
            # 根据差异调整突变组合
            if current_score < target_difficulty:
                # 分数太低，尝试添加突变
                available = [m for m in self.mutations 
                           if m not in best_mutations 
                           and self._is_valid_addition(m, best_mutations)]
                if available and len(best_mutations) < self.max_mutations:
                    mutation = self._weighted_sample(available, 1)[0]
                    best_mutations.append(mutation)
            else:
                # 分数太高，尝试移除突变
                if len(best_mutations) > self.min_mutations:
                    # 优先移除非依赖项
                    removable = [m for m in best_mutations 
                               if not self._is_required_by_others(m, best_mutations)]
                    if removable:
                        mutation = random.choice(removable)
                        best_mutations.remove(mutation)
        
        return best_mutations, best_score
    
    def _is_valid_addition(self, mutation: str, current_mutations: List[str]) -> bool:
        """检查添加突变因子是否有效。"""
        # 检查互斥规则
        for m1, m2 in self.incompatible_pairs:
            if (mutation == m1 and m2 in current_mutations) or \
               (mutation == m2 and m1 in current_mutations):
                return False
        
        # 检查是否需要前置条件
        for prereq, dependent in self.required_pairs:
            if mutation == dependent and prereq not in current_mutations:
                return False
        
        return True
    
    def _is_required_by_others(self, mutation: str, current_mutations: List[str]) -> bool:
        """检查突变因子是否被其他因子依赖。"""
        for prereq, dependent in self.required_pairs:
            if mutation == prereq and dependent in current_mutations:
                return True
        return False
    
    def generate(self,
                target_difficulty: float,
                map_name: Optional[str] = None,
                commanders: Optional[List[str]] = None,
                tolerance: float = 0.5) -> GenerationResult:
        """生成突变组合。
        
        Args:
            target_difficulty: 目标难度 (1-5)
            map_name: 可选的地图名称
            commanders: 可选的指挥官列表
            tolerance: 可接受的误差范围
            
        Returns:
            生成结果
            
        Raises:
            ValueError: 如果无法生成满足条件的组合
        """
        # 验证输入
        if target_difficulty < 1 or target_difficulty > 5:
            raise ValueError("目标难度必须在1-5之间")
        
        if commanders and len(commanders) > (1 if self.mode == 'solo' else 2):
            raise ValueError("指挥官数量超出限制")
        
        # 生成或使用给定的地图和指挥官
        self.current_map = map_name or random.choice(self.maps)
        if commanders is None:
            self.current_commanders = ([random.choice(self.commanders)] 
                                     if self.mode == 'solo' 
                                     else random.sample(self.commanders, 2))
        else:
            self.current_commanders = commanders.copy()
        
        # 初始生成
        num_mutations = random.randint(self.min_mutations, self.max_mutations)
        initial_mutations = self._weighted_sample(self.mutations, num_mutations)
        
        # 优化组合
        best_mutations, difficulty = self._optimize_combination(
            initial_mutations, target_difficulty, tolerance)
        
        if not best_mutations:
            raise ValueError("无法生成有效的突变组合")
        
        return GenerationResult(
            mutations=best_mutations,
            difficulty=difficulty,
            map_name=self.current_map,
            commanders=self.current_commanders
        ) 