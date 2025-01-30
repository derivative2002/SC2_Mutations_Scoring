"""突变组合生成器."""

import random
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from ..config import Config
from ..models.scorer import MutationScorer
from ..logger import logger
from ..exceptions import GenerationError


@dataclass
class GenerationResult:
    """生成结果."""
    
    map_name: str
    commanders: List[str]
    mutations: List[str]
    difficulty: float


class MutationGenerator:
    """突变组合生成器."""
    
    def __init__(self, mode: str, scorer: MutationScorer, config: Optional[Config] = None):
        """初始化生成器.
        
        Args:
            mode: 游戏模式 ('solo'/'duo')
            scorer: 难度评分器
            config: 配置对象（可选）
        """
        self.mode = mode
        self.scorer = scorer
        self.config = config or Config()
        
        # 加载配置
        mode_config = self.config.settings.generator.mode[mode]
        self.max_mutations = mode_config.max_mutations
        self.min_mutations = mode_config.min_mutations
        self.difficulty_range = mode_config.difficulty_range
        self.tolerance = mode_config.tolerance
        
        # 加载优化配置
        opt_config = self.config.settings.generator.optimization
        self.max_iterations = opt_config.max_iterations
        self.sample_size = opt_config.sample_size
        self.temperature = opt_config.temperature
        
        # 加载规则权重
        rules_config = self.config.settings.generator.rules
        self.incompatible_weight = rules_config.incompatible_weight
        self.required_weight = rules_config.required_weight
        self.balance_weight = rules_config.balance_weight
        
        # 加载游戏数据
        self.maps = self.config.get_maps()
        self.commanders = self.config.get_commanders()
        self.mutations = self.config.get_mutations()
        self.incompatible_pairs = self.config.get_incompatible_pairs()
        self.required_pairs = self.config.get_required_pairs()
        
        # 当前生成状态
        self.current_map: Optional[str] = None
        self.current_commanders: List[str] = []
        
        logger.info(
            f"生成器初始化完成 - 模式: {mode}, "
            f"突变数量范围: {self.min_mutations}-{self.max_mutations}"
        )
    
    def _get_mutation_weight(self, mutation: str, current_mutations: List[str]) -> float:
        """计算突变因子的选择权重.
        
        Args:
            mutation: 待选突变因子
            current_mutations: 当前已选突变因子列表
            
        Returns:
            选择权重
        """
        weight = 1.0
        
        # 检查互斥规则
        for m1, m2 in self.incompatible_pairs:
            if mutation == m1 and m2 in current_mutations:
                weight += self.incompatible_weight
            elif mutation == m2 and m1 in current_mutations:
                weight += self.incompatible_weight
        
        # 检查依赖规则
        for prereq, dep in self.required_pairs:
            if mutation == dep and prereq not in current_mutations:
                weight += self.required_weight
            elif mutation == prereq and dep not in current_mutations:
                weight += self.required_weight
        
        # 平衡性调整
        if current_mutations:
            try:
                # 计算添加和不添加该突变的难度差异
                current_score = self.scorer.predict(
                    self.current_map,
                    self.current_commanders,
                    current_mutations
                )
                with_mutation = current_mutations + [mutation]
                new_score = self.scorer.predict(
                    self.current_map,
                    self.current_commanders,
                    with_mutation
                )
                # 根据难度变化调整权重
                diff = abs(new_score - current_score)
                weight += self.balance_weight * (1.0 / (1.0 + diff))
            except Exception as e:
                logger.warning(f"计算难度时出错: {str(e)}")
        
        return max(0.0, weight)  # 确保权重非负
    
    def _weighted_sample(self,
                        candidates: List[str],
                        num_samples: int,
                        current_mutations: Optional[List[str]] = None) -> List[str]:
        """带权重的随机采样.
        
        Args:
            candidates: 候选突变因子列表
            num_samples: 采样数量
            current_mutations: 当前已选突变因子列表（可选）
            
        Returns:
            采样结果列表
        """
        if not candidates:
            return []
        
        current_mutations = current_mutations or []
        
        # 计算每个候选项的权重
        weights = [
            self._get_mutation_weight(m, current_mutations)
            for m in candidates
        ]
        
        # 应用温度系数
        weights = [w ** (1.0 / self.temperature) for w in weights]
        
        # 如果所有权重都为0，使用均匀分布
        if sum(weights) == 0:
            weights = [1.0] * len(candidates)
        
        # 归一化权重
        total = sum(weights)
        weights = [w / total for w in weights]
        
        # 采样
        try:
            return random.choices(
                candidates,
                weights=weights,
                k=min(num_samples, len(candidates))
            )
        except Exception as e:
            logger.error(f"采样出错: {str(e)}")
            # 发生错误时使用无权重采样
            return random.sample(
                candidates,
                k=min(num_samples, len(candidates))
            )
    
    def _is_valid_addition(self, mutation: str, current_mutations: List[str]) -> bool:
        """检查添加突变因子是否有效.
        
        Args:
            mutation: 待添加的突变因子
            current_mutations: 当前已选突变因子列表
            
        Returns:
            是否可以添加
        """
        # 检查互斥规则
        for m1, m2 in self.incompatible_pairs:
            if (mutation == m1 and m2 in current_mutations) or \
               (mutation == m2 and m1 in current_mutations):
                return False
        
        # 检查依赖规则
        for prereq, dep in self.required_pairs:
            if mutation == dep and prereq not in current_mutations:
                return False
        
        return True
    
    def _optimize_combination(self,
                            initial_mutations: List[str],
                            target_difficulty: float,
                            tolerance: Optional[float] = None,
                            max_iterations: Optional[int] = None) -> Tuple[List[str], float]:
        """优化突变组合以达到目标难度.
        
        Args:
            initial_mutations: 初始突变组合
            target_difficulty: 目标难度
            tolerance: 难度容忍度（可选）
            max_iterations: 最大迭代次数（可选）
            
        Returns:
            优化后的突变组合和难度值的元组
        """
        tolerance = tolerance or self.tolerance
        max_iterations = max_iterations or self.max_iterations
        
        current_mutations = initial_mutations.copy()
        available_mutations = set(self.mutations) - set(current_mutations)
        
        # 获取当前难度
        try:
            current_difficulty = self.scorer.predict(
                self.current_map,
                self.current_commanders,
                current_mutations
            )
        except Exception as e:
            logger.error(f"计算初始难度时出错: {str(e)}")
            return current_mutations, 0.0
        
        # 如果已经在容忍范围内，直接返回
        if abs(current_difficulty - target_difficulty) <= tolerance:
            return current_mutations, current_difficulty
        
        # 迭代优化
        for _ in range(max_iterations):
            if current_difficulty < target_difficulty:
                # 难度太低，尝试添加突变
                if len(current_mutations) >= self.max_mutations:
                    break
                
                # 获取可添加的突变
                valid_additions = [
                    m for m in available_mutations
                    if self._is_valid_addition(m, current_mutations)
                ]
                
                if not valid_additions:
                    break
                
                # 采样并添加突变
                mutation = self._weighted_sample(valid_additions, 1)[0]
                current_mutations.append(mutation)
                available_mutations.remove(mutation)
                
            else:
                # 难度太高，尝试移除突变
                if len(current_mutations) <= self.min_mutations:
                    break
                
                # 随机移除一个突变
                mutation = random.choice(current_mutations)
                current_mutations.remove(mutation)
                available_mutations.add(mutation)
            
            # 重新计算难度
            try:
                current_difficulty = self.scorer.predict(
                    self.current_map,
                    self.current_commanders,
                    current_mutations
                )
                
                # 检查是否达到目标
                if abs(current_difficulty - target_difficulty) <= tolerance:
                    break
                    
            except Exception as e:
                logger.error(f"优化过程中计算难度时出错: {str(e)}")
                break
        
        return current_mutations, current_difficulty
    
    def generate(self,
                target_difficulty: float,
                map_name: Optional[str] = None,
                commanders: Optional[List[str]] = None,
                tolerance: Optional[float] = None) -> GenerationResult:
        """生成突变组合.
        
        Args:
            target_difficulty: 目标难度
            map_name: 地图名称（可选）
            commanders: 指挥官列表（可选）
            tolerance: 难度容忍度（可选）
            
        Returns:
            生成结果对象
            
        Raises:
            GenerationError: 生成过程出错
        """
        try:
            # 验证输入
            if not (self.difficulty_range[0] <= target_difficulty <= self.difficulty_range[1]):
                raise ValueError(
                    f"目标难度 {target_difficulty} 超出范围 "
                    f"[{self.difficulty_range[0]}, {self.difficulty_range[1]}]"
                )
            
            # 选择或验证地图
            if map_name is None:
                self.current_map = random.choice(self.maps)
            else:
                if map_name not in self.maps:
                    raise ValueError(f"未知地图: {map_name}")
                self.current_map = map_name
            
            # 选择或验证指挥官
            if commanders is None:
                num_commanders = 2 if self.mode == 'duo' else 1
                self.current_commanders = random.sample(self.commanders, num_commanders)
            else:
                if not all(c in self.commanders for c in commanders):
                    raise ValueError(f"未知指挥官: {commanders}")
                if len(commanders) != (2 if self.mode == 'duo' else 1):
                    raise ValueError(
                        f"{self.mode}模式需要{2 if self.mode == 'duo' else 1}个指挥官"
                    )
                self.current_commanders = commanders
            
            # 初始采样
            initial_size = (self.min_mutations + self.max_mutations) // 2
            initial_mutations = []
            available_mutations = set(self.mutations)
            
            # 逐个添加突变
            while len(initial_mutations) < initial_size:
                # 获取可添加的突变
                valid_mutations = [
                    m for m in available_mutations
                    if self._is_valid_addition(m, initial_mutations)
                ]
                
                if not valid_mutations:
                    break
                
                # 采样并添加突变
                mutation = self._weighted_sample(valid_mutations, 1)[0]
                initial_mutations.append(mutation)
                available_mutations.remove(mutation)
            
            # 优化组合
            final_mutations, difficulty = self._optimize_combination(
                initial_mutations,
                target_difficulty,
                tolerance
            )
            
            return GenerationResult(
                map_name=self.current_map,
                commanders=self.current_commanders,
                mutations=final_mutations,
                difficulty=difficulty
            )
            
        except ValueError as e:
            raise GenerationError(str(e))
        except Exception as e:
            logger.error(f"生成突变组合时出错: {str(e)}")
            raise GenerationError(f"生成突变组合时出错: {str(e)}") 