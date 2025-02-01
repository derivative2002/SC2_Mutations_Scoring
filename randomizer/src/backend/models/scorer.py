"""评分模型."""

import os
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch

from scoring.src.models.networks import MutationScorer as BaseScorer
from scoring.src.data.preprocess import Vocab
from ..config import Config  # 改为相对导入
from ..logger import logger  # 改为相对导入
from ..exceptions import ModelError  # 改为相对导入

_SCORER = None


def get_scorer() -> 'MutationScorer':
    """获取评分器单例."""
    global _SCORER
    if _SCORER is None:
        _SCORER = MutationScorer()
    return _SCORER


class MutationScorer:
    """突变难度评分器."""
    
    def __init__(self, cache_size: int = 1000, test_mode: bool = False):
        """初始化评分器.
        
        Args:
            cache_size: 缓存大小
            test_mode: 是否为测试模式
        """
        try:
            config = Config()
            
            # 加载词表
            vocab_dir = Path("data/processed/vocabs")  # 使用训练数据的词表
            self.map_vocab = Vocab.load(vocab_dir, 'map') if not test_mode else None
            self.commander_vocab = Vocab.load(vocab_dir, 'commander') if not test_mode else None
            self.mutation_vocab = Vocab.load(vocab_dir, 'mutation') if not test_mode else None
            self.ai_vocab = Vocab.load(vocab_dir, 'ai') if not test_mode else None
            
            # 创建模型
            if not test_mode:
                self.model = BaseScorer(
                    num_maps=len(self.map_vocab),
                    num_commanders=len(self.commander_vocab),
                    num_mutations=len(self.mutation_vocab),
                    num_ais=len(self.ai_vocab),
                    **config.settings.model.network
                )
                
                # 加载模型权重
                model_path = Path("experiments/focal_loss/checkpoints/focal_loss_v7/best_acc_checkpoint.pt")
                state_dict = torch.load(model_path, map_location='cpu')
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.model.load_state_dict(state_dict)
                self.model.eval()
            else:
                self.model = None
            
            # 初始化缓存
            self.cache: Dict[str, int] = {}  # 修改为整数缓存
            self.cache_size = cache_size
            self.test_mode = test_mode
            
            logger.info("评分器初始化成功")
            
        except Exception as e:
            logger.error(f"评分器初始化失败: {str(e)}")
            if not test_mode:
                raise ModelError(f"评分器初始化失败: {str(e)}")
    
    def _generate_cache_key(self,
                          map_name: str,
                          commanders: List[str],
                          mutations: List[str],
                          ai_type: str) -> str:
        """生成缓存键.
        
        Args:
            map_name: 地图名称
            commanders: 指挥官列表
            mutations: 突变因子列表
            ai_type: AI类型
            
        Returns:
            缓存键
        """
        data = {
            "map": map_name,
            "commanders": sorted(commanders),
            "mutations": sorted(mutations),
            "ai": ai_type
        }
        return hashlib.md5(json.dumps(data).encode()).hexdigest()
    
    def _manage_cache(self, key: str, value: int):  # 修改为整数值
        """管理缓存大小.
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if len(self.cache) >= self.cache_size:
            # 移除最早的缓存项
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def predict(self,
               map_name: str,
               commanders: List[str],
               mutations: List[str],
               ai_type: Optional[str] = 'standard') -> int:  # 修改返回类型为整数
        """预测难度分数.
        
        Args:
            map_name: 地图名称
            commanders: 指挥官列表
            mutations: 突变因子列表
            ai_type: AI类型
            
        Returns:
            难度分数 (1-5)
            
        Raises:
            ModelError: 如果预测过程出错
        """
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(
                map_name, commanders, mutations, ai_type)
            if cache_key in self.cache:
                logger.debug(f"使用缓存结果: {cache_key}")
                return self.cache[cache_key]
            
            if self.test_mode:
                # 测试模式下使用简单的评分规则
                base_score = len(mutations) * 0.8
                if "丧尸大战" in mutations and "行尸走肉" in mutations:
                    base_score += 0.5  # 增加依赖组合的难度
                if "虚空裂隙" in mutations and "暗无天日" not in mutations:
                    base_score += 0.3  # 增加单个强力突变的难度
                score = min(5, max(1, round(base_score)))  # 确保是1-5的整数
                self._manage_cache(cache_key, score)
                return score
            
            with torch.no_grad():
                # 转换为ID
                try:
                    map_id = torch.tensor([self.map_vocab[map_name]], dtype=torch.long)
                except KeyError:
                    raise ModelError(f"未知地图: {map_name}")
                
                # 填充指挥官ID
                commander_ids = []
                for commander in commanders:
                    try:
                        commander_ids.append(self.commander_vocab[commander])
                    except KeyError:
                        raise ModelError(f"未知指挥官: {commander}")
                
                while len(commander_ids) < 2:  # 补充到2个
                    commander_ids.append(self.commander_vocab.pad_id)
                commander_ids = torch.tensor([commander_ids], dtype=torch.long)
                
                # 填充突变ID
                mutation_ids = []
                for mutation in mutations:
                    try:
                        mutation_ids.append(self.mutation_vocab[mutation])
                    except KeyError:
                        raise ModelError(f"未知突变因子: {mutation}")
                
                while len(mutation_ids) < 8:  # 补充到8个
                    mutation_ids.append(self.mutation_vocab.pad_id)
                mutation_ids = torch.tensor([mutation_ids], dtype=torch.long)
                
                # 创建突变掩码
                mutation_mask = torch.zeros(1, 8, dtype=torch.float)
                mutation_mask[0, :len(mutations)] = 1
                
                # AI ID
                try:
                    ai_id = torch.tensor([self.ai_vocab[ai_type]], dtype=torch.long)
                except KeyError:
                    raise ModelError(f"未知AI类型: {ai_type}")
                
                # 预测
                logits = self.model(
                    map_ids=map_id,
                    commander_ids=commander_ids,
                    mutation_ids=mutation_ids,
                    ai_ids=ai_id,
                    mutation_mask=mutation_mask
                )
                
                # 直接使用分类结果（0-4 映射到 1-5）
                pred_class = torch.argmax(logits, dim=-1).item()  # 获取最高概率的类别
                score = pred_class + 1  # 0-4 映射到 1-5
                
                # 缓存结果
                self._manage_cache(cache_key, score)
                
                return score
                
        except ModelError:
            raise
        except Exception as e:
            logger.error(f"预测过程出错: {str(e)}")
            raise ModelError(f"预测过程出错: {str(e)}") 