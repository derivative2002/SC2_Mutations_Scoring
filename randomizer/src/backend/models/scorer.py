"""评分模型."""

import os
import json
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F

from ..config import Config
from ..logger import logger
from ..exceptions import ModelError

try:
    from scoring.src.models.networks import MutationScorer as BaseScorer
    from scoring.src.data.preprocess import Vocab
except ImportError as e:
    logger.error(f"导入scoring包失败: {str(e)}")
    logger.info("尝试使用相对导入")
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
    from scoring.src.models.networks import MutationScorer as BaseScorer
    from scoring.src.data.preprocess import Vocab

# 单例实例
_SCORER = None
_SCORER_LOCK = False

def get_scorer() -> 'MutationScorer':
    """获取评分器单例."""
    global _SCORER, _SCORER_LOCK
    if _SCORER is None and not _SCORER_LOCK:
        try:
            _SCORER_LOCK = True  # 防止递归初始化
            logger.info("开始创建评分器单例")
            scorer = MutationScorer()
            _SCORER = scorer
            logger.info("成功创建评分器单例")
        except Exception as e:
            logger.error(f"创建评分器单例失败: {str(e)}")
            # 创建一个测试模式的评分器作为后备
            logger.info("尝试创建测试模式评分器")
            scorer = MutationScorer(test_mode=True)
            _SCORER = scorer
            logger.info("使用测试模式评分器作为后备")
        finally:
            _SCORER_LOCK = False
    return _SCORER


class MutationScorer(object):
    """突变难度评分器."""
    
    __slots__ = ['test_mode', 'cache', 'cache_size', 'model', 'map_vocab', 
                 'commander_vocab', 'mutation_vocab', 'ai_vocab', 'config']
    
    def __new__(cls, *args, **kwargs):
        """创建新实例时初始化所有属性."""
        instance = super().__new__(cls)
        # 预先设置所有属性为None
        for slot in cls.__slots__:
            setattr(instance, slot, None)
        return instance
    
    def __init__(self, cache_size: int = 1000, test_mode: bool = False):
        """初始化评分器.
        
        Args:
            cache_size: 缓存大小
            test_mode: 是否为测试模式
        """
        # 基本属性初始化
        self.test_mode = test_mode
        self.cache = {}
        self.cache_size = cache_size
        self.model = None
        self.map_vocab = None
        self.commander_vocab = None
        self.mutation_vocab = None
        self.ai_vocab = None
        self.config = None
        
        logger.info(f"基本属性初始化完成: test_mode={self.test_mode}")
        
        try:
            # 加载配置
            self.config = Config()
            settings = self.config.settings
            logger.info(f"配置加载成功: {settings}")
            logger.info(f"配置类型: {type(settings)}")
            
            # 确保settings是字典类型
            if not isinstance(settings, dict):
                try:
                    settings = json.loads(settings) if isinstance(settings, str) else vars(settings)
                    logger.info(f"转换后的配置类型: {type(settings)}")
                except Exception as e:
                    logger.error(f"配置转换失败: {str(e)}")
                    settings = {}
            
            # 加载词表
            vocab_dir = Path("data/processed/vocabs")
            if not vocab_dir.is_absolute():
                vocab_dir = self.config.root_dir / vocab_dir
            logger.info(f"词表目录: {vocab_dir}")
            
            if not vocab_dir.exists():
                logger.warning(f"词表目录不存在: {vocab_dir}")
                self.test_mode = True
            else:
                try:
                    if not test_mode:
                        logger.info("开始加载词表...")
                        # 加载地图词表
                        try:
                            self.map_vocab = Vocab.load(vocab_dir, 'map')
                            logger.info("地图词表加载成功")
                        except Exception as e:
                            logger.error(f"地图词表加载失败: {str(e)}")
                            self.test_mode = True
                            
                        # 加载指挥官词表
                        try:
                            self.commander_vocab = Vocab.load(vocab_dir, 'commander')
                            logger.info("指挥官词表加载成功")
                        except Exception as e:
                            logger.error(f"指挥官词表加载失败: {str(e)}")
                            self.test_mode = True
                            
                        # 加载突变因子词表
                        try:
                            self.mutation_vocab = Vocab.load(vocab_dir, 'mutation')
                            logger.info("突变因子词表加载成功")
                        except Exception as e:
                            logger.error(f"突变因子词表加载失败: {str(e)}")
                            self.test_mode = True
                            
                        # 加载AI词表
                        try:
                            self.ai_vocab = Vocab.load(vocab_dir, 'ai')
                            logger.info("AI词表加载成功")
                        except Exception as e:
                            logger.error(f"AI词表加载失败: {str(e)}")
                            self.test_mode = True
                            
                        # 验证词表是否正确加载
                        if not all([
                            isinstance(self.map_vocab, Vocab),
                            isinstance(self.commander_vocab, Vocab),
                            isinstance(self.mutation_vocab, Vocab),
                            isinstance(self.ai_vocab, Vocab)
                        ]):
                            logger.error("词表类型验证失败")
                            self.test_mode = True
                        else:
                            logger.info("所有词表加载成功")
                            logger.info(f"词表大小: map={len(self.map_vocab)}, commander={len(self.commander_vocab)}, "
                                      f"mutation={len(self.mutation_vocab)}, ai={len(self.ai_vocab)}")
                except Exception as e:
                    logger.error(f"词表加载失败: {str(e)}")
                    self.test_mode = True
            
            # 创建模型
            if not test_mode and not self.test_mode:
                # 获取模型配置
                model_config = {
                    "map_dim": 64,
                    "commander_dim": 96,
                    "mutation_dim": 96,
                    "ai_dim": 64,
                    "hidden_dims": [256, 128, 64],
                    "num_classes": 5,
                    "dropout": 0.3,
                    "embed_dropout": 0.2,
                    "use_batch_norm": True
                }
                
                try:
                    # 尝试从配置中获取模型设置
                    if isinstance(settings, dict):
                        logger.info(f"配置键: {list(settings.keys())}")
                        if 'model' in settings and isinstance(settings['model'], dict):
                            model_settings = settings['model']
                            logger.info(f"模型配置: {model_settings}")
                            
                            if 'network' in model_settings and isinstance(model_settings['network'], dict):
                                network_config = model_settings['network']
                                logger.info(f"网络配置: {network_config}")
                                model_config.update(network_config)
                                logger.info("成功更新模型配置")
                            else:
                                logger.warning("模型配置中缺少network字段或格式错误")
                        else:
                            logger.warning("配置中缺少model字段或格式错误")
                    else:
                        logger.warning(f"配置不是字典类型: {type(settings)}")
                except Exception as e:
                    logger.error(f"处理模型配置时出错: {str(e)}")
                    logger.warning("使用默认模型配置")
                
                logger.info(f"最终模型配置: {model_config}")
                
                # 创建模型
                try:
                    if not self.map_vocab or not self.commander_vocab or not self.mutation_vocab or not self.ai_vocab:
                        logger.warning("词表未正确加载，切换到测试模式")
                        self.test_mode = True
                    else:
                        vocab_sizes = {
                            "maps": len(self.map_vocab),
                            "commanders": len(self.commander_vocab),
                            "mutations": len(self.mutation_vocab),
                            "ais": len(self.ai_vocab)
                        }
                        logger.info(f"词表大小: {vocab_sizes}")
                        
                        self.model = BaseScorer(
                            num_maps=vocab_sizes["maps"],
                            num_commanders=vocab_sizes["commanders"],
                            num_mutations=vocab_sizes["mutations"],
                            num_ais=vocab_sizes["ais"],
                            **model_config
                        )
                        logger.info("模型创建成功")
                        
                        # 加载模型权重
                        model_path = Path("experiments/focal_loss/checkpoints/focal_loss_v7/best_acc_checkpoint.pt")
                        if not model_path.is_absolute():
                            model_path = self.config.root_dir / model_path
                        logger.info(f"模型路径: {model_path}")
                        
                        if not model_path.exists():
                            logger.warning(f"模型文件不存在: {model_path}")
                            self.test_mode = True
                        else:
                            try:
                                state_dict = torch.load(model_path, map_location='cpu')
                                logger.info(f"加载的权重类型: {type(state_dict)}")
                                
                                if isinstance(state_dict, dict):
                                    if 'model_state_dict' in state_dict:
                                        state_dict = state_dict['model_state_dict']
                                    
                                    # 检查权重文件的键
                                    weight_keys = list(state_dict.keys())
                                    logger.info(f"权重文件的前5个键: {weight_keys[:5]}")
                                    
                                    # 检查模型参数的键
                                    model_keys = list(self.model.state_dict().keys())
                                    logger.info(f"模型参数的前5个键: {model_keys[:5]}")
                                    
                                    # 验证键的匹配情况
                                    missing_keys = set(model_keys) - set(weight_keys)
                                    extra_keys = set(weight_keys) - set(model_keys)
                                    
                                    if missing_keys:
                                        logger.warning(f"权重文件缺少的键: {missing_keys}")
                                    if extra_keys:
                                        logger.warning(f"权重文件多余的键: {extra_keys}")
                                    
                                    self.model.load_state_dict(state_dict)
                                    self.model.eval()
                                    logger.info("模型加载成功")
                                else:
                                    logger.error(f"权重文件格式错误: {type(state_dict)}")
                                    self.test_mode = True
                            except Exception as e:
                                logger.error(f"加载模型权重失败: {str(e)}")
                                self.test_mode = True
                except Exception as e:
                    logger.error(f"模型创建失败: {str(e)}")
                    self.test_mode = True
            
            logger.info(f"评分器初始化完成，测试模式: {self.test_mode}")
            
        except Exception as e:
            logger.error(f"评分器初始化失败: {str(e)}")
            self.test_mode = True
    
    def _generate_cache_key(self, map_name: str, commanders: List[str], mutations: List[str], ai_type: str) -> str:
        """生成缓存键.
        
        Args:
            map_name: 地图名称
            commanders: 指挥官列表
            mutations: 突变因子列表
            ai_type: AI类型
            
        Returns:
            缓存键
        """
        key_parts = [
            map_name,
            ",".join(sorted(commanders)),
            ",".join(sorted(mutations)),
            ai_type
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _manage_cache(self, key: str, value: float):
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
    
    def predict(self, map_name: str, commanders: List[str], mutations: List[str], ai_type: str = "标准") -> float:
        """预测突变难度评分.
        
        Args:
            map_name: 地图名称
            commanders: 指挥官列表
            mutations: 突变因子列表
            ai_type: AI类型
            
        Returns:
            难度评分 (1-5)
        """
        try:
            logger.info(f"开始预测: map={map_name}, commanders={commanders}, mutations={mutations}, ai={ai_type}")
            
            # 测试模式返回随机分数
            if self.test_mode or not self.model:
                logger.info(f"使用测试模式，返回随机分数 (test_mode={self.test_mode}, model={self.model is not None})")
                return float(np.random.randint(1, 6))
            
            # 检查词表是否正确加载
            vocab_status = {
                'map_vocab': self.map_vocab is not None,
                'commander_vocab': self.commander_vocab is not None,
                'mutation_vocab': self.mutation_vocab is not None,
                'ai_vocab': self.ai_vocab is not None
            }
            logger.info(f"词表状态: {vocab_status}")
            
            if not all(vocab_status.values()):
                logger.error("词表未正确加载")
                return float(np.random.randint(1, 6))
            
            # 计算缓存键
            cache_key = self._generate_cache_key(map_name, commanders, mutations, ai_type)
            if cache_key in self.cache:
                logger.info("命中缓存")
                return self.cache[cache_key]
            
            try:
                # 转换输入
                logger.info(f"词表类型: map={type(self.map_vocab)}, commander={type(self.commander_vocab)}, "
                          f"mutation={type(self.mutation_vocab)}, ai={type(self.ai_vocab)}")
                logger.info(f"词表方法: {dir(self.map_vocab)}")
                
                map_id = self.map_vocab[map_name]  # 使用[]操作符
                commander_ids = [self.commander_vocab[c] for c in commanders]  # 使用[]操作符
                mutation_ids = [self.mutation_vocab[m] for m in mutations]  # 使用[]操作符
                ai_id = self.ai_vocab[ai_type]  # 使用[]操作符
                
                logger.info(f"转换后的输入: map_id={map_id}, commander_ids={commander_ids}, mutation_ids={mutation_ids}, ai_id={ai_id}")
            except Exception as e:
                logger.error(f"特征转换失败: {str(e)}")
                logger.error(f"错误类型: {type(e)}")
                logger.error(f"错误堆栈: ", exc_info=True)
                return float(np.random.randint(1, 6))
            
            # 填充指挥官ID到固定长度2
            while len(commander_ids) < 2:
                commander_ids.append(self.commander_vocab.pad_id)  # 使用pad_id
            
            # 填充突变因子ID到固定长度8
            while len(mutation_ids) < 8:
                mutation_ids.append(self.mutation_vocab.pad_id)  # 使用pad_id
            
            # 创建输入张量
            map_tensor = torch.tensor([map_id], dtype=torch.long)  # [1]
            commander_tensor = torch.tensor([commander_ids], dtype=torch.long)  # [1, 2]
            mutation_tensor = torch.tensor([mutation_ids], dtype=torch.long)  # [1, 8]
            ai_tensor = torch.tensor([ai_id], dtype=torch.long)  # [1]
            
            # 创建突变因子掩码
            mutation_mask = torch.zeros(1, 8, dtype=torch.float)
            mutation_mask[0, :len(mutations)] = 1
            
            logger.info(f"输入张量形状: map={map_tensor.shape}, commander={commander_tensor.shape}, mutation={mutation_tensor.shape}, ai={ai_tensor.shape}, mask={mutation_mask.shape}")
            
            # 预测分数
            with torch.no_grad():
                # 使用模型的predict方法获取预测类别和概率
                preds, probs = self.model.predict(
                    map_tensor,
                    commander_tensor,
                    mutation_tensor,
                    ai_tensor,
                    mutation_mask
                )
                
                # 记录预测详情
                logger.info(f"预测概率分布: {probs.squeeze().tolist()}")
                pred_class = preds.item()
                
                # 将预测类别（0-4）映射到分数（1-5）
                score = pred_class + 1
                
                logger.info(f"预测类别: {pred_class}")
                logger.info(f"最终分数: {score}")
            
            # 更新缓存
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[cache_key] = float(score)
            
            return float(score)
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            if self.test_mode:
                return float(np.random.randint(1, 6))
            raise ModelError(f"预测失败: {str(e)}") 