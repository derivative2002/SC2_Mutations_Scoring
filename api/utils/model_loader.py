import os
import torch
import logging
from pathlib import Path
from src.data.preprocess import Vocab
from src.models.networks import MutationScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_path: str = "experiments/focal_loss_v2/best_acc_checkpoint.pt"):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocabs = {}
        self._load_vocabs()
        self._load_model()
    
    def _load_vocabs(self):
        """加载词表"""
        try:
            # 词表应该在模型checkpoint同级目录的vocabs目录下
            vocab_dir = self.model_path.parent / "vocabs"
            if not vocab_dir.exists():
                raise ValueError(f"词表目录不存在: {vocab_dir}")
            
            # 加载四个特征的词表
            for name in ['map', 'commander', 'mutation', 'ai']:
                self.vocabs[name] = Vocab.load(str(vocab_dir), name)
                logger.info(f"加载{name}词表，大小: {len(self.vocabs[name])}")
        
        except Exception as e:
            logger.error(f"加载词表失败: {str(e)}")
            raise
    
    def _load_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"从{self.model_path}加载模型")
            
            # 获取词表大小
            vocab_sizes = {
                'map': len(self.vocabs['map']),
                'commander': len(self.vocabs['commander']),
                'mutation': len(self.vocabs['mutation']),
                'ai': len(self.vocabs['ai'])
            }
            
            # 创建模型实例
            self.model = MutationScorer(
                num_maps=vocab_sizes['map'],
                num_commanders=vocab_sizes['commander'],
                num_mutations=vocab_sizes['mutation'],
                num_ais=vocab_sizes['ai'],
                map_dim=64,  # 这些参数可以通过配置文件设置
                commander_dim=128,
                mutation_dim=96,
                ai_dim=32,
                hidden_dims=[512, 256, 128],
                dropout=0.1,
                embed_dropout=0.1
            )
            
            # 加载模型权重
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def _prepare_inputs(self, map_name: str, commanders: list, mutations: list, enemy_ai: str):
        """准备模型输入"""
        try:
            # 将输入转换为ID
            map_id = torch.tensor([self.vocabs['map'].token2idx.get(map_name, 0)], device=self.device)
            commander_ids = torch.tensor(
                [self.vocabs['commander'].token2idx.get(c, 0) for c in commanders],
                device=self.device
            ).unsqueeze(0)  # 添加batch维度
            mutation_ids = torch.tensor(
                [self.vocabs['mutation'].token2idx.get(m, 0) for m in mutations],
                device=self.device
            ).unsqueeze(0)  # 添加batch维度
            ai_id = torch.tensor([self.vocabs['ai'].token2idx.get(enemy_ai, 0)], device=self.device)
            
            return {
                'map_ids': map_id,
                'commander_ids': commander_ids,
                'mutation_ids': mutation_ids,
                'ai_ids': ai_id
            }
        except Exception as e:
            logger.error(f"准备输入失败: {str(e)}")
            raise
    
    def predict(self, map_name: str, commanders: list, mutations: list, enemy_ai: str):
        """预测难度分数"""
        try:
            # 准备输入
            inputs = self._prepare_inputs(map_name, commanders, mutations, enemy_ai)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 返回预测结果
            return {
                'difficulty_score': outputs['score'].item(),
                'difficulty_level': outputs['level'].item(),
                'mutation_weights': outputs['attention_weights'].tolist()
            }
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            raise 