"""配置加载模块."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """数据配置."""
    
    raw_data_path: str
    processed_dir: str
    max_mutations: int = 10
    val_ratio: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    use_weighted_sampler: bool = True


@dataclass
class ModelConfig:
    """模型配置."""
    
    # 嵌入维度
    map_dim: int = 64
    commander_dim: int = 128
    mutation_dim: int = 96
    ai_dim: int = 32
    
    # MLP配置
    hidden_dims: List[int] = None
    dropout: float = 0.2
    embed_dropout: float = 0.1
    
    # 正则化配置
    l1_lambda: float = 1e-5
    l2_lambda: float = 1e-4
    
    # 先验知识配置
    strong_commanders: List[str] = None
    commander_strength_factor: float = 0.2
    
    def __post_init__(self):
        """初始化后处理."""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.strong_commanders is None:
            self.strong_commanders = []


@dataclass
class SchedulerConfig:
    """学习率调度器配置."""
    
    factor: float = 0.5
    patience: int = 5
    min_lr: float = 1e-6
    
    def __post_init__(self):
        """初始化后处理."""
        # 确保数值类型正确
        self.factor = float(self.factor)
        self.patience = int(self.patience)
        self.min_lr = float(self.min_lr)


@dataclass
class TrainingConfig:
    """训练配置."""
    
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = 'cuda'
    scheduler: Optional[SchedulerConfig] = None
    
    # focal loss参数
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    def __post_init__(self):
        """初始化后处理."""
        # 确保数值类型正确
        self.num_epochs = int(self.num_epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.focal_alpha = float(self.focal_alpha)
        self.focal_gamma = float(self.focal_gamma)
        
        # 处理scheduler配置
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)


@dataclass
class ExperimentConfig:
    """实验配置."""
    
    name: str
    save_dir: str
    seed: int = 42


@dataclass
class Config:
    """总配置."""
    
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """从YAML文件加载配置.
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置对象
        """
        config_path = Path(config_path)
        
        # 读取YAML文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 创建配置对象
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            experiment=ExperimentConfig(**config_dict['experiment'])
        )
    
    def to_yaml(self, save_path: str):
        """保存配置到YAML文件.
        
        Args:
            save_path: 保存路径
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为字典
        config_dict = {
            'data': vars(self.data),
            'model': vars(self.model),
            'training': {
                **vars(self.training),
                'scheduler': vars(self.training.scheduler)
                if self.training.scheduler else None
            },
            'experiment': vars(self.experiment)
        }
        
        # 保存到YAML文件
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False,
                     allow_unicode=True)
        
        logger.info(f"配置已保存到: {save_path}") 