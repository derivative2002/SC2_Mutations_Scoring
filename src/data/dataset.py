"""数据集模块."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from src.data.preprocess import Vocab

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SC2MutationDataset(Dataset):
    """星际2突变数据集."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 val_ratio: float = 0.1):
        """初始化数据集.
        
        Args:
            data_dir: 数据目录
            split: 数据集划分，可选 'train', 'val', 'test'
            val_ratio: 验证集比例
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # 加载数据
        data_path = self.data_dir / "processed_data.npz"
        data = np.load(data_path)
        
        # 加载元数据
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 加载词表
        vocab_dir = self.data_dir / "vocabs"
        self.map_vocab = Vocab.load(vocab_dir, 'map')
        self.commander_vocab = Vocab.load(vocab_dir, 'commander')
        self.mutation_vocab = Vocab.load(vocab_dir, 'mutation')
        self.ai_vocab = Vocab.load(vocab_dir, 'ai')
        
        # 划分数据集
        num_samples = len(data['labels'])
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        val_size = int(num_samples * val_ratio)
        if split == 'train':
            self.indices = indices[val_size:]
        elif split == 'val':
            self.indices = indices[:val_size]
        else:  # test
            self.indices = indices
        
        # 保存特征和标签
        self.features = {
            'map_ids': data['map_ids'],
            'commander_ids': data['commander_ids'],
            'mutation_ids': data['mutation_ids'],
            'mutation_mask': data['mutation_mask'],
            'ai_ids': data['ai_ids']
        }
        self.labels = data['labels']
        
        logger.info(f"加载{split}数据集: {len(self)} 个样本")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取数据样本.
        
        Args:
            idx: 样本索引
            
        Returns:
            包含特征和标签的字典
        """
        idx = self.indices[idx]
        return {
            'map_ids': torch.tensor(self.features['map_ids'][idx], dtype=torch.long),
            'commander_ids': torch.tensor(
                self.features['commander_ids'][idx], dtype=torch.long),
            'mutation_ids': torch.tensor(
                self.features['mutation_ids'][idx], dtype=torch.long),
            'mutation_mask': torch.tensor(
                self.features['mutation_mask'][idx], dtype=torch.float),
            'ai_ids': torch.tensor(self.features['ai_ids'][idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    @property
    def num_classes(self) -> int:
        """获取类别数量."""
        return self.metadata['num_classes']
    
    @property
    def class_weights(self) -> Optional[torch.Tensor]:
        """获取类别权重."""
        if self.metadata['class_weights'] is not None:
            return torch.tensor(self.metadata['class_weights'], dtype=torch.float)
        return None
    
    @property
    def vocab_sizes(self) -> Dict[str, int]:
        """获取词表大小."""
        return {
            'map': len(self.map_vocab),
            'commander': len(self.commander_vocab),
            'mutation': len(self.mutation_vocab),
            'ai': len(self.ai_vocab)
        }


def create_dataloader(
    dataset: SC2MutationDataset,
    batch_size: int,
    shuffle: bool = True,
    use_weighted_sampler: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """创建数据加载器.
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        use_weighted_sampler: 是否使用加权采样
        num_workers: 数据加载线程数
        
    Returns:
        数据加载器
    """
    if use_weighted_sampler and dataset.split == 'train':
        # 计算样本权重
        sample_weights = [
            dataset.class_weights[label] 
            for label in dataset.labels[dataset.indices]
        ]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=dataset.split == 'train'  # 只在训练集丢弃最后一个不完整的batch
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    val_ratio: float = 0.1,
    use_weighted_sampler: bool = False,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """获取训练和验证数据加载器.
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        val_ratio: 验证集比例
        use_weighted_sampler: 是否使用加权采样
        num_workers: 数据加载线程数
        
    Returns:
        训练和验证数据加载器的元组
    """
    # 创建训练集
    train_dataset = SC2MutationDataset(
        data_dir, split='train', val_ratio=val_ratio)
    train_loader = create_dataloader(
        train_dataset, 
        batch_size=batch_size,
        use_weighted_sampler=use_weighted_sampler,
        num_workers=num_workers
    )
    
    # 创建验证集
    val_dataset = SC2MutationDataset(
        data_dir, split='val', val_ratio=val_ratio)
    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader 