"""评估指标和损失函数."""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss实现.
    
    参考论文:
        "Focal Loss for Dense Object Detection"
        https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self,
                 alpha: torch.Tensor = None,
                 gamma: float = 2.0,
                 reduction: str = 'mean'):
        """初始化Focal Loss.
        
        Args:
            alpha: 各类别权重
            gamma: focusing参数
            reduction: 损失计算方式，可选'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """前向传播.
        
        Args:
            inputs: 预测logits，形状为[N, C]
            targets: 目标类别，形状为[N]
            
        Returns:
            损失值
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def analyze_class_distribution(labels: np.ndarray) -> Tuple[dict, np.ndarray]:
    """分析数据集的类别分布.
    
    Args:
        labels: 标签数组
        
    Returns:
        类别分布信息字典和类别权重数组的元组
    """
    # 统计每个类别的样本数
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_samples = len(labels)
    
    # 计算每个类别的比例
    class_dist = {}
    for label, count in zip(unique_labels, counts):
        percentage = count / total_samples * 100
        class_dist[int(label)] = {
            'count': int(count),
            'percentage': f"{percentage:.2f}%"
        }
        logger.info(
            f"难度等级 {label}: {count} 个样本 ({percentage:.2f}%)")
    
    # 计算类别权重
    class_weights = np.zeros(len(unique_labels))
    for label, count in zip(unique_labels, counts):
        class_weights[int(label)] = 1.0 / count
    
    # 归一化权重
    class_weights = class_weights / class_weights.sum() * len(unique_labels)
    class_weights = torch.FloatTensor(class_weights)
    
    return class_dist, class_weights 