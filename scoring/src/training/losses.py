"""损失函数模块."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLossWithRegularization(nn.Module):
    """带有正则化和先验知识的Focal Loss."""
    
    def __init__(self,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 l1_lambda: float = 1e-5,
                 l2_lambda: float = 1e-4,
                 strong_commanders: list = None,
                 commander_strength_factor: float = 0.2,
                 commander_vocab = None,
                 reduction: str = 'mean'):
        """初始化损失函数.
        
        Args:
            alpha: focal loss的alpha参数
            gamma: focal loss的gamma参数
            l1_lambda: L1正则化系数
            l2_lambda: L2正则化系数
            strong_commanders: 强力指挥官列表
            commander_strength_factor: 强力指挥官的影响因子
            commander_vocab: 指挥官词表
            reduction: 损失计算方式，可选'none', 'mean', 'sum'
        """
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.reduction = reduction
        
        # 强力指挥官相关
        self.strong_commanders = strong_commanders
        self.commander_strength_factor = float(commander_strength_factor)
        self.commander_vocab = commander_vocab
        
        # 注册strong_commander_ids作为缓冲区
        if strong_commanders and commander_vocab:
            self.register_buffer(
                'strong_commander_ids',
                torch.tensor([commander_vocab.token2idx.get(name, 0) 
                            for name in strong_commanders])
            )
        else:
            self.register_buffer('strong_commander_ids', None)
    
    def forward(self, 
                logits: torch.Tensor, 
                targets: torch.Tensor,
                commander_ids: torch.Tensor = None,
                model: nn.Module = None) -> torch.Tensor:
        """计算损失.
        
        Args:
            logits: 预测logits，形状为(batch_size, num_classes)
            targets: 目标标签，形状为(batch_size,)
            commander_ids: 指挥官ID，形状为(batch_size, 2)
            model: 模型实例，用于计算正则化损失
            
        Returns:
            损失值
        """
        device = logits.device
        
        # 计算focal loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # 考虑强力指挥官的影响
        if commander_ids is not None and self.strong_commanders:
            # 确保strong_commander_ids在正确的设备上
            if self.strong_commander_ids.device != device:
                self.strong_commander_ids = self.strong_commander_ids.to(device)
            # 检查每个样本是否包含强力指挥官
            has_strong = torch.any(
                commander_ids.unsqueeze(-1) == self.strong_commander_ids, dim=2)
            # 如果任一指挥官是强力指挥官，增加预期难度
            strong_mask = torch.any(has_strong, dim=1)
            focal_loss = torch.where(
                strong_mask,
                focal_loss * (1 - self.commander_strength_factor),
                focal_loss
            )
        
        # 计算正则化损失
        l1_reg = torch.tensor(0., device=device)
        l2_reg = torch.tensor(0., device=device)
        
        if model is not None:
            for name, param in model.named_parameters():
                if 'bias' not in name:  # 不对偏置项进行正则化
                    l1_reg += torch.norm(param, 1)
                    l2_reg += torch.norm(param, 2)
        
        # 合并所有损失
        total_loss = focal_loss
        if self.reduction == 'mean':
            total_loss = total_loss.mean()
        elif self.reduction == 'sum':
            total_loss = total_loss.sum()
            
        # 将正则化系数转换为张量并移动到正确的设备
        l1_lambda = torch.tensor(self.l1_lambda, device=device, dtype=torch.float32)
        l2_lambda = torch.tensor(self.l2_lambda, device=device, dtype=torch.float32)
        
        total_loss = total_loss + l1_lambda * l1_reg + l2_lambda * l2_reg
        
        return total_loss 