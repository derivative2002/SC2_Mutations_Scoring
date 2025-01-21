import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
    last_epoch: int = -1
):
    """
    创建带预热的余弦学习率调度器
    
    Args:
        optimizer: PyTorch优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数
        min_lr: 最小学习率
        last_epoch: 上一轮epoch数
    
    Returns:
        scheduler: 学习率调度器
    """
    def lr_lambda(current_step):
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦衰减阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        scale = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        
        # 应用最小学习率
        base_lr = optimizer.param_groups[0]['lr']
        min_lr_scale = min_lr / base_lr
        return max(min_lr_scale, scale)
    
    return LambdaLR(optimizer, lr_lambda, last_epoch) 