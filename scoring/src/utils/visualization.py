"""可视化工具."""

import io
from typing import Tuple

import torch
import torch.nn as nn
from torchinfo import summary

def get_model_summary(model: nn.Module) -> Tuple[str, int]:
    """获取模型结构摘要.
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型结构摘要字符串和参数总量
    """
    # 捕获模型摘要
    output = io.StringIO()
    model_summary = summary(
        model,
        input_data={
            'map_ids': torch.zeros(1, dtype=torch.long),
            'commander_ids': torch.zeros((1, 2), dtype=torch.long),
            'mutation_ids': torch.zeros((1, 8), dtype=torch.long),
            'mutation_mask': torch.ones((1, 8), dtype=torch.float),
            'ai_ids': torch.zeros(1, dtype=torch.long)
        },
        verbose=2,
        col_names=['input_size', 'output_size', 'num_params', 'kernel_size', 
                  'mult_adds'],
        col_width=20,
        row_settings=['var_names']
    )
    
    # 获取参数总量
    total_params = sum(p.numel() for p in model.parameters())
    
    return str(model_summary), total_params 