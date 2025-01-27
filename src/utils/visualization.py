"""Visualization utilities for model analysis."""

import os
import torch
import torch.nn as nn
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

def get_model_summary(model: nn.Module) -> Tuple[str, dict]:
    """获取模型结构摘要。
    
    Args:
        model: 模型实例
        
    Returns:
        model_summary: 模型结构摘要字符串
        model_stats: 模型统计信息字典
    """
    try:
        from torchinfo import summary
        
        # 创建示例输入
        batch_size = 2
        num_mutations = 10
        device = next(model.parameters()).device
        
        input_size = [
            (batch_size,),  # map_ids
            (batch_size, 2),  # commander_ids
            (batch_size, num_mutations),  # mutation_ids
            (batch_size,),  # ai_ids
            (batch_size, num_mutations)  # mutation_mask
        ]
        
        # 生成模型摘要
        model_summary = summary(
            model, 
            input_size=input_size,
            dtypes=[
                torch.long,  # map_ids
                torch.long,  # commander_ids
                torch.long,  # mutation_ids
                torch.long,  # ai_ids
                torch.float  # mutation_mask
            ],
            device=device,
            verbose=2,
            col_names=["output_size", "num_params"],
        )
        
        # 获取模型统计信息
        model_stats = {
            'total_params': model_summary.total_params,
            'trainable_params': model_summary.trainable_params,
            'input_size': str(model_summary.input_size),
            'total_output': str(model_summary.summary_list[-1].output_size)
        }
        
        return str(model_summary), model_stats
        
    except ImportError:
        logger.warning("请安装torchinfo以启用模型结构可视化功能")
        return "", {}

def visualize_model(model: nn.Module, save_dir: str):
    """生成并保存模型结构信息。
    
    Args:
        model: 模型实例
        save_dir: 保存目录
    """
    # 获取模型摘要
    model_summary, model_stats = get_model_summary(model)
    
    if model_summary:
        # 保存模型结构信息
        os.makedirs(save_dir, exist_ok=True)
        summary_file = os.path.join(save_dir, "model_structure.txt")
        
        with open(summary_file, 'w') as f:
            f.write("模型结构摘要:\n")
            f.write("=" * 80 + "\n")
            f.write(model_summary)
            f.write("\n\n")
            f.write("模型统计信息:\n")
            f.write("=" * 80 + "\n")
            for key, value in model_stats.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"模型结构信息已保存到: {summary_file}")
    else:
        logger.warning("无法生成模型结构信息") 