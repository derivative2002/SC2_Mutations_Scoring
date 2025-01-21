import os
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import Dict, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str):
        """
        初始化TensorBoard日志记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.writer = SummaryWriter(log_dir)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], global_step: int):
        """
        记录标量值
        
        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-标量值字典
            global_step: 全局步数
        """
        self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)
    
    def log_scalar(self, tag: str, value: float, global_step: int):
        """
        记录单个标量值
        
        Args:
            tag: 标签
            value: 标量值
            global_step: 全局步数
        """
        self.writer.add_scalar(tag, value, global_step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, global_step: int):
        """
        记录直方图
        
        Args:
            tag: 标签
            values: 张量值
            global_step: 全局步数
        """
        self.writer.add_histogram(tag, values, global_step)
    
    def log_graph(self, model: torch.nn.Module, input_data: Dict[str, torch.Tensor]):
        """
        记录模型图结构
        
        Args:
            model: PyTorch模型
            input_data: 输入数据字典
        """
        self.writer.add_graph(model, input_data)
    
    def log_embeddings(
        self,
        tag: str,
        embeddings: torch.Tensor,
        metadata: Optional[List[str]] = None,
        global_step: Optional[int] = None
    ):
        """
        记录嵌入向量
        
        Args:
            tag: 标签
            embeddings: 嵌入向量张量
            metadata: 元数据列表(可选)
            global_step: 全局步数(可选)
        """
        self.writer.add_embedding(
            embeddings,
            metadata=metadata,
            tag=tag,
            global_step=global_step
        )
    
    def log_attention(self, tag: str, attention_weights: torch.Tensor, global_step: int):
        """
        记录注意力权重
        
        Args:
            tag: 标签
            attention_weights: 注意力权重张量
            global_step: 全局步数
        """
        # 将注意力权重转换为图像格式
        fig_size = (12, 8)
        fig = plt.figure(figsize=fig_size)
        sns.heatmap(
            attention_weights.detach().cpu().numpy(),
            cmap='YlOrRd',
            annot=True,
            fmt='.2f'
        )
        self.writer.add_figure(tag, fig, global_step)
        plt.close()
    
    def log_pr_curve(
        self,
        tag: str,
        labels: torch.Tensor,
        predictions: torch.Tensor,
        global_step: Optional[int] = None,
        num_thresholds: int = 127
    ):
        """
        记录PR曲线
        
        Args:
            tag: 标签
            labels: 真实标签
            predictions: 预测值
            global_step: 全局步数(可选)
            num_thresholds: 阈值数量
        """
        self.writer.add_pr_curve(
            tag,
            labels,
            predictions,
            global_step=global_step,
            num_thresholds=num_thresholds
        )
    
    def log_confusion_matrix(
        self,
        tag: str,
        confusion_matrix: np.ndarray,
        global_step: Optional[int] = None
    ):
        """
        记录混淆矩阵
        
        Args:
            tag: 标签
            confusion_matrix: 混淆矩阵
            global_step: 全局步数(可选)
        """
        fig = plt.figure(figsize=(10, 8))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        self.writer.add_figure(tag, fig, global_step)
        plt.close()
    
    def close(self):
        """关闭写入器"""
        self.writer.close() 