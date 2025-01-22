"""
可视化工具模块
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
from typing import List, Dict, Any, Optional
import seaborn as sns
import torch.nn as nn

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class ModelVisualizer:
    """模型可视化器"""
    def __init__(self, save_dir: str):
        """
        初始化可视化器
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def visualize_model_structure(self, model: nn.Module):
        """
        可视化模型结构
        
        Args:
            model: 要可视化的模型
        """
        dot = Digraph(comment='模型结构')
        dot.attr(rankdir='TB')
        
        # 添加输入节点
        dot.node('input', '输入\n(因子, 边索引, 地图ID)', shape='box')
        
        # 添加GNN层
        for i, layer in enumerate(model.gnn_layers):
            dot.node(f'gnn_{i}', f'GNN层 {i}\n{layer.__class__.__name__}', shape='box')
            if i == 0:
                dot.edge('input', f'gnn_{i}')
            else:
                dot.edge(f'gnn_{i-1}', f'gnn_{i}')
        
        # 添加地图嵌入层
        dot.node('map_embedding', '地图嵌入层', shape='box')
        dot.edge('input', 'map_embedding')
        
        # 添加注意力层
        dot.node('attention', '注意力层', shape='box')
        dot.edge(f'gnn_{len(model.gnn_layers)-1}', 'attention')
        dot.edge('map_embedding', 'attention')
        
        # 添加全连接层
        fc_layers = [model.fc1, model.fc2, model.fc3]
        for i, layer in enumerate(fc_layers):
            dot.node(f'fc_{i}', f'全连接层 {i}\n{layer.in_features}->{layer.out_features}', shape='box')
            if i == 0:
                dot.edge('attention', f'fc_{i}')
            else:
                dot.edge(f'fc_{i-1}', f'fc_{i}')
        
        # 添加输出节点
        dot.node('output', '输出\n(评级)', shape='box')
        dot.edge(f'fc_{len(fc_layers)-1}', 'output')
        
        # 保存图像
        dot.render(os.path.join(self.save_dir, 'model_structure'), format='png', cleanup=True)
    
    def plot_training_history(
        self,
        train_losses: List[float],
        val_losses: List[float],
        val_accs: List[float],
        filename: str = 'training_history'
    ):
        """
        绘制训练历史
        
        Args:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            val_accs: 验证准确率列表
            filename: 输出文件名
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 创建图形
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # 绘制损失曲线
        ax1.plot(epochs, train_losses, 'b-', label='训练损失')
        ax1.plot(epochs, val_losses, 'r-', label='验证损失')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制准确率曲线
        ax2.plot(epochs, val_accs, 'g-', label='验证准确率')
        ax2.set_title('验证准确率')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        plt.close()
    
    def plot_attention_weights(
        self,
        attention_weights: dict,
        map_names: list,
        filename: str = 'attention_weights'
    ):
        """
        可视化注意力权重
        
        Args:
            attention_weights: 注意力权重字典
            map_names: 地图名称列表
            filename: 保存的文件名
        """
        plt.figure(figsize=(15, 6))
        
        # 1. 自注意力权重
        plt.subplot(1, 2, 1)
        self_attn = attention_weights['self_attention'].squeeze().detach().cpu().numpy()
        if len(self_attn.shape) == 1:
            self_attn = self_attn.reshape(-1, 1)
        sns.heatmap(
            self_attn,
            cmap='YlOrRd',
            xticklabels=['注意力'] if self_attn.shape[1] == 1 else map_names,
            yticklabels=map_names
        )
        plt.title('自注意力权重')
        
        # 2. 交叉注意力权重
        plt.subplot(1, 2, 2)
        cross_attn = attention_weights['cross_attention'].squeeze().detach().cpu().numpy()
        if len(cross_attn.shape) == 1:
            cross_attn = cross_attn.reshape(-1, 1)
        sns.heatmap(
            cross_attn,
            cmap='YlOrRd',
            xticklabels=['注意力'] if cross_attn.shape[1] == 1 else map_names,
            yticklabels=map_names
        )
        plt.title('交叉注意力权重')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        plt.close()
    
    def plot_factor_embeddings(
        self,
        embeddings: torch.Tensor,
        factors: List[str],
        filename: str = 'factor_embeddings'
    ):
        """
        可视化因子嵌入
        使用t-SNE降维到2D空间
        
        Args:
            embeddings: 因子嵌入矩阵 [num_factors, embedding_dim]
            factors: 因子名称列表
            filename: 输出文件名
        """
        from sklearn.manifold import TSNE
        
        # 降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
        
        # 绘制散点图
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        
        # 添加标签
        for i, factor in enumerate(factors):
            plt.annotate(factor, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.title('因子嵌入可视化 (t-SNE)')
        plt.xlabel('维度 1')
        plt.ylabel('维度 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        plt.close()
    
    def plot_prediction_distribution(
        self,
        predictions: List[int],
        true_labels: List[int],
        filename: str = 'prediction_distribution'
    ):
        """
        可视化预测分布
        
        Args:
            predictions: 预测评级列表
            true_labels: 真实评级列表
            filename: 输出文件名
        """
        plt.figure(figsize=(12, 6))
        
        # 绘制直方图
        plt.hist(
            [predictions, true_labels],
            label=['预测值', '真实值'],
            bins=range(1, 12),  # 1-10的评级
            alpha=0.5,
            rwidth=0.8
        )
        
        plt.title('预测值与真实值分布对比')
        plt.xlabel('评级')
        plt.ylabel('数量')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{filename}.png'))
        plt.close() 