import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GraphDataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from data.preprocess import prepare_data, get_all_factors
from models.gnn_attention import MutationGNN, MutationDataset, train_epoch, evaluate
from config.default_config import ModelConfig, TrainingConfig, DataConfig, PathConfig
from utils.logger import TrainingLogger
from utils.progress import EpochProgressBar, BatchProgressBar
from utils.visualization import ModelVisualizer

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='训练突变评级预测模型')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=ModelConfig.hidden_dim,
                      help='隐藏层维度')
    parser.add_argument('--num_gnn_layers', type=int, default=ModelConfig.num_gnn_layers,
                      help='GNN层数')
    parser.add_argument('--dropout', type=float, default=ModelConfig.dropout,
                      help='Dropout率')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=TrainingConfig.batch_size,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=TrainingConfig.num_epochs,
                      help='训练轮数')
    parser.add_argument('--lr', type=float, default=TrainingConfig.learning_rate,
                      help='学习率')
    parser.add_argument('--weight_decay', type=float, default=TrainingConfig.weight_decay,
                      help='权重衰减')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=DataConfig.train_data_path,
                      help='训练数据路径')
    parser.add_argument('--val_split', type=float, default=DataConfig.val_split,
                      help='验证集比例')
    parser.add_argument('--seed', type=int, default=DataConfig.random_seed,
                      help='随机种子')
    
    # 路径参数
    parser.add_argument('--model_dir', type=str, default=PathConfig.model_dir,
                      help='模型保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default=PathConfig.checkpoint_dir,
                      help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default=PathConfig.log_dir,
                      help='日志保存目录')
    parser.add_argument('--vis_dir', type=str, default=os.path.join(PathConfig.model_dir, 'visualization'),
                      help='可视化结果保存目录')
    
    return parser.parse_args()

def visualize_model(model: MutationGNN, train_loader: Optional[GraphDataLoader] = None, data_path: str = None):
    """
    可视化模型的各个组件
    
    Args:
        model: 训练好的模型
        train_loader: 训练数据加载器
        data_path: 数据文件路径
    """
    # 创建可视化目录
    os.makedirs('visualizations', exist_ok=True)
    visualizer = ModelVisualizer(save_dir='visualizations')
    
    # 使用TensorBoard可视化模型结构
    if train_loader is not None:
        batch = next(iter(train_loader))
        batch = batch.to(model.device)
        writer = SummaryWriter('runs/mutation_scoring')
        writer.add_graph(model, (batch.x, batch.edge_index, batch.map_id, batch.batch))
        writer.close()
        print("模型结构已保存到TensorBoard，可以使用 tensorboard --logdir runs/mutation_scoring 查看")
        
        # 获取注意力权重
        with torch.no_grad():
            _, attention_weights = model(batch.x, batch.edge_index, batch.map_id, batch.batch, return_attention=True)
        
        # 获取地图名称列表
        if data_path is not None:
            df = pd.read_csv(data_path)
            map_names = df['地图'].unique().tolist()
        else:
            map_names = [f'Map {i}' for i in range(attention_weights.size(0))]
        
        # 可视化注意力权重
        visualizer.plot_attention_weights(attention_weights, map_names)

def main():
    """
    主函数
    """
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs('runs/mutation_scoring', exist_ok=True)
    
    # 初始化日志记录器
    logger = TrainingLogger(log_dir=args.log_dir)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # 加载和预处理数据
    logger.logger.info("Loading and preprocessing data...")
    data_list, map_to_id = prepare_data(args.data_path)
    
    # 获取第一个数据样本来了解数据结构
    first_data = data_list[0]
    edge_index = first_data.edge_index
    factor_mapping = {str(i): i for i in range(first_data.x.size(1))}  # 从特征维度推断因子数量
    
    # 打印因子映射
    logger.logger.info("\nFactor mapping:")
    for factor, idx in factor_mapping.items():
        logger.logger.info(f"{factor}: {idx}")
    
    # 打印边的信息
    logger.logger.info(f"\nNumber of edges: {edge_index.size(1)}")
    logger.logger.info(f"Edge index range: [{edge_index.min().item()}, {edge_index.max().item()}]")
    logger.logger.info(f"Number of factors: {len(factor_mapping)}")
    logger.logger.info(f"Edge index shape: {edge_index.shape}")
    logger.logger.info(f"Edge index max value: {edge_index.max().item()}")
    logger.logger.info(f"Edge index min value: {edge_index.min().item()}")
    
    # 划分训练集和验证集
    indices = np.arange(len(data_list))
    train_indices, val_indices = train_test_split(
        indices, test_size=args.val_split, random_state=args.seed
    )
    
    # 创建训练集和验证集
    train_dataset = [data_list[i] for i in train_indices]
    val_dataset = [data_list[i] for i in val_indices]
    
    # 创建数据加载器
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size)
    
    logger.logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # 初始化模型
    model = MutationGNN(
        num_factors=len(factor_mapping),
        num_maps=len(map_to_id),
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        dropout=args.dropout,
        device=device
    ).to(device)
    
    logger.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=TrainingConfig.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=TrainingConfig.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 创建学习率调度器
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = len(train_loader) * TrainingConfig.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 创建 TensorBoard writer
    writer = SummaryWriter('runs/mutation_scoring')
    
    # 开始训练
    logger.logger.info("Starting training...")
    best_val_acc = 0.0
    progress_bar = EpochProgressBar(args.num_epochs)
    
    try:
        for epoch in range(1, args.num_epochs + 1):
            # 训练一个epoch
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, grad_clip=1.0)
            
            # 在验证集上评估
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 记录指标
            logger.logger.info(f"Epoch [{epoch}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - Best Val Acc: {best_val_acc:.4f}")
            
            # 记录TensorBoard日志
            try:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_acc, epoch)
            except Exception as e:
                logger.logger.warning(f"Failed to write to TensorBoard: {str(e)}")
            
            # 如果是最佳模型，保存检查点
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
                logger.logger.info("Model saved to model/best_model.pth")
                
                # 在保存最佳模型时进行可视化
                try:
                    visualize_model(model, train_loader, args.data_path)
                except Exception as e:
                    logger.logger.warning(f"Failed to visualize model: {str(e)}")
    
    except KeyboardInterrupt:
        logger.logger.info("Training interrupted by user")
    except Exception as e:
        logger.logger.error(f"Training failed with error: {str(e)}")
    finally:
        writer.close()
        progress_bar.close()

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=1.0):
    """
    训练一个epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 计算设备
        grad_clip: 梯度裁剪阈值
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    # 创建批次进度条
    pbar = BatchProgressBar(len(loader), desc='Training Batches')
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out, _ = model(data.x, data.edge_index, data.map_id, data.batch, return_attention=True)
        loss = criterion(out, data.y)
        
        # 添加L2正则化损失
        if hasattr(model, 'current_l2_loss'):
            loss = loss + model.current_l2_loss
        
        loss.backward()
        
        # 梯度裁剪
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        scheduler.step()
        
        batch_loss = loss.item() * data.num_graphs
        total_loss += batch_loss
        
        # 更新进度条
        pbar.update_loss(batch_loss / data.num_graphs)
        pbar.update()
    
    pbar.close()
    return total_loss / len(loader.dataset)

if __name__ == '__main__':
    main() 