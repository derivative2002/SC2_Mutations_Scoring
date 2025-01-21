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

from data.preprocess import prepare_data, get_all_factors
from models.gnn_attention import MutationGNN, MutationDataset, train_epoch, evaluate
from config.default_config import ModelConfig, TrainingConfig, DataConfig, PathConfig
from utils.logger import TrainingLogger
from utils.progress import EpochProgressBar, BatchProgressBar
from utils.visualization import ModelVisualizer
from utils.tensorboard import TensorBoardLogger
from utils.scheduler import get_cosine_schedule_with_warmup

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
    parser.add_argument('--tensorboard_dir', type=str, default=os.path.join(PathConfig.log_dir, 'tensorboard'),
                      help='TensorBoard日志保存目录')
    
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
    
    # 可视化模型结构
    visualizer.visualize_model_structure(model)
    
    # 获取一个批次的数据
    if train_loader is not None:
        batch = next(iter(train_loader))
        batch = batch.to(model.device)
        
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
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 创建日志记录器和可视化器
    logger = TrainingLogger(args.log_dir)
    visualizer = ModelVisualizer(args.vis_dir)
    tb_logger = TensorBoardLogger(args.tensorboard_dir)
    
    # 3. 设置随机种子和设备
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # 4. 创建必要的目录
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    
    # 5. 加载和预处理数据
    logger.logger.info("Loading and preprocessing data...")
    data_list, map_to_id = prepare_data(args.data_path)
    
    # 6. 划分训练集和验证集
    train_data, val_data = train_test_split(
        data_list,
        test_size=args.val_split,
        random_state=args.seed
    )
    logger.logger.info(f"Train size: {len(train_data)}, Val size: {len(val_data)}")
    
    # 7. 创建数据加载器
    train_dataset = MutationDataset(train_data)
    val_dataset = MutationDataset(val_data)
    
    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=DataConfig.num_workers
    )
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=DataConfig.num_workers
    )
    
    # 8. 初始化模型
    num_factors = train_data[0].x.size(1)
    num_maps = len(map_to_id)
    
    model = MutationGNN(
        num_factors=num_factors,
        num_maps=num_maps,
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_classes=ModelConfig.num_classes,
        dropout=args.dropout
    ).to(device)
    logger.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 9. 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=TrainingConfig.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = len(train_loader) * TrainingConfig.warmup_epochs
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,
        min_lr=TrainingConfig.min_lr
    )
    
    # 10. 训练模型
    best_val_acc = 0
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    patience_counter = 0
    
    logger.logger.info("Starting training...")
    pbar = EpochProgressBar(args.num_epochs)
    
    train_losses = []
    val_losses = []
    val_accs = []
    
    for epoch in range(args.num_epochs):
        # 训练一个epoch
        train_loss = train_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            scheduler,
            device,
            grad_clip=TrainingConfig.grad_clip
        )
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 记录训练信息
        logger.log_epoch(epoch + 1, train_loss, val_loss, val_acc)
        
        # 更新进度条
        pbar.update_metrics(
            epoch + 1,
            train_loss,
            val_loss,
            val_acc,
            best_val_acc
        )
        
        # TensorBoard记录
        tb_logger.log_scalars(
            'loss',
            {
                'train': train_loss,
                'val': val_loss
            },
            epoch + 1
        )
        tb_logger.log_scalar('accuracy/val', val_acc, epoch + 1)
        tb_logger.log_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
        
        # 计算并记录混淆矩阵
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.map_id, data.batch)
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(data.y.cpu().numpy())
        
        conf_matrix = confusion_matrix(all_labels, all_preds)
        tb_logger.log_confusion_matrix(
            'confusion_matrix',
            conf_matrix,
            epoch + 1
        )
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs
            }, best_model_path)
            logger.log_model_save(best_model_path)
            patience_counter = 0
            
            # 可视化最佳模型的中间结果
            visualize_model(
                model,
                train_loader,
                args.data_path
            )
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= TrainingConfig.early_stopping_patience:
            logger.log_early_stopping(epoch + 1)
            break
        
        # 保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accs': val_accs
            }, checkpoint_path)
            logger.log_model_save(checkpoint_path)
    
    pbar.close()
    
    # 11. 绘制训练历史
    visualizer.plot_training_history(
        train_losses,
        val_losses,
        val_accs
    )
    
    # 12. 关闭TensorBoard写入器
    tb_logger.close()
    
    logger.log_training_complete()

def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip=None):
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