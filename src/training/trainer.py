"""训练器模块."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.networks import MutationScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """模型训练器."""
    
    def __init__(self,
                 model: MutationScorer,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                 num_epochs: int,
                 device: torch.device,
                 save_dir: str,
                 experiment_name: str):
        """初始化训练器.
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            optimizer: 优化器
            scheduler: 学习率调度器
            num_epochs: 训练轮数
            device: 设备
            save_dir: 模型保存目录
            experiment_name: 实验名称
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        
        # 创建保存目录
        self.save_dir = Path(save_dir) / experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建TensorBoard目录
        tensorboard_dir = self.save_dir / 'tensorboard'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建TensorBoard写入器
        self.writer = SummaryWriter(str(tensorboard_dir))
        logger.info(f"TensorBoard日志目录: {tensorboard_dir}")
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 最佳验证指标
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        logger.info(f"训练器初始化完成，实验名称: {experiment_name}")
        logger.info(f"模型保存目录: {self.save_dir}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch.
        
        Args:
            epoch: 当前epoch
            
        Returns:
            训练指标字典
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch}")
        for batch in pbar:
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            logits = self.model(
                map_ids=batch['map_ids'],
                commander_ids=batch['commander_ids'],
                mutation_ids=batch['mutation_ids'],
                ai_ids=batch['ai_ids'],
                mutation_mask=batch['mutation_mask']
            )
            
            # 计算损失
            loss = self.criterion(logits, batch['labels'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            
            # 更新统计
            batch_size = batch['labels'].size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}"
            })
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        metrics = {
            'loss': avg_loss,
            'acc': avg_acc
        }
        
        # 记录到TensorBoard
        self.writer.add_scalar('train/loss', avg_loss, epoch)
        self.writer.add_scalar('train/acc', avg_acc, epoch)
        if self.scheduler is not None:
            self.writer.add_scalar(
                'train/lr', self.optimizer.param_groups[0]['lr'], epoch)
        
        # 确保数据被写入
        self.writer.flush()
        
        return metrics
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型.
        
        Args:
            epoch: 当前epoch
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch}")
        for batch in pbar:
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            logits = self.model(
                map_ids=batch['map_ids'],
                commander_ids=batch['commander_ids'],
                mutation_ids=batch['mutation_ids'],
                ai_ids=batch['ai_ids'],
                mutation_mask=batch['mutation_mask']
            )
            
            # 计算损失
            loss = self.criterion(logits, batch['labels'])
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            acc = (preds == batch['labels']).float().mean()
            
            # 更新统计
            batch_size = batch['labels'].size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc.item() * batch_size
            total_samples += batch_size
            
            # 收集预测和标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}"
            })
        
        # 计算平均指标
        avg_loss = total_loss / total_samples
        avg_acc = total_acc / total_samples
        
        # 计算F1分数
        f1_score = self._compute_f1_score(
            np.array(all_preds), np.array(all_labels))
        
        metrics = {
            'loss': avg_loss,
            'acc': avg_acc,
            'f1': f1_score
        }
        
        # 记录到TensorBoard
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/acc', avg_acc, epoch)
        self.writer.add_scalar('val/f1', f1_score, epoch)
        
        # 确保数据被写入
        self.writer.flush()
        
        # 保存最佳模型
        if avg_acc > self.best_val_acc:
            self.best_val_acc = avg_acc
            self._save_checkpoint(
                epoch, metrics, name='best_acc_checkpoint.pt')
        
        if f1_score > self.best_val_f1:
            self.best_val_f1 = f1_score
            self._save_checkpoint(
                epoch, metrics, name='best_f1_checkpoint.pt')
        
        return metrics
    
    def train(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """训练模型.
        
        Returns:
            最佳验证指标和最后一轮指标的元组
        """
        logger.info("开始训练...")
        best_metrics = {
            'acc': 0.0,
            'f1': 0.0,
            'epoch': 0
        }
        
        try:
            for epoch in range(1, self.num_epochs + 1):
                logger.info(f"\nEpoch {epoch}/{self.num_epochs}")
                
                # 训练一个epoch
                train_metrics = self.train_epoch(epoch)
                
                # 验证
                val_metrics = self.validate(epoch)
                
                # 更新学习率
                if self.scheduler is not None:
                    self.scheduler.step(val_metrics['loss'])
                
                # 更新最佳指标
                if val_metrics['acc'] > best_metrics['acc']:
                    best_metrics['acc'] = val_metrics['acc']
                    best_metrics['epoch'] = epoch
                if val_metrics['f1'] > best_metrics['f1']:
                    best_metrics['f1'] = val_metrics['f1']
                
                # 打印指标
                logger.info(
                    f"Train Loss: {train_metrics['loss']:.4f} "
                    f"Acc: {train_metrics['acc']:.4f}"
                )
                logger.info(
                    f"Val Loss: {val_metrics['loss']:.4f} "
                    f"Acc: {val_metrics['acc']:.4f} "
                    f"F1: {val_metrics['f1']:.4f}"
                )
                logger.info(
                    f"Best Val Acc: {best_metrics['acc']:.4f} "
                    f"(Epoch {best_metrics['epoch']})"
                )
                
                # 保存最后一个checkpoint
                self._save_checkpoint(
                    epoch, val_metrics, name='last_checkpoint.pt')
        finally:
            # 确保在训练结束或发生异常时关闭writer
            self.writer.close()
            logger.info("已关闭TensorBoard writer")
        
        return best_metrics, val_metrics
    
    def _compute_f1_score(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """计算多分类F1分数.
        
        Args:
            preds: 预测数组
            labels: 标签数组
            
        Returns:
            F1分数
        """
        # 计算每个类别的F1分数
        f1_scores = []
        for cls in range(5):  # 5个难度等级
            true_pos = np.sum((preds == cls) & (labels == cls))
            false_pos = np.sum((preds == cls) & (labels != cls))
            false_neg = np.sum((preds != cls) & (labels == cls))
            
            precision = true_pos / (true_pos + false_pos + 1e-10)
            recall = true_pos / (true_pos + false_neg + 1e-10)
            
            f1 = 2 * precision * recall / (precision + recall + 1e-10)
            f1_scores.append(f1)
        
        # 返回宏平均F1分数
        return np.mean(f1_scores)
    
    def _save_checkpoint(self,
                        epoch: int,
                        metrics: Dict[str, float],
                        name: str = 'checkpoint.pt'):
        """保存模型checkpoint.
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标
            name: 保存的文件名
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        save_path = self.save_dir / name
        torch.save(checkpoint, save_path)
        logger.info(f"保存checkpoint到: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, float]:
        """加载模型checkpoint.
        
        Args:
            checkpoint_path: checkpoint文件路径
            
        Returns:
            验证指标
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(
            f"加载checkpoint: {checkpoint_path} "
            f"(Epoch {checkpoint['epoch']})")
        
        return checkpoint['metrics'] 