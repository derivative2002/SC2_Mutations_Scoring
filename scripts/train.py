"""训练脚本."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import get_dataloaders
from src.models.networks import MutationScorer
from src.training.trainer import Trainer
from src.utils.config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description='训练SC2突变难度预测模型')
    
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--resume', type=str,
                       help='恢复训练的checkpoint路径')
    
    if args is None:
        args = sys.argv[1:]
    return parser.parse_args(args)


def main(args=None):
    """主函数."""
    # 解析参数
    args = parse_args(args)
    
    # 加载配置
    config = Config.from_yaml(args.config)
    logger.info(f"加载配置: {args.config}")
    
    # 设置设备
    device = torch.device(
        config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=config.data.processed_dir,
        batch_size=config.data.batch_size,
        val_ratio=config.data.val_ratio,
        use_weighted_sampler=config.data.use_weighted_sampler,
        num_workers=config.data.num_workers
    )
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 获取词表大小
    vocab_sizes = train_loader.dataset.vocab_sizes
    logger.info(f"词表大小: {vocab_sizes}")
    
    # 创建模型
    model = MutationScorer(
        num_maps=vocab_sizes['map'],
        num_commanders=vocab_sizes['commander'],
        num_mutations=vocab_sizes['mutation'],
        num_ais=vocab_sizes['ai'],
        map_dim=config.model.map_dim,
        commander_dim=config.model.commander_dim,
        mutation_dim=config.model.mutation_dim,
        ai_dim=config.model.ai_dim,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
        embed_dropout=config.model.embed_dropout
    )
    model = model.to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 创建优化器和学习率调度器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config.training.scheduler.factor,
        patience=config.training.scheduler.patience,
        min_lr=config.training.scheduler.min_lr,
        verbose=True
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config.training.num_epochs,
        device=device,
        save_dir=config.experiment.save_dir,
        experiment_name=config.experiment.name
    )
    
    # 恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"恢复训练: {args.resume}")
    
    # 开始训练
    best_metrics, last_metrics = trainer.train()
    
    # 打印最终结果
    logger.info("\n训练完成!")
    logger.info(
        f"最佳验证准确率: {best_metrics['acc']:.4f} "
        f"(Epoch {best_metrics['epoch']})")
    logger.info(f"最佳验证F1分数: {best_metrics['f1']:.4f}")
    logger.info(f"最终验证准确率: {last_metrics['acc']:.4f}")
    logger.info(f"最终验证F1分数: {last_metrics['f1']:.4f}")


if __name__ == '__main__':
    main() 