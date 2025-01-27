"""训练脚本."""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

from src.data.dataset import get_dataloaders
from src.models.networks import MutationScorer
from src.training.trainer import Trainer
from src.utils.config import Config
from src.utils.metrics import analyze_class_distribution
from src.training.losses import FocalLossWithRegularization
from src.utils.visualization import visualize_model

def setup_logging(log_dir: Path):
    """配置日志输出到文件和控制台."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

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
    
    # 设置日志
    log_dir = Path(config.experiment.save_dir) / config.experiment.name / 'logs'
    log_file = setup_logging(log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件保存在: {log_file}")
    logger.info(f"加载配置: {args.config}")
    
    # 设置设备
    device = torch.device(
        config.training.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"当前GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
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
    
    # 分析训练集和验证集的类别分布
    train_labels = train_loader.dataset.labels
    val_labels = val_loader.dataset.labels
    
    logger.info("\n训练集分布:")
    train_dist, train_weights = analyze_class_distribution(train_labels)
    
    logger.info("\n验证集分布:")
    val_dist, val_weights = analyze_class_distribution(val_labels)
    
    # 分析类别分布
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['labels'].numpy())
    class_dist, class_weights = analyze_class_distribution(np.array(all_labels))

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
    
    # 生成模型结构图
    model_dir = Path(config.experiment.save_dir) / config.experiment.name
    visualize_model(model, str(model_dir))
    
    # 创建损失函数
    criterion = FocalLossWithRegularization(
        alpha=config.training.focal_alpha,
        gamma=config.training.focal_gamma,
        l1_lambda=config.model.l1_lambda,
        l2_lambda=config.model.l2_lambda,
        strong_commanders=config.model.strong_commanders,
        commander_strength_factor=config.model.commander_strength_factor,
        commander_vocab=train_loader.dataset.commander_vocab
    ).to(device)
    
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
        criterion=criterion,
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