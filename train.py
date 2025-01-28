"""训练脚本."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.dataset import get_dataloaders
from src.models.networks import MutationScorer
from src.training.losses import FocalLossWithRegularization
from src.training.trainer import Trainer
from src.utils.metrics import print_class_distribution
from src.utils.visualization import get_model_summary

# 配置日志
def setup_logging(save_dir: Path, experiment_name: str):
    """配置日志处理器.
    
    Args:
        save_dir: 保存目录
        experiment_name: 实验名称
    """
    # 创建实验目录
    log_dir = save_dir / experiment_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'train.log'
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 清除现有的处理器
    root_logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    return root_logger

def main():
    """主函数."""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/focal_loss.yaml',
                       help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置日志
    save_dir = Path(config['experiment']['save_dir'])
    experiment_name = config['experiment']['name']
    logger = setup_logging(save_dir, experiment_name)
    
    logger.info(f"加载配置: {args.config}")
    
    # 设置设备
    device = torch.device(config['training']['device'])
    logger.info(f"使用设备: {device}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"当前GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    # 获取数据加载器
    train_loader, val_loader = get_dataloaders(
        data_dir=config['data']['processed_dir'],
        batch_size=config['data']['batch_size'],
        val_ratio=config['data']['val_ratio'],
        use_weighted_sampler=config['data']['use_weighted_sampler'],
        num_workers=config['data']['num_workers']
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    logger.info(f"词表大小: {train_loader.dataset.vocab_sizes}")
    
    # 打印类别分布
    print_class_distribution(train_loader.dataset)
    
    # 创建模型
    model = MutationScorer(
        num_maps=train_loader.dataset.vocab_sizes['map'],
        num_commanders=train_loader.dataset.vocab_sizes['commander'],
        num_mutations=train_loader.dataset.vocab_sizes['mutation'],
        num_ais=train_loader.dataset.vocab_sizes['ai'],
        map_dim=config['model']['map_dim'],
        commander_dim=config['model']['commander_dim'],
        mutation_dim=config['model']['mutation_dim'],
        ai_dim=config['model']['ai_dim'],
        hidden_dims=config['model']['hidden_dims'],
        dropout=config['model']['dropout'],
        embed_dropout=config['model']['embed_dropout']
    ).to(device)
    
    # 打印模型信息
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 获取并保存模型结构信息
    save_dir = Path(config['experiment']['save_dir']) / config['experiment']['name']
    save_dir.mkdir(parents=True, exist_ok=True)
    model_structure_path = save_dir / 'model_structure.txt'

    summary_str, _ = get_model_summary(model)
    with open(model_structure_path, 'w', encoding='utf-8') as f:
        f.write(summary_str)
    logger.info(f"模型结构信息已保存到: {model_structure_path}")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['training']['scheduler']['factor'],
        patience=config['training']['scheduler']['patience'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # 创建损失函数
    criterion = FocalLossWithRegularization(
        alpha=config['training']['focal_alpha'],
        gamma=config['training']['focal_gamma'],
        l1_lambda=config['model']['l1_lambda'],
        l2_lambda=config['model']['l2_lambda'],
        strong_commanders=config['model']['strong_commanders'],
        commander_strength_factor=config['model']['commander_strength_factor'],
        commander_vocab=train_loader.dataset.commander_vocab
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config['training']['num_epochs'],
        device=device,
        save_dir=config['experiment']['save_dir'],
        experiment_name=config['experiment']['name']
    )
    
    # 训练模型
    best_metrics, final_metrics = trainer.train()
    
    # 打印训练结果
    logger.info("\n训练完成!")
    logger.info(f"最佳验证准确率: {best_metrics['acc']:.4f} (Epoch {best_metrics['epoch']})")
    logger.info(f"最佳验证F1分数: {best_metrics['f1']:.4f}")
    logger.info(f"最终验证准确率: {final_metrics['acc']:.4f}")
    logger.info(f"最终验证F1分数: {final_metrics['f1']:.4f}")


if __name__ == '__main__':
    main() 