"""运行实验脚本."""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.data.preprocess import SC2MutationPreprocessor
from src.utils.config import Config
from scripts.train import main as train_main

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """设置随机种子.
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_data(config: Config):
    """预处理数据.
    
    Args:
        config: 配置对象
    """
    logger.info("开始数据预处理...")
    
    # 创建预处理器
    preprocessor = SC2MutationPreprocessor(
        raw_data_path=config.data.raw_data_path,
        processed_dir=config.data.processed_dir,
        max_mutations=config.data.max_mutations
    )
    
    # 处理数据
    preprocessor.process()
    
    logger.info("数据预处理完成")


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description='运行SC2突变难度预测实验')
    
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='是否跳过数据预处理')
    
    args = parser.parse_args()
    return args


def main():
    """主函数."""
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = Config.from_yaml(args.config)
    logger.info(f"加载配置: {args.config}")
    
    # 设置随机种子
    set_seed(config.experiment.seed)
    logger.info(f"设置随机种子: {config.experiment.seed}")
    
    # 创建实验目录
    experiment_dir = Path(config.experiment.save_dir) / config.experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config.save(experiment_dir / 'config.yaml')
    
    # 数据预处理
    if not args.skip_preprocess:
        preprocess_data(config)
    
    # 训练模型
    train_main(['--config', args.config])


if __name__ == '__main__':
    main() 