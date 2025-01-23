"""运行脚本,用于执行完整的训练流程."""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch

from src.data.preprocess import SC2MutationPreprocessor
from src.utils.config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """设置随机种子以确保可重复性."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def preprocess_data(config):
    """预处理数据."""
    logger.info("开始数据预处理...")
    
    preprocessor = SC2MutationPreprocessor(
        raw_data_path=config.data.raw_data_path,
        processed_dir=config.data.processed_dir,
        max_mutations=config.data.max_mutations
    )
    preprocessor.process()
    logger.info("数据预处理完成!")


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description='运行SC2突变难度预测模型训练')
    
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径')
    parser.add_argument('--skip_preprocess', action='store_true',
                       help='是否跳过数据预处理')
    
    return parser.parse_args()


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
    experiment_dir = Path(config.experiment.save_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_save_path = experiment_dir / 'config.yaml'
    config.to_yaml(config_save_path)
    logger.info(f"保存配置到: {config_save_path}")
    
    # 数据预处理
    if not args.skip_preprocess:
        preprocess_data(config)
    
    # 训练模型
    from scripts.train import main as train_main
    train_main(['--config', str(args.config)])
    
    logger.info("训练流程完成!")


if __name__ == '__main__':
    main() 