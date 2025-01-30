"""数据预处理脚本."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from scoring.src.data.preprocess import SC2MutationPreprocessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数."""
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='原始数据目录')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='处理后数据保存目录')
    parser.add_argument('--max_mutations', type=int, default=8,
                       help='最大突变因子数量')
    args = parser.parse_args()
    
    # 读取原始数据
    raw_dir = Path(args.raw_dir)
    train_data = pd.read_csv(
        raw_dir / '【实验数据】mutation_tasks_初版生成数据.csv',
        encoding='utf-8'
    )
    val_data = pd.read_csv(
        raw_dir / '【实验数据】高玩实际测试的高质量数据_评测集.csv',
        encoding='utf-8'
    )
    
    # 合并数据
    val_data['is_verified'] = 1  # 标记验证数据
    train_data['is_verified'] = 0  # 标记训练数据
    df = pd.concat([train_data, val_data], ignore_index=True)
    
    # 创建预处理器
    preprocessor = SC2MutationPreprocessor(
        processed_dir=args.processed_dir,
        max_mutations=args.max_mutations
    )
    
    # 处理数据
    preprocessor.process(df)
    logger.info("数据预处理完成!")


if __name__ == '__main__':
    main() 