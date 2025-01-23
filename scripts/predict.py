"""推理脚本."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.data.preprocess import Vocab
from src.models.networks import MutationScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_vocabs(vocab_dir: str) -> Dict[str, Vocab]:
    """加载词表.
    
    Args:
        vocab_dir: 词表目录
        
    Returns:
        词表字典
    """
    vocabs = {}
    for name in ['map', 'commander', 'mutation', 'ai']:
        vocabs[name] = Vocab.load(vocab_dir, name)
    return vocabs


def load_model(
    checkpoint_path: str,
    vocab_sizes: Dict[str, int],
    map_dim: int = 64,
    commander_dim: int = 128,
    mutation_dim: int = 96,
    ai_dim: int = 32,
    hidden_dims: List[int] = [512, 256, 128],
    device: torch.device = None
) -> MutationScorer:
    """加载模型.
    
    Args:
        checkpoint_path: checkpoint文件路径
        vocab_sizes: 词表大小字典
        map_dim: 地图嵌入维度
        commander_dim: 指挥官嵌入维度
        mutation_dim: 突变因子嵌入维度
        ai_dim: AI嵌入维度
        hidden_dims: MLP隐藏层维度
        device: 设备
        
    Returns:
        加载的模型
    """
    # 创建模型
    model = MutationScorer(
        num_maps=vocab_sizes['map'],
        num_commanders=vocab_sizes['commander'],
        num_mutations=vocab_sizes['mutation'],
        num_ais=vocab_sizes['ai'],
        map_dim=map_dim,
        commander_dim=commander_dim,
        mutation_dim=mutation_dim,
        ai_dim=ai_dim,
        hidden_dims=hidden_dims
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"加载模型: {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}")
    logger.info(f"指标: {checkpoint['metrics']}")
    
    return model


def preprocess_data(
    df: pd.DataFrame,
    vocabs: Dict[str, Vocab],
    max_mutations: int = 10
) -> Dict[str, torch.Tensor]:
    """预处理数据.
    
    Args:
        df: 输入数据
        vocabs: 词表字典
        max_mutations: 最大突变因子数量
        
    Returns:
        特征字典
    """
    num_samples = len(df)
    features = {
        'map_ids': torch.zeros(num_samples, dtype=torch.long),
        'commander_ids': torch.zeros((num_samples, 2), dtype=torch.long),
        'mutation_ids': torch.zeros(
            (num_samples, max_mutations), dtype=torch.long),
        'mutation_mask': torch.zeros(
            (num_samples, max_mutations), dtype=torch.float),
        'ai_ids': torch.zeros(num_samples, dtype=torch.long)
    }
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="预处理数据"):
        # 地图ID
        features['map_ids'][i] = vocabs['map'].token2idx[row['map_name']]
        
        # 指挥官ID
        commanders = eval(row['commanders']) if isinstance(
            row['commanders'], str) else row['commanders']
        for j, cmd in enumerate(commanders[:2]):
            if cmd:
                features['commander_ids'][i, j] = (
                    vocabs['commander'].token2idx[cmd])
        
        # 突变因子ID和mask
        mutations = eval(row['mutation_factors']) if isinstance(
            row['mutation_factors'], str) else row['mutation_factors']
        for j, mutation in enumerate(mutations[:max_mutations]):
            features['mutation_ids'][i, j] = (
                vocabs['mutation'].token2idx[mutation])
            features['mutation_mask'][i, j] = 1.0
        
        # AI ID
        features['ai_ids'][i] = vocabs['ai'].token2idx[row['enemy_ai']]
    
    return features


@torch.no_grad()
def predict(
    model: MutationScorer,
    features: Dict[str, torch.Tensor],
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """模型预测.
    
    Args:
        model: 模型
        features: 特征字典
        device: 设备
        
    Returns:
        预测类别和概率的元组
    """
    # 将数据移到设备
    features = {k: v.to(device) for k, v in features.items()}
    
    # 模型预测
    preds, probs = model.predict(
        map_ids=features['map_ids'],
        commander_ids=features['commander_ids'],
        mutation_ids=features['mutation_ids'],
        ai_ids=features['ai_ids'],
        mutation_mask=features['mutation_mask']
    )
    
    return preds.cpu().numpy(), probs.cpu().numpy()


def save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    output_path: str
):
    """保存预测结果.
    
    Args:
        df: 输入数据
        preds: 预测类别
        probs: 预测概率
        output_path: 输出文件路径
    """
    # 添加预测结果
    df['predicted_score'] = preds + 1  # 转换回1-5
    for i in range(5):
        df[f'prob_score_{i+1}'] = probs[:, i]
    
    # 保存结果
    df.to_csv(output_path, index=False)
    logger.info(f"预测结果已保存到: {output_path}")


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description='SC2突变难度预测')
    
    # 数据参数
    parser.add_argument('--input_file', type=str, required=True,
                       help='输入数据文件路径')
    parser.add_argument('--output_file', type=str, required=True,
                       help='输出文件路径')
    parser.add_argument('--vocab_dir', type=str, required=True,
                       help='词表目录')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--map_dim', type=int, default=64,
                       help='地图嵌入维度')
    parser.add_argument('--commander_dim', type=int, default=128,
                       help='指挥官嵌入维度')
    parser.add_argument('--mutation_dim', type=int, default=96,
                       help='突变因子嵌入维度')
    parser.add_argument('--ai_dim', type=int, default=32,
                       help='AI嵌入维度')
    parser.add_argument('--hidden_dims', type=int, nargs='+',
                       default=[512, 256, 128],
                       help='MLP隐藏层维度')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    parser.add_argument('--max_mutations', type=int, default=10,
                       help='最大突变因子数量')
    
    args = parser.parse_args()
    return args


def main():
    """主函数."""
    # 解析参数
    args = parse_args()
    logger.info(f"参数配置:\n{vars(args)}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载词表
    vocabs = load_vocabs(args.vocab_dir)
    vocab_sizes = {name: len(vocab) for name, vocab in vocabs.items()}
    logger.info(f"词表大小: {vocab_sizes}")
    
    # 加载模型
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_sizes=vocab_sizes,
        map_dim=args.map_dim,
        commander_dim=args.commander_dim,
        mutation_dim=args.mutation_dim,
        ai_dim=args.ai_dim,
        hidden_dims=args.hidden_dims,
        device=device
    )
    
    # 读取数据
    logger.info(f"读取数据: {args.input_file}")
    df = pd.read_csv(args.input_file)
    logger.info(f"数据样本数: {len(df)}")
    
    # 预处理数据
    features = preprocess_data(
        df=df,
        vocabs=vocabs,
        max_mutations=args.max_mutations
    )
    
    # 模型预测
    logger.info("开始预测...")
    preds, probs = predict(model, features, device)
    
    # 保存结果
    save_predictions(df, preds, probs, args.output_file)


if __name__ == '__main__':
    main() 