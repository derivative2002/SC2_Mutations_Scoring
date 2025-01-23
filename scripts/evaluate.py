"""测试脚本."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from tqdm import tqdm

from src.data.dataset import SC2MutationDataset
from src.models.networks import MutationScorer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


@torch.no_grad()
def evaluate(
    model: MutationScorer,
    dataset: SC2MutationDataset,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """评估模型.
    
    Args:
        model: 模型
        dataset: 数据集
        device: 设备
        batch_size: 批次大小
        
    Returns:
        指标字典、预测数组和标签数组的元组
    """
    all_preds = []
    all_labels = []
    
    # 批次迭代
    for i in tqdm(range(0, len(dataset), batch_size), desc="评估"):
        # 获取批次数据
        batch_data = [dataset[j] for j in range(
            i, min(i + batch_size, len(dataset)))]
        
        # 合并批次
        batch = {
            k: torch.stack([d[k] for d in batch_data])
            for k in batch_data[0].keys()
        }
        
        # 将数据移到设备
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # 模型预测
        preds, _ = model.predict(
            map_ids=batch['map_ids'],
            commander_ids=batch['commander_ids'],
            mutation_ids=batch['mutation_ids'],
            ai_ids=batch['ai_ids'],
            mutation_mask=batch['mutation_mask']
        )
        
        # 收集预测和标签
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
    
    # 转换为数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    metrics = compute_metrics(all_preds, all_labels)
    
    return metrics, all_preds, all_labels


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """计算评估指标.
    
    Args:
        preds: 预测数组
        labels: 标签数组
        
    Returns:
        指标字典
    """
    # 计算准确率
    acc = accuracy_score(labels, preds)
    
    # 计算每个类别的精确率、召回率和F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None)
    
    # 计算宏平均和加权平均
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted')
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    metrics = {
        'accuracy': acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # 添加每个类别的指标
    for i in range(5):  # 5个难度等级
        metrics[f'class_{i+1}_precision'] = precision[i]
        metrics[f'class_{i+1}_recall'] = recall[i]
        metrics[f'class_{i+1}_f1'] = f1[i]
    
    return metrics


def save_results(
    metrics: Dict[str, float],
    preds: np.ndarray,
    labels: np.ndarray,
    output_dir: str
):
    """保存评估结果.
    
    Args:
        metrics: 指标字典
        preds: 预测数组
        labels: 标签数组
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存指标
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"指标已保存到: {metrics_path}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'true_score': labels + 1,  # 转换回1-5
        'predicted_score': preds + 1
    })
    results_path = output_dir / 'predictions.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"预测结果已保存到: {results_path}")
    
    # 保存混淆矩阵
    cm = confusion_matrix(labels, preds)
    cm_path = output_dir / 'confusion_matrix.csv'
    cm_df = pd.DataFrame(
        cm,
        index=[f'True_{i+1}' for i in range(5)],
        columns=[f'Pred_{i+1}' for i in range(5)]
    )
    cm_df.to_csv(cm_path)
    logger.info(f"混淆矩阵已保存到: {cm_path}")


def parse_args():
    """解析命令行参数."""
    parser = argparse.ArgumentParser(description='评估SC2突变难度预测模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录')
    
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
    
    # 评估参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备')
    
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
    
    # 加载测试集
    test_dataset = SC2MutationDataset(
        data_dir=args.data_dir,
        split='test'
    )
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 加载模型
    model = load_model(
        checkpoint_path=args.checkpoint,
        vocab_sizes=test_dataset.vocab_sizes,
        map_dim=args.map_dim,
        commander_dim=args.commander_dim,
        mutation_dim=args.mutation_dim,
        ai_dim=args.ai_dim,
        hidden_dims=args.hidden_dims,
        device=device
    )
    
    # 评估模型
    logger.info("开始评估...")
    metrics, preds, labels = evaluate(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=args.batch_size
    )
    
    # 打印主要指标
    logger.info("\n评估结果:")
    logger.info(f"准确率: {metrics['accuracy']:.4f}")
    logger.info(f"宏平均F1: {metrics['macro_f1']:.4f}")
    logger.info(f"加权平均F1: {metrics['weighted_f1']:.4f}")
    
    # 保存结果
    save_results(metrics, preds, labels, args.output_dir)


if __name__ == '__main__':
    main() 