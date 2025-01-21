import os
import torch
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import confusion_matrix, classification_report
from torch_geometric.loader import DataLoader as GraphDataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

from data.preprocess import prepare_data
from models.gnn_attention import MutationGNN, MutationDataset
from predict import load_model
from config.default_config import ModelConfig, PathConfig, DataConfig
from utils.logger import EvaluationLogger

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='评估突变评级预测模型')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=DataConfig.train_data_path,
                      help='数据路径')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批次大小')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default=PathConfig.best_model_path,
                      help='模型路径')
    parser.add_argument('--hidden_dim', type=int, default=ModelConfig.hidden_dim,
                      help='隐藏层维度')
    parser.add_argument('--num_gnn_layers', type=int, default=ModelConfig.num_gnn_layers,
                      help='GNN层数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=PathConfig.results_dir,
                      help='结果输出目录')
    parser.add_argument('--log_dir', type=str, default=PathConfig.log_dir,
                      help='日志保存目录')
    
    return parser.parse_args()

def evaluate_model(
    model: MutationGNN,
    data_loader: GraphDataLoader,
    device: torch.device
) -> Tuple[List[int], List[int], float]:
    """
    评估模型性能
    返回真实标签、预测标签和准确率
    """
    model.eval()
    predictions = []
    true_labels = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.map_id, data.batch)
            pred = out.argmax(dim=1)
            
            # 收集预测和真实标签
            predictions.extend((pred + 1).cpu().numpy())  # 转换回1-10的评级
            true_labels.extend((data.y + 1).cpu().numpy())
            
            # 计算准确率
            correct += (pred == data.y).sum().item()
            total += data.num_graphs
    
    accuracy = correct / total
    return true_labels, predictions, accuracy

def plot_confusion_matrix(true_labels: List[int], predictions: List[int], save_path: str):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

def analyze_errors(
    true_labels: List[int],
    predictions: List[int],
    data_list: List,
    map_to_id: Dict[str, int]
) -> pd.DataFrame:
    """
    分析预测错误的案例
    """
    # 创建反向映射
    id_to_map = {v: k for k, v in map_to_id.items()}
    
    # 收集错误案例
    errors = []
    for i, (true, pred) in enumerate(zip(true_labels, predictions)):
        if true != pred:
            data = data_list[i]
            map_id = data.map_id.item()
            errors.append({
                'True Rating': true,
                'Predicted Rating': pred,
                'Error Margin': abs(true - pred),
                'Map': id_to_map[map_id]
            })
    
    return pd.DataFrame(errors)

def main():
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 创建日志记录器
    logger = EvaluationLogger(args.log_dir)
    
    # 3. 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.logger.info(f"Using device: {device}")
    
    # 4. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. 加载数据
    logger.logger.info("Loading data...")
    data_list, map_to_id = prepare_data(args.data_path)
    
    # 6. 加载模型
    num_factors = data_list[0].x.size(1)
    num_maps = len(map_to_id)
    
    logger.log_evaluation_start(args.model_path)
    model = load_model(
        args.model_path,
        num_factors,
        num_maps,
        device
    )
    logger.logger.info("Model loaded successfully!")
    
    # 7. 创建数据加载器
    dataset = MutationDataset(data_list)
    data_loader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # 8. 评估模型
    logger.logger.info("Starting evaluation...")
    true_labels, predictions, accuracy = evaluate_model(model, data_loader, device)
    
    # 9. 记录评估结果
    report = classification_report(true_labels, predictions)
    logger.log_metrics(accuracy, report)
    
    # 10. 绘制混淆矩阵
    confusion_matrix_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(true_labels, predictions, confusion_matrix_path)
    logger.logger.info(f"Confusion matrix has been saved to '{confusion_matrix_path}'")
    
    # 11. 分析错误案例
    error_analysis = analyze_errors(true_labels, predictions, data_list, map_to_id)
    error_analysis_path = os.path.join(args.output_dir, 'error_analysis.csv')
    error_analysis.to_csv(error_analysis_path, index=False)
    logger.logger.info(f"Error analysis has been saved to '{error_analysis_path}'")
    
    # 12. 记录错误统计
    error_margins = error_analysis['Error Margin'].value_counts().sort_index().to_dict()
    map_errors = error_analysis['Map'].value_counts().to_dict()
    logger.log_error_analysis(error_margins, map_errors)

if __name__ == '__main__':
    main() 