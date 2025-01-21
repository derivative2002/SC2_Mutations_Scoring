import os
import torch
import pandas as pd
import argparse
from typing import List, Tuple
from data.preprocess import prepare_data
from models.gnn_attention import MutationGNN
from config.default_config import ModelConfig, PathConfig, DataConfig
from utils.logger import PredictionLogger

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default=DataConfig.train_data_path,
                      help='数据路径')
    parser.add_argument('--num_samples', type=int, default=5,
                      help='预测样本数量')
    
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

def load_model(model_path: str, num_factors: int, num_maps: int, device: torch.device) -> MutationGNN:
    """
    加载训练好的模型
    """
    model = MutationGNN(
        num_factors=num_factors,
        num_maps=num_maps,
        hidden_dim=ModelConfig.hidden_dim,
        num_gnn_layers=ModelConfig.num_gnn_layers,
        num_classes=ModelConfig.num_classes,
        dropout=0.0  # 预测时不使用dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_single(
    model: MutationGNN,
    data,
    device: torch.device
) -> int:
    """
    对单个突变进行预测
    返回预测的评级（1-10）
    """
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        out = model(data.x, data.edge_index, data.map_id, None)
        pred = out.argmax(dim=1).item()
        return pred + 1  # 转换回1-10的评级

def predict_batch(
    model: MutationGNN,
    data_list: List,
    device: torch.device
) -> List[int]:
    """
    批量预测
    返回预测评级列表
    """
    predictions = []
    for data in data_list:
        pred = predict_single(model, data, device)
        predictions.append(pred)
    return predictions

def save_predictions(predictions: List[int], data_list: List, save_path: str):
    """
    保存预测结果
    """
    results = []
    for i, (pred, data) in enumerate(zip(predictions, data_list)):
        results.append({
            'Sample': i + 1,
            'Predicted Rating': pred,
            'True Rating': data.y.item() + 1  # 转换回1-10的评级
        })
    
    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)

def main():
    # 1. 解析命令行参数
    args = parse_args()
    
    # 2. 创建日志记录器
    logger = PredictionLogger(args.log_dir)
    
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
    
    logger.log_prediction_start(args.model_path, args.num_samples)
    model = load_model(args.model_path, num_factors, num_maps, device)
    logger.logger.info("Model loaded successfully!")
    
    # 7. 进行预测
    predictions = predict_batch(
        model,
        data_list[:args.num_samples],
        device
    )
    
    # 8. 保存预测结果
    predictions_path = os.path.join(args.output_dir, 'predictions.csv')
    save_predictions(
        predictions,
        data_list[:args.num_samples],
        predictions_path
    )
    logger.log_predictions_saved(predictions_path)
    
    # 9. 记录预测结果
    logger.logger.info("\nPrediction Results:")
    logger.logger.info("-" * 50)
    for i, pred in enumerate(predictions):
        true_rating = data_list[i].y.item() + 1
        logger.log_prediction(i + 1, pred, true_rating)

if __name__ == '__main__':
    main() 