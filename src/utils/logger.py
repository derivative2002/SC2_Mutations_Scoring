import os
import logging
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str,
    log_dir: str,
    filename: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志保存目录
        filename: 日志文件名，如果为None则自动生成
        level: 日志级别
    
    Returns:
        logger: 日志记录器
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果没有指定文件名，使用时间戳创建
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{name}_{timestamp}.log'
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已经有处理器，不重复添加
    if logger.handlers:
        return logger
    
    # 创建文件处理器
    file_handler = logging.FileHandler(
        os.path.join(log_dir, filename),
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 设置格式器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class TrainingLogger:
    """训练日志记录器"""
    def __init__(self, log_dir: str):
        self.logger = setup_logger('training', log_dir)
        self.epoch = 0
        self.best_val_acc = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, val_acc: float):
        """记录每个epoch的训练信息"""
        self.epoch = epoch
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
        
        self.logger.info(
            f"Epoch [{epoch}] - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f} - "
            f"Best Val Acc: {self.best_val_acc:.4f}"
        )
    
    def log_early_stopping(self, epoch: int):
        """记录早停信息"""
        self.logger.info(f"Early stopping triggered after {epoch} epochs")
    
    def log_lr_update(self, new_lr: float):
        """记录学习率更新信息"""
        self.logger.info(f"Learning rate updated to {new_lr:.6f}")
    
    def log_model_save(self, path: str):
        """记录模型保存信息"""
        self.logger.info(f"Model saved to {path}")
    
    def log_training_complete(self):
        """记录训练完成信息"""
        self.logger.info(
            f"Training completed! "
            f"Best validation accuracy: {self.best_val_acc:.4f}"
        )

class EvaluationLogger:
    """评估日志记录器"""
    def __init__(self, log_dir: str):
        self.logger = setup_logger('evaluation', log_dir)
    
    def log_evaluation_start(self, model_path: str):
        """记录评估开始信息"""
        self.logger.info(f"Starting evaluation using model from {model_path}")
    
    def log_metrics(self, accuracy: float, report: str):
        """记录评估指标"""
        self.logger.info(f"Overall Accuracy: {accuracy:.4f}")
        self.logger.info(f"\nClassification Report:\n{report}")
    
    def log_error_analysis(self, error_margins: dict, map_errors: dict):
        """记录错误分析"""
        self.logger.info("\nError Distribution:")
        for margin, count in error_margins.items():
            self.logger.info(f"Error Margin {margin}: {count} cases")
        
        self.logger.info("\nErrors by Map:")
        for map_name, error_count in map_errors.items():
            self.logger.info(f"{map_name}: {error_count} errors")

class PredictionLogger:
    """预测日志记录器"""
    def __init__(self, log_dir: str):
        self.logger = setup_logger('prediction', log_dir)
    
    def log_prediction_start(self, model_path: str, num_samples: int):
        """记录预测开始信息"""
        self.logger.info(
            f"Starting prediction for {num_samples} samples "
            f"using model from {model_path}"
        )
    
    def log_prediction(self, sample_id: int, pred: int, true: int):
        """记录单个预测结果"""
        self.logger.info(
            f"Sample {sample_id}: "
            f"Predicted = {pred}, "
            f"True = {true}, "
            f"Difference = {abs(pred - true)}"
        )
    
    def log_predictions_saved(self, path: str):
        """记录预测结果保存信息"""
        self.logger.info(f"Predictions saved to {path}") 