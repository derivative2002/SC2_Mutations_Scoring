"""
日志工具模块
"""
import os
import logging
from datetime import datetime
from typing import Optional


class BaseLogger:
    """基础日志记录器"""
    
    def __init__(self, name: str, log_dir: str, filename: Optional[str] = None):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志保存目录
            filename: 日志文件名，如果为None则自动生成
        """
        self.name = name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{name}_{timestamp}.log'
            
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # 文件处理器
            file_handler = logging.FileHandler(
                os.path.join(log_dir, filename),
                encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)


class TrainingLogger(BaseLogger):
    """训练日志记录器"""
    
    def __init__(self, log_dir: str):
        super().__init__('训练', log_dir)
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
            f"轮次 [{epoch}] - "
            f"训练损失: {train_loss:.4f} - "
            f"验证损失: {val_loss:.4f} - "
            f"验证准确率: {val_acc:.4f} - "
            f"最佳验证准确率: {self.best_val_acc:.4f}"
        )
    
    def log_early_stopping(self, epoch: int):
        """记录早停信息"""
        self.logger.info(f"在第 {epoch} 轮触发早停")
    
    def log_lr_update(self, new_lr: float):
        """记录学习率更新信息"""
        self.logger.info(f"学习率更新为 {new_lr:.6f}")
    
    def log_model_save(self, path: str):
        """记录模型保存信息"""
        self.logger.info(f"模型已保存至 {path}")
    
    def log_training_complete(self):
        """记录训练完成信息"""
        self.logger.info(
            f"训练完成！"
            f"最佳验证准确率: {self.best_val_acc:.4f}"
        )


class EvaluationLogger(BaseLogger):
    """评估日志记录器"""
    
    def __init__(self, log_dir: str):
        super().__init__('评估', log_dir)
    
    def log_evaluation_start(self, model_path: str):
        """记录评估开始信息"""
        self.logger.info(f"开始使用模型 {model_path} 进行评估")
    
    def log_metrics(self, accuracy: float, report: str):
        """记录评估指标"""
        self.logger.info(f"总体准确率: {accuracy:.4f}")
        self.logger.info(f"\n分类报告:\n{report}")
    
    def log_error_analysis(self, error_margins: dict, map_errors: dict):
        """记录错误分析"""
        self.logger.info("\n错误分布:")
        for margin, count in error_margins.items():
            self.logger.info(f"误差范围 {margin}: {count} 个案例")
        
        self.logger.info("\n地图错误统计:")
        for map_name, error_count in map_errors.items():
            self.logger.info(f"{map_name}: {error_count} 个错误")


class PredictionLogger(BaseLogger):
    """预测日志记录器"""
    
    def __init__(self, log_dir: str):
        super().__init__('预测', log_dir)
    
    def log_prediction_start(self, model_path: str, num_samples: int):
        """记录预测开始信息"""
        self.logger.info(
            f"开始使用模型 {model_path} "
            f"对 {num_samples} 个样本进行预测"
        )
    
    def log_prediction(self, sample_id: int, pred: int, true: int):
        """记录单个预测结果"""
        self.logger.info(
            f"样本 {sample_id}: "
            f"预测值 = {pred}, "
            f"真实值 = {true}, "
            f"差异 = {abs(pred - true)}"
        )
    
    def log_predictions_saved(self, path: str):
        """记录预测结果保存信息"""
        self.logger.info(f"预测结果已保存至 {path}") 