"""
默认配置文件
"""

class ModelConfig:
    """模型配置"""
    hidden_dim = 256
    num_gnn_layers = 4
    num_classes = 10
    dropout = 0.2
    num_heads = 8
    attention_dropout = 0.1

class TrainingConfig:
    """训练配置"""
    batch_size = 16
    num_epochs = 200
    learning_rate = 2e-4
    weight_decay = 1e-5
    lr_reduce_factor = 0.7
    lr_reduce_patience = 8
    min_lr = 1e-6
    warmup_epochs = 10
    early_stopping_patience = 20
    grad_clip = 0.5
    label_smoothing = 0.15

class DataConfig:
    """数据配置"""
    train_data_path = 'data/raw/train.csv'
    val_split = 0.2
    random_seed = 42
    num_workers = 4

class PathConfig:
    """路径配置"""
    model_dir = 'model'
    best_model_path = 'model/best_model.pth'
    checkpoint_dir = 'model/checkpoints'
    log_dir = 'logs'
    results_dir = 'model/results' 