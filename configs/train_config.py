"""Training configuration for SC2 Mutations Scoring."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    
    # Data paths
    train_data_path: str = "data/processed/sc2_mutations_duo.csv"
    val_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    
    # Data split
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = False
    commander_swap_prob: float = 0.5  # 交换指挥官顺序的概率


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    
    optimizer_type: str = "adam"
    learning_rate: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    momentum: float = 0.9  # for SGD
    
    # Learning rate scheduler
    scheduler_type: str = "cosine"  # or "step", "plateau"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # For StepLR
    step_size: int = 30
    gamma: float = 0.1
    
    # For ReduceLROnPlateau
    patience: int = 10
    factor: float = 0.1


@dataclass
class TrainingConfig:
    """Main training configuration."""
    
    # Basic training settings
    max_epochs: int = 100
    early_stopping_patience: int = 15
    grad_clip_val: float = 1.0
    
    # Device settings
    device: str = "cuda"
    precision: str = "float32"  # or "float16", "bfloat16"
    
    # Logging and checkpoints
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    save_freq: int = 5
    log_freq: int = 100
    
    # Validation
    val_freq: int = 1  # epochs
    val_metrics: Tuple[str, ...] = ("accuracy", "f1", "confusion_matrix")
    
    # Sub-configs
    data: DataConfig = DataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.data.val_data_path is None and self.data.val_size <= 0:
            raise ValueError("Must either provide val_data_path or set val_size > 0")


# Default configuration
DEFAULT_CONFIG = TrainingConfig() 