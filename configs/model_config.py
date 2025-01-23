"""Model configuration for SC2 Mutations Scoring."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    """Configuration for embedding layers."""
    
    # Embedding dimensions
    map_embed_dim: int = 64
    commander_embed_dim: int = 128
    mutation_embed_dim: int = 96
    ai_embed_dim: int = 32
    
    # Vocabulary sizes (will be updated during training)
    map_vocab_size: int = 0
    commander_vocab_size: int = 0
    mutation_vocab_size: int = 0
    ai_vocab_size: int = 0
    
    # Dropout rates
    embed_dropout: float = 0.1


@dataclass
class MLPConfig:
    """Configuration for MLP layers."""
    
    hidden_dims: List[int] = (512, 256, 128)
    dropout: float = 0.2
    activation: str = "relu"
    use_batch_norm: bool = True
    num_classes: int = 5  # 1-5分难度评分


@dataclass
class ModelConfig:
    """Main model configuration."""
    
    # Sub-configs
    embedding: EmbeddingConfig = EmbeddingConfig()
    mlp: MLPConfig = MLPConfig()
    
    # Feature fusion
    fusion_method: str = "concat"  # or "sum"
    use_attention: bool = True
    attention_heads: int = 4
    
    # Regularization
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    
    # Loss function
    loss_type: str = "cross_entropy"  # or "mse"
    class_weights: Optional[List[float]] = None
    
    def update_vocab_sizes(self,
                          map_vocab_size: int,
                          commander_vocab_size: int,
                          mutation_vocab_size: int,
                          ai_vocab_size: int) -> None:
        """Update vocabulary sizes based on data."""
        self.embedding.map_vocab_size = map_vocab_size
        self.embedding.commander_vocab_size = commander_vocab_size
        self.embedding.mutation_vocab_size = mutation_vocab_size
        self.embedding.ai_vocab_size = ai_vocab_size


# Default configuration
DEFAULT_CONFIG = ModelConfig() 