"""Neural network models for SC2 Mutations Scoring."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from src.models.embeddings import FeatureEmbedding


class MLP(nn.Module):
    """Multi-layer perceptron network."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        """Initialize MLP.
        
        Args:
            input_dim: Input dimension.
            hidden_dims: List of hidden layer dimensions.
            output_dim: Output dimension.
            dropout: Dropout rate.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.layers(x)


class MutationScorer(nn.Module):
    """Main model for mutation difficulty scoring."""
    
    def __init__(self,
                 num_maps: int,
                 num_commanders: int,
                 num_mutations: int,
                 num_ais: int,
                 map_dim: int = 64,
                 commander_dim: int = 128,
                 mutation_dim: int = 96,
                 ai_dim: int = 32,
                 hidden_dims: List[int] = (512, 256, 128),
                 num_classes: int = 5,
                 dropout: float = 0.2,
                 embed_dropout: float = 0.1,
                 use_batch_norm: bool = True):
        """Initialize mutation scorer.
        
        Args:
            num_maps: Number of unique maps.
            num_commanders: Number of unique commanders.
            num_mutations: Number of unique mutation factors.
            num_ais: Number of unique AI types.
            map_dim: Map embedding dimension.
            commander_dim: Commander embedding dimension.
            mutation_dim: Mutation embedding dimension.
            ai_dim: AI embedding dimension.
            hidden_dims: List of hidden layer dimensions.
            num_classes: Number of difficulty classes (1-5).
            dropout: Dropout rate for MLP.
            embed_dropout: Dropout rate for embeddings.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        # Feature embedding
        self.embedding = FeatureEmbedding(
            num_maps=num_maps,
            num_commanders=num_commanders,
            num_mutations=num_mutations,
            num_ais=num_ais,
            map_dim=map_dim,
            commander_dim=commander_dim,
            mutation_dim=mutation_dim,
            ai_dim=ai_dim,
            dropout=embed_dropout
        )
        
        # MLP classifier
        self.classifier = MLP(
            input_dim=self.embedding.output_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
        
    def forward(self,
                map_ids: torch.Tensor,
                commander_ids: torch.Tensor,
                mutation_ids: torch.Tensor,
                ai_ids: torch.Tensor,
                mutation_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            map_ids: Map indices of shape (batch_size,)
            commander_ids: Commander indices of shape (batch_size, 2)
            mutation_ids: Mutation indices of shape (batch_size, num_mutations)
            ai_ids: AI indices of shape (batch_size,)
            mutation_mask: Optional boolean mask for mutations
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Get combined feature embedding
        x = self.embedding(
            map_ids, commander_ids, mutation_ids, ai_ids, mutation_mask)
        
        # Classify
        return self.classifier(x)
    
    def predict(self,
               map_ids: torch.Tensor,
               commander_ids: torch.Tensor,
               mutation_ids: torch.Tensor,
               ai_ids: torch.Tensor,
               mutation_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make prediction.
        
        Args:
            map_ids: Map indices of shape (batch_size,)
            commander_ids: Commander indices of shape (batch_size, 2)
            mutation_ids: Mutation indices of shape (batch_size, num_mutations)
            ai_ids: AI indices of shape (batch_size,)
            mutation_mask: Optional boolean mask for mutations
            
        Returns:
            Tuple of:
                - Predicted class indices of shape (batch_size,)
                - Prediction probabilities of shape (batch_size, num_classes)
        """
        logits = self.forward(
            map_ids, commander_ids, mutation_ids, ai_ids, mutation_mask)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs 