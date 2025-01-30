"""Feature embedding layers for SC2 Mutations Scoring."""

from typing import List, Optional

import torch
import torch.nn as nn


class MapEmbedding(nn.Module):
    """Embedding layer for map features."""

    def __init__(self, num_maps: int, embed_dim: int, dropout: float = 0.2):
        """Initialize map embedding layer.
        
        Args:
            num_maps: Number of unique maps.
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_maps, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Map indices tensor of shape (batch_size,)
            
        Returns:
            Embedded tensor of shape (batch_size, embed_dim)
        """
        x = self.embedding(x)
        return self.dropout(x)


class CommanderEmbedding(nn.Module):
    """Simplified embedding layer for commander pair features."""
    
    def __init__(self, 
                 num_commanders: int, 
                 embed_dim: int,
                 dropout: float = 0.2):
        """Initialize commander embedding layer.
        
        Args:
            num_commanders: Number of unique commanders.
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_commanders, embed_dim)
        self.combine = nn.Linear(2 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Commander indices tensor of shape (batch_size, 2)
            
        Returns:
            Embedded tensor of shape (batch_size, embed_dim)
        """
        # 基础embedding
        embedded = self.embedding(x)  # (batch_size, 2, embed_dim)
        # 简单连接两个指挥官的embedding
        combined = torch.cat([embedded[:, 0], embedded[:, 1]], dim=-1)
        # 线性变换到目标维度
        output = self.combine(combined)
        return self.dropout(output)


class MutationEmbedding(nn.Module):
    """Simplified embedding layer for mutation factor features."""
    
    def __init__(self, 
                 num_mutations: int, 
                 embed_dim: int,
                 dropout: float = 0.2):
        """Initialize mutation embedding layer.
        
        Args:
            num_mutations: Number of unique mutation factors.
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_mutations, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Mutation indices tensor of shape (batch_size, num_mutations)
            mask: Optional boolean mask of shape (batch_size, num_mutations)
            
        Returns:
            Embedded tensor of shape (batch_size, embed_dim)
        """
        # 基础embedding
        embedded = self.embedding(x)  # (batch_size, num_mutations, embed_dim)
        
        if mask is not None:
            embedded = embedded * mask.unsqueeze(-1)
            # 加权平均
            output = embedded.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-9)
        else:
            # 简单平均
            output = embedded.mean(dim=1)
            
        return self.dropout(output)


class AIEmbedding(nn.Module):
    """Embedding layer for enemy AI features."""
    
    def __init__(self, num_ais: int, embed_dim: int, dropout: float = 0.2):
        """Initialize AI embedding layer.
        
        Args:
            num_ais: Number of unique AI types.
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_ais, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: AI indices tensor of shape (batch_size,)
            
        Returns:
            Embedded tensor of shape (batch_size, embed_dim)
        """
        x = self.embedding(x)
        return self.dropout(x)


class FeatureEmbedding(nn.Module):
    """Simplified combined embedding layer for all features."""
    
    def __init__(self,
                 num_maps: int,
                 num_commanders: int,
                 num_mutations: int,
                 num_ais: int,
                 map_dim: int = 32,
                 commander_dim: int = 48,
                 mutation_dim: int = 48,
                 ai_dim: int = 16,
                 dropout: float = 0.2):
        """Initialize feature embedding layer.
        
        Args:
            num_maps: Number of unique maps.
            num_commanders: Number of unique commanders.
            num_mutations: Number of unique mutation factors.
            num_ais: Number of unique AI types.
            map_dim: Map embedding dimension.
            commander_dim: Commander embedding dimension.
            mutation_dim: Mutation embedding dimension.
            ai_dim: AI embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        
        self.map_embedding = MapEmbedding(num_maps, map_dim, dropout)
        self.commander_embedding = CommanderEmbedding(
            num_commanders, commander_dim, dropout)
        self.mutation_embedding = MutationEmbedding(
            num_mutations, mutation_dim, dropout)
        self.ai_embedding = AIEmbedding(num_ais, ai_dim, dropout)
        
        total_dim = map_dim + commander_dim + mutation_dim + ai_dim
        self.output_dim = total_dim
        
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
            Combined embedding tensor of shape (batch_size, output_dim)
        """
        map_embed = self.map_embedding(map_ids)
        commander_embed = self.commander_embedding(commander_ids)
        mutation_embed = self.mutation_embedding(mutation_ids, mutation_mask)
        ai_embed = self.ai_embedding(ai_ids)
        
        # 简单连接所有特征
        return torch.cat([
            map_embed,
            commander_embed,
            mutation_embed,
            ai_embed
        ], dim=-1) 