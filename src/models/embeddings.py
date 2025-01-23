"""Feature embedding layers for SC2 Mutations Scoring."""

from typing import List, Optional

import torch
import torch.nn as nn


class MapEmbedding(nn.Module):
    """Embedding layer for map features."""

    def __init__(self, num_maps: int, embed_dim: int, dropout: float = 0.1):
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
    """Embedding layer for commander pair features."""
    
    def __init__(self, 
                 num_commanders: int, 
                 embed_dim: int,
                 dropout: float = 0.1):
        """Initialize commander embedding layer.
        
        Args:
            num_commanders: Number of unique commanders.
            embed_dim: Embedding dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding = nn.Embedding(num_commanders, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Commander indices tensor of shape (batch_size, 2)
            
        Returns:
            Embedded tensor of shape (batch_size, 2 * embed_dim)
        """
        # Embed each commander
        x = self.embedding(x)  # (batch_size, 2, embed_dim)
        x = self.dropout(x)
        
        # Concatenate commander embeddings
        return x.view(x.size(0), -1)  # (batch_size, 2 * embed_dim)


class MutationEmbedding(nn.Module):
    """Embedding layer for mutation factor features."""
    
    def __init__(self, 
                 num_mutations: int, 
                 embed_dim: int,
                 dropout: float = 0.1):
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
        # Embed each mutation factor
        x = self.embedding(x)  # (batch_size, num_mutations, embed_dim)
        
        if mask is not None:
            # Apply mask
            mask = mask.unsqueeze(-1)  # (batch_size, num_mutations, 1)
            x = x * mask
        
        # Average pooling over mutation factors
        x = x.mean(dim=1)  # (batch_size, embed_dim)
        return self.dropout(x)


class AIEmbedding(nn.Module):
    """Embedding layer for enemy AI features."""
    
    def __init__(self, num_ais: int, embed_dim: int, dropout: float = 0.1):
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
    """Combined embedding layer for all features."""
    
    def __init__(self,
                 num_maps: int,
                 num_commanders: int,
                 num_mutations: int,
                 num_ais: int,
                 map_dim: int = 64,
                 commander_dim: int = 128,
                 mutation_dim: int = 96,
                 ai_dim: int = 32,
                 dropout: float = 0.1):
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
        
        self.output_dim = map_dim + 2 * commander_dim + mutation_dim + ai_dim
        
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
            Combined embedded tensor of shape (batch_size, output_dim)
        """
        map_embed = self.map_embedding(map_ids)
        commander_embed = self.commander_embedding(commander_ids)
        mutation_embed = self.mutation_embedding(mutation_ids, mutation_mask)
        ai_embed = self.ai_embedding(ai_ids)
        
        # Concatenate all embeddings
        return torch.cat([
            map_embed,
            commander_embed,
            mutation_embed,
            ai_embed
        ], dim=1) 