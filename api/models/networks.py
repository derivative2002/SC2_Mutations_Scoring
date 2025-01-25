import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """多层感知机网络"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list,
                 output_dim: int,
                 dropout: float = 0.2,
                 use_batch_norm: bool = True):
        """初始化MLP.
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout: Dropout比率
            use_batch_norm: 是否使用BatchNorm
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            prev_dim = dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.layers(x)


class MutationScorer(nn.Module):
    """突变难度评分模型"""
    
    def __init__(self,
                 num_maps: int,
                 num_commanders: int,
                 num_mutations: int,
                 num_ais: int,
                 map_dim: int = 48,
                 commander_dim: int = 48,
                 mutation_dim: int = 48,
                 ai_dim: int = 48,
                 hidden_dims: list = [128, 64, 32],
                 num_classes: int = 5,
                 dropout: float = 0.5,
                 embed_dropout: float = 0.3):
        """初始化模型.
        
        Args:
            num_maps: 地图数量
            num_commanders: 指挥官数量
            num_mutations: 突变因子数量
            num_ais: AI类型数量
            map_dim: 地图嵌入维度
            commander_dim: 指挥官嵌入维度
            mutation_dim: 突变因子嵌入维度
            ai_dim: AI嵌入维度
            hidden_dims: MLP隐藏层维度
            num_classes: 类别数量（1-5分）
            dropout: MLP的dropout比率
            embed_dropout: 嵌入层的dropout比率
        """
        super().__init__()
        
        # 特征嵌入层
        self.map_embedding = nn.Embedding(num_maps, map_dim)
        self.commander_embedding = nn.Embedding(num_commanders, commander_dim)
        self.mutation_embedding = nn.Embedding(num_mutations, mutation_dim)
        self.ai_embedding = nn.Embedding(num_ais, ai_dim)
        
        # Embedding dropout
        self.embed_dropout = nn.Dropout(embed_dropout)
        
        # 计算MLP输入维度
        self.input_dim = (map_dim + 
                         commander_dim * 2 +  # 双指挥官
                         mutation_dim +  # 突变因子的平均池化
                         ai_dim)
        
        # MLP分类器
        self.classifier = MLP(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
            dropout=dropout
        )
        
    def forward(self,
                map_ids: torch.Tensor,
                commander_ids: torch.Tensor,
                mutation_ids: torch.Tensor,
                ai_ids: torch.Tensor,
                mutation_mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播.
        
        Args:
            map_ids: 地图ID，形状为(batch_size,)
            commander_ids: 指挥官ID，形状为(batch_size, 2)
            mutation_ids: 突变因子ID，形状为(batch_size, num_mutations)
            ai_ids: AI类型ID，形状为(batch_size,)
            mutation_mask: 突变因子的mask，形状为(batch_size, num_mutations)
            
        Returns:
            形状为(batch_size, num_classes)的logits
        """
        # 特征嵌入
        map_embed = self.embed_dropout(self.map_embedding(map_ids))
        commander_embed = self.embed_dropout(self.commander_embedding(commander_ids))
        mutation_embed = self.embed_dropout(self.mutation_embedding(mutation_ids))
        ai_embed = self.embed_dropout(self.ai_embedding(ai_ids))
        
        # 处理双指挥官
        commander_embed = commander_embed.view(commander_embed.size(0), -1)  # 展平
        
        # 处理突变因子（平均池化）
        if mutation_mask is not None:
            mutation_embed = mutation_embed * mutation_mask.unsqueeze(-1)
            mutation_embed = mutation_embed.sum(1) / mutation_mask.sum(1, keepdim=True).clamp(min=1)
        else:
            mutation_embed = mutation_embed.mean(1)
        
        # 特征拼接
        x = torch.cat([
            map_embed,
            commander_embed,
            mutation_embed,
            ai_embed
        ], dim=-1)
        
        # 分类
        logits = self.classifier(x)
        
        return logits
    
    def predict(self,
                map_ids: torch.Tensor,
                commander_ids: torch.Tensor,
                mutation_ids: torch.Tensor,
                ai_ids: torch.Tensor,
                mutation_mask: torch.Tensor = None) -> tuple:
        """预测.
        
        Returns:
            (预测类别, 预测概率) 的元组
        """
        logits = self.forward(
            map_ids, commander_ids, mutation_ids, ai_ids, mutation_mask)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return preds, probs 