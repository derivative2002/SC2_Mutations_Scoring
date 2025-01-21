import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from utils.progress import BatchProgressBar

class MutationGNN(nn.Module):
    """
    突变评级预测模型
    包含GNN用于处理因子关系，注意力机制用于处理地图影响
    """
    def __init__(
        self,
        num_factors: int,
        num_maps: int,
        hidden_dim: int = 128,  # 增加基础隐藏维度
        num_gnn_layers: int = 3,
        num_classes: int = 10,
        dropout: float = 0.3,
        device: torch.device = None
    ):
        super().__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hidden_dim = hidden_dim
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(1, hidden_dim, heads=8))
        
        # 中间GNN层带残差连接和跳跃连接
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(
                GATConv(hidden_dim * 8, hidden_dim, heads=8)
            )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 8) for _ in range(num_gnn_layers)
        ])
        
        # 跳跃连接投影
        self.skip_projections = nn.ModuleList([
            nn.Linear(hidden_dim * 8, hidden_dim * 8) 
            for _ in range(num_gnn_layers - 1)
        ])
            
        # 地图嵌入
        self.map_embedding = nn.Embedding(num_maps, hidden_dim * 4)
        self.map_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 节点嵌入投影
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 3, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 8,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 8,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 8 * 3, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.fc1 = nn.Linear(hidden_dim * 8, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_final = nn.LayerNorm(hidden_dim * 4)
        
        # 将模型移动到指定设备
        self.to(self.device)
        
    def forward(self, x, edge_index, map_id, batch=None, return_attention=False):
        """
        前向传播
        
        Args:
            x: 因子特征 [num_nodes, 1]
            edge_index: 边索引 [2, num_edges]
            map_id: 地图ID [batch_size]
            batch: 批处理索引 [num_nodes]
            return_attention: 是否返回注意力权重
        
        Returns:
            out: 输出预测 [batch_size, num_classes]
            attention_weights: 注意力权重 [batch_size, 1, 1]（如果return_attention=True）
        """
        # 1. GNN处理因子关系
        factor_embeddings = []
        skip_connection = None
        
        for i, gnn_layer in enumerate(self.gnn_layers):
            # 应用GNN层
            x_new = gnn_layer(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x_new = self.layer_norms[i](x_new)
            
            # 残差连接和跳跃连接
            if i > 0:
                # 残差连接
                x = x + x_new
                # 跳跃连接
                skip_proj = self.skip_projections[i-1]
                if skip_connection is None:
                    skip_connection = skip_proj(x)
                else:
                    skip_connection = skip_connection + skip_proj(x)
            else:
                x = x_new
                
            factor_embeddings.append(x)
        
        # 2. 多尺度图池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            
        # 使用不同的池化方法
        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        pooled_add = global_add_pool(x, batch)
        
        # 合并不同池化结果并投影
        node_embeddings = torch.cat([pooled_mean, pooled_max, pooled_add], dim=1)
        node_embeddings = self.node_proj(node_embeddings)  # [batch_size, hidden_dim * 8]
        
        # 3. 地图嵌入
        map_embeddings = self.map_embedding(map_id)  # [batch_size, hidden_dim * 4]
        map_embeddings = self.map_proj(map_embeddings)  # [batch_size, hidden_dim * 8]
        
        # 4. 多层注意力机制
        # 自注意力
        query = node_embeddings.unsqueeze(1)
        self_attn_output, self_attn_weights = self.self_attention(
            query, query, query
        )
        
        # 交叉注意力
        key = map_embeddings.unsqueeze(1)
        value = map_embeddings.unsqueeze(1)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            query, key, value
        )
        
        # 5. 特征融合
        if skip_connection is not None:
            skip_connection = global_mean_pool(skip_connection, batch)
        else:
            skip_connection = torch.zeros_like(node_embeddings)
            
        combined = torch.cat([
            node_embeddings,
            self_attn_output.squeeze(1),
            cross_attn_output.squeeze(1)
        ], dim=1)
        
        fused = self.feature_fusion(combined)
        
        # 6. 输出层
        out = self.fc1(fused)
        out = F.gelu(out)
        out = self.dropout(out)
        out = self.layer_norm_final(out)
        
        out = self.fc2(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        # 保存中间结果供可视化使用
        self._last_factor_embeddings = factor_embeddings[-1]
        self._last_attention_weights = {
            'self_attention': self_attn_weights,
            'cross_attention': cross_attn_weights
        }
        
        if return_attention:
            return out, self._last_attention_weights
        return out

    def get_factor_embeddings(self) -> torch.Tensor:
        """
        获取最后一层GNN的因子嵌入
        
        Returns:
            factor_embeddings: 因子嵌入矩阵 [num_factors, hidden_dim * 4]
        """
        if not hasattr(self, '_last_factor_embeddings'):
            raise RuntimeError("需要先运行forward才能获取因子嵌入")
        return self._last_factor_embeddings

    def get_attention_weights(self) -> torch.Tensor:
        """
        获取注意力权重
        
        Returns:
            attention_weights: 注意力权重矩阵 [batch_size, 1, 1]
        """
        if not hasattr(self, '_last_attention_weights'):
            raise RuntimeError("需要先运行forward才能获取注意力权重")
        return self._last_attention_weights

    def get_map_embeddings(self) -> torch.Tensor:
        """
        获取地图嵌入
        
        Returns:
            map_embeddings: 地图嵌入矩阵 [num_maps, hidden_dim]
        """
        return self.map_embedding.weight.data

class MutationDataset(torch.utils.data.Dataset):
    """
    突变数据集类
    """
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def train_epoch(model, loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    
    # 创建批次进度条
    pbar = BatchProgressBar(len(loader), desc='Training Batches')
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out, _ = model(data.x, data.edge_index, data.map_id, data.batch, return_attention=True)
        loss = criterion(out, data.y)
        
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item() * data.num_graphs
        total_loss += batch_loss
        
        # 更新进度条
        pbar.update_loss(batch_loss / data.num_graphs)
        pbar.update()
    
    pbar.close()
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    """
    评估模型
    
    Args:
        model: 模型
        loader: 数据加载器
        criterion: 损失函数
        device: 计算设备
    
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.eval()
    total_loss = 0
    correct = 0
    
    # 创建批次进度条
    pbar = BatchProgressBar(len(loader), desc='Evaluating')
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out, _ = model(data.x, data.edge_index, data.map_id, data.batch, return_attention=True)
            loss = criterion(out, data.y)
            
            batch_loss = loss.item() * data.num_graphs
            total_loss += batch_loss
            
            pred = out.argmax(dim=1)
            batch_correct = (pred == data.y).sum().item()
            correct += batch_correct
            
            # 更新进度条
            batch_acc = batch_correct / data.num_graphs
            pbar.set_postfix(loss=f'{batch_loss/data.num_graphs:.4f}', acc=f'{batch_acc:.4f}')
            pbar.update()
    
    pbar.close()
    return total_loss / len(loader.dataset), correct / len(loader.dataset) 