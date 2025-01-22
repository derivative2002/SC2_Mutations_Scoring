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
        hidden_dim: int = 256,  # 增加隐藏维度
        num_gnn_layers: int = 4,  # 增加GNN层数
        num_classes: int = 10,
        dropout: float = 0.5,
        l2_lambda: float = 0.01,
        device: torch.device = None
    ):
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        self.hidden_dim = hidden_dim
        self.l2_lambda = l2_lambda
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=12, dropout=dropout))  # 增加注意力头数
        
        # 中间GNN层带残差连接和跳跃连接
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(
                GATConv(hidden_dim * 12, hidden_dim, heads=12, dropout=dropout)
            )
        
        # 层归一化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim * 12) for _ in range(num_gnn_layers)
        ])
        
        # 跳跃连接投影
        self.skip_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 12, hidden_dim * 12),
                nn.LayerNorm(hidden_dim * 12),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_gnn_layers - 1)
        ])
            
        # 地图嵌入
        self.map_embedding = nn.Embedding(num_maps, hidden_dim * 6)
        self.map_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim * 12),
            nn.LayerNorm(hidden_dim * 12),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 节点嵌入投影
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim * 12 * 3, hidden_dim * 12),
            nn.LayerNorm(hidden_dim * 12),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 多头自注意力层
        self.self_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim * 12,
                num_heads=12,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 使用2层自注意力
        ])
        
        # 多头交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim * 12,
                num_heads=12,
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 使用2层交叉注意力
        ])
        
        # 特征融合网络
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 12 * 3, hidden_dim * 12),
            nn.LayerNorm(hidden_dim * 12),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 12, hidden_dim * 8),
            nn.LayerNorm(hidden_dim * 8),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 输出层
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.LayerNorm(hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 将模型移动到指定设备
        self.to(self.device)
        
    def l2_regularization(self):
        """计算L2正则化损失"""
        l2_loss = 0.0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return self.l2_lambda * l2_loss
        
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
        # 输入特征投影
        x = self.input_proj(x)
        
        # GNN特征提取
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
        
        # 图池化
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
            
        pooled_mean = global_mean_pool(x, batch)
        pooled_max = global_max_pool(x, batch)
        pooled_add = global_add_pool(x, batch)
        
        # 节点表示融合
        node_embeddings = torch.cat([pooled_mean, pooled_max, pooled_add], dim=1)
        node_embeddings = self.node_proj(node_embeddings)
        
        # 地图嵌入
        map_embeddings = self.map_embedding(map_id)
        map_embeddings = self.map_proj(map_embeddings)
        
        # 多层自注意力
        query = node_embeddings.unsqueeze(1)
        self_attn_output = query
        self_attn_weights_list = []
        
        for self_attn in self.self_attention_layers:
            self_attn_output, self_attn_weights = self_attn(
                self_attn_output, self_attn_output, self_attn_output
            )
            self_attn_weights_list.append(self_attn_weights)
        
        # 多层交叉注意力
        key = map_embeddings.unsqueeze(1)
        value = map_embeddings.unsqueeze(1)
        cross_attn_output = query
        cross_attn_weights_list = []
        
        for cross_attn in self.cross_attention_layers:
            cross_attn_output, cross_attn_weights = cross_attn(
                cross_attn_output, key, value
            )
            cross_attn_weights_list.append(cross_attn_weights)
        
        # 特征融合
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
        
        # 输出层
        out = self.fc_layers(fused)
        
        # 添加L2正则化损失
        if self.training:
            self.current_l2_loss = self.l2_regularization()
            out = out + self.current_l2_loss
        
        # 保存中间结果供可视化使用
        self._last_factor_embeddings = factor_embeddings[-1]
        self._last_attention_weights = {
            'self_attention': self_attn_weights_list,
            'cross_attention': cross_attn_weights_list
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