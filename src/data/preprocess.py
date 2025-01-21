import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Tuple, List, Dict, Set

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    加载并预处理数据
    """
    # 读取数据
    df = pd.read_csv(data_path)
    
    # 创建地图ID映射
    unique_maps = df['地图'].unique()
    map_to_id = {map_name: idx for idx, map_name in enumerate(unique_maps)}
    
    return df, map_to_id

def get_all_factors(df: pd.DataFrame) -> Set[str]:
    """
    获取所有唯一的突变因子
    """
    factors = set()
    for col in ['因子1', '因子2', '因子3', '因子4']:
        factors.update(df[col].dropna().unique())
    return factors

def create_factor_mapping(factors: Set[str]) -> Dict[str, int]:
    """
    创建因子到ID的映射
    """
    return {factor: idx for idx, factor in enumerate(sorted(factors))}

def extract_factors(df: pd.DataFrame) -> np.ndarray:
    """
    提取突变因子特征
    返回形状为 (n_samples, n_factors) 的one-hot编码矩阵
    """
    # 获取所有唯一因子
    all_factors = get_all_factors(df)
    factor_to_id = create_factor_mapping(all_factors)
    n_factors = len(factor_to_id)
    
    # 创建特征矩阵
    features = np.zeros((len(df), n_factors))
    
    # 填充特征矩阵
    for i, row in df.iterrows():
        for col in ['因子1', '因子2', '因子3', '因子4']:
            if pd.notna(row[col]):
                factor_id = factor_to_id[row[col]]
                features[i, factor_id] = 1
    
    return features

def build_factor_relationships() -> torch.Tensor:
    """
    构建因子之间的关系图
    返回边索引张量 (2, num_edges)
    
    因子分为五大类：
    1. 敌方部队增益因子：增加敌方单位能力的因子
    2. 环境因子：影响地图环境和地形的因子
    3. 额外机制因子：改变游戏基本机制的因子
    4. 视野限制因子：影响玩家视野的因子
    5. 强力因子：对游戏难度有显著影响的因子
    """
    # 获取所有因子
    df = pd.read_csv('data/raw/train.csv')
    all_factors = get_all_factors(df)
    factor_to_id = create_factor_mapping(all_factors)
    
    # 打印调试信息
    print(f"\nFactor mapping:")
    for factor, idx in factor_to_id.items():
        print(f"{factor}: {idx}")
    
    # 定义因子关系
    factor_relations = [
        # 强力因子关系
        ("风暴英雄", "虚空裂隙"),  # 两个强力因子的组合
        ("风暴英雄", "复仇战士"),  # 与部队增益的关联
        ("风暴英雄", "力量蜕变"),  # 与部队增益的关联
        ("虚空裂隙", "时空立场"),  # 与环境的关联
        ("虚空裂隙", "岩浆爆发"),  # 与环境的关联
        
        # 敌方部队增益因子关系
        ("复仇战士", "力量蜕变"),
        ("复仇战士", "鼓舞人心"),
        ("力量蜕变", "鼓舞人心"),
        ("坚强意志", "力量蜕变"),
        ("坚强意志", "鼓舞人心"),
        ("同化体", "异形寄生"),
        ("同化体", "虚空重生者"),
        ("异形寄生", "虚空重生者"),
        
        # 环境因子关系
        ("暴风雪", "龙卷风暴"),
        ("暴风雪", "岩浆爆发"),
        ("龙卷风暴", "岩浆爆发"),
        ("焦土政策", "岩浆爆发"),
        ("核弹打击", "轨道轰炸"),
        ("核弹打击", "飞弹大战"),
        ("轨道轰炸", "飞弹大战"),
        ("强磁雷场", "震荡攻击"),
        
        # 额外机制因子关系
        ("时间扭曲", "速度狂魔"),
        ("时间扭曲", "时空立场"),
        ("速度狂魔", "来去无踪"),
        ("闪避机动", "来去无踪"),
        ("减伤屏障", "晶矿护盾"),
        ("黑死病", "丧尸大战"),
        ("黑死病", "行尸走肉"),
        ("丧尸大战", "行尸走肉"),
        ("极性不定", "相互摧毁"),
        ("双重压力", "相互摧毁"),
        ("生命汲取", "黑死病"),
        
        # 视野限制因子关系
        ("暗无天日", "短视症"),
        ("超远视距", "短视症"),
        ("暗无天日", "超远视距"),
        ("光子过载", "短视症"),
        ("光子过载", "超远视距")
    ]
    
    # 创建边索引
    edges = []
    for f1, f2 in factor_relations:
        if f1 in factor_to_id and f2 in factor_to_id:
            # 添加双向边
            edges.append([factor_to_id[f1], factor_to_id[f2]])
            edges.append([factor_to_id[f2], factor_to_id[f1]])
            
            # 添加自环边
            edges.append([factor_to_id[f1], factor_to_id[f1]])
            edges.append([factor_to_id[f2], factor_to_id[f2]])
    
    # 确保每个节点都有自环边
    for factor_id in factor_to_id.values():
        edges.append([factor_id, factor_id])
    
    # 去除重复边
    edges = list(set(tuple(edge) for edge in edges))
    edges = [list(edge) for edge in edges]
    
    # 打印调试信息
    print(f"\nNumber of edges: {len(edges)}")
    print(f"Edge index range: [{min(min(edge) for edge in edges)}, {max(max(edge) for edge in edges)}]")
    
    # 转换为PyTorch张量
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return edge_index

def create_graph_data(
    factors: np.ndarray,
    map_ids: np.ndarray,
    edge_index: torch.Tensor,
    labels: np.ndarray
) -> List[Data]:
    """
    创建图数据对象列表
    
    Args:
        factors: 因子特征矩阵 [n_samples, n_factors]
        map_ids: 地图ID数组 [n_samples]
        edge_index: 边索引张量 [2, n_edges]
        labels: 标签数组 [n_samples]
    
    Returns:
        data_list: 图数据对象列表
    """
    data_list = []
    for i in range(len(factors)):
        # 将特征向量转换为2D张量 [n_factors, 1]
        x = torch.FloatTensor(factors[i]).unsqueeze(1)
        
        data = Data(
            x=x,  # [n_factors, 1]
            edge_index=edge_index,  # [2, n_edges]
            map_id=torch.LongTensor([map_ids[i]]),  # [1]
            y=torch.LongTensor([labels[i] - 1])  # [1] 评级从1开始，转换为从0开始
        )
        data_list.append(data)
    
    return data_list

def prepare_data(data_path: str) -> Tuple[List[Data], Dict[str, int]]:
    """
    准备模型训练所需的所有数据
    """
    # 1. 加载和预处理数据
    df, map_to_id = load_and_preprocess_data(data_path)
    
    # 2. 提取特征
    factors = extract_factors(df)
    map_ids = np.array([map_to_id[m] for m in df['地图']])
    labels = df['评级'].values
    
    # 3. 构建因子关系图
    edge_index = build_factor_relationships()
    
    # 打印调试信息
    print(f"Number of factors: {factors.shape[1]}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge index max value: {edge_index.max().item()}")
    print(f"Edge index min value: {edge_index.min().item()}")
    
    # 4. 创建图数据对象
    data_list = create_graph_data(factors, map_ids, edge_index, labels)
    
    return data_list, map_to_id 