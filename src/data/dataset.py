"""
数据集模块
"""
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from typing import List, Dict, Set, Tuple

class MutationDataset(Dataset):
    """突变数据集类"""
    
    def __init__(self, data_list: List[Data]):
        """
        初始化数据集
        
        Args:
            data_list: 数据列表
        """
        self.data_list = data_list
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Data:
        """获取单个数据样本"""
        return self.data_list[idx]

class MutationDataProcessor:
    """突变数据处理器"""
    
    def __init__(self, data_path: str):
        """
        初始化数据处理器
        
        Args:
            data_path: 数据文件路径
        """
        self.data_path = data_path
        self.df = None
        self.map_to_id = None
        self.factor_to_id = None
    
    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        加载并预处理数据
        
        Returns:
            df: 数据框
            map_to_id: 地图到ID的映射
        """
        # 读取数据
        self.df = pd.read_csv(self.data_path)
        
        # 创建地图ID映射
        unique_maps = self.df['地图'].unique()
        self.map_to_id = {map_name: idx for idx, map_name in enumerate(unique_maps)}
        
        return self.df, self.map_to_id
    
    def get_all_factors(self) -> Set[str]:
        """
        获取所有唯一的突变因子
        
        Returns:
            factors: 因子集合
        """
        if self.df is None:
            self.load_data()
            
        factors = set()
        for col in ['因子1', '因子2', '因子3', '因子4']:
            factors.update(self.df[col].dropna().unique())
        return factors
    
    def create_factor_mapping(self, factors: Set[str]) -> Dict[str, int]:
        """
        创建因子到ID的映射
        
        Args:
            factors: 因子集合
        
        Returns:
            factor_to_id: 因子到ID的映射
        """
        self.factor_to_id = {factor: idx for idx, factor in enumerate(sorted(factors))}
        return self.factor_to_id
    
    def extract_factors(self) -> np.ndarray:
        """
        提取突变因子特征
        
        Returns:
            features: 形状为 (n_samples, n_factors) 的one-hot编码矩阵
        """
        # 获取所有唯一因子
        all_factors = self.get_all_factors()
        if self.factor_to_id is None:
            self.create_factor_mapping(all_factors)
            
        n_factors = len(self.factor_to_id)
        
        # 创建特征矩阵
        features = np.zeros((len(self.df), n_factors))
        
        # 填充特征矩阵
        for i, row in self.df.iterrows():
            for col in ['因子1', '因子2', '因子3', '因子4']:
                if pd.notna(row[col]):
                    factor_id = self.factor_to_id[row[col]]
                    features[i, factor_id] = 1
        
        return features
    
    def build_factor_graph(self) -> torch.Tensor:
        """
        构建因子之间的关系图
        
        Returns:
            edge_index: 边索引张量 (2, num_edges)
        """
        # 定义因子关系
        factor_relations = [
            # 强力因子关系
            ("风暴英雄", "虚空裂隙"),
            ("风暴英雄", "复仇战士"),
            ("风暴英雄", "力量蜕变"),
            ("虚空裂隙", "时空立场"),
            ("虚空裂隙", "岩浆爆发"),
            
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
            if f1 in self.factor_to_id and f2 in self.factor_to_id:
                # 添加双向边
                edges.append([self.factor_to_id[f1], self.factor_to_id[f2]])
                edges.append([self.factor_to_id[f2], self.factor_to_id[f1]])
                
                # 添加自环边
                edges.append([self.factor_to_id[f1], self.factor_to_id[f1]])
                edges.append([self.factor_to_id[f2], self.factor_to_id[f2]])
        
        # 确保每个节点都有自环边
        for factor_id in self.factor_to_id.values():
            edges.append([factor_id, factor_id])
        
        # 去除重复边
        edges = list(set(tuple(edge) for edge in edges))
        edges = [list(edge) for edge in edges]
        
        # 转换为PyTorch张量
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return edge_index
    
    def create_graph_data(self) -> List[Data]:
        """
        创建图数据对象列表
        
        Returns:
            data_list: 图数据对象列表
        """
        # 提取特征
        factors = self.extract_factors()
        map_ids = np.array([self.map_to_id[m] for m in self.df['地图']])
        labels = self.df['评级'].values
        
        # 构建因子关系图
        edge_index = self.build_factor_graph()
        
        # 创建数据列表
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
    
    def prepare_dataset(self) -> Tuple[MutationDataset, Dict[str, int]]:
        """
        准备完整的数据集
        
        Returns:
            dataset: 数据集对象
            map_to_id: 地图到ID的映射
        """
        # 加载数据
        self.load_data()
        
        # 创建图数据列表
        data_list = self.create_graph_data()
        
        # 创建数据集
        dataset = MutationDataset(data_list)
        
        return dataset, self.map_to_id 