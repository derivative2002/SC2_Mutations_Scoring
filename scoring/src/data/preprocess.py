"""数据预处理模块."""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Set
import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Vocab:
    """词表类，用于特征转换."""
    
    def __init__(self, name: str, special_tokens: List[str] = None):
        """初始化词表.
        
        Args:
            name: 词表名称
            special_tokens: 特殊token列表，例如[PAD]等
        """
        self.name = name
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        
        # 确保special_tokens中包含[UNK]和[PAD]
        if special_tokens is None:
            special_tokens = []
        if '[UNK]' not in special_tokens:
            special_tokens = ['[UNK]'] + special_tokens
        if '[PAD]' not in special_tokens:
            special_tokens.append('[PAD]')
        self.special_tokens = special_tokens
        
        # 添加特殊token
        for token in self.special_tokens:
            self.add_token(token)
    
    def add_token(self, token: str) -> int:
        """添加token到词表.
        
        Args:
            token: 要添加的token
            
        Returns:
            token的索引
        """
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def __getitem__(self, token: str) -> int:
        """获取token的索引.
        
        Args:
            token: 输入token
            
        Returns:
            token的索引，如果不存在则返回[UNK]的索引
        """
        return self.token2idx.get(token, self.token2idx['[UNK]'])
    
    @property
    def pad_id(self) -> int:
        """获取[PAD]的索引."""
        return self.token2idx['[PAD]']
    
    def save(self, vocab_dir: str):
        """保存词表到文件.
        
        Args:
            vocab_dir: 词表保存目录
        """
        vocab_path = Path(vocab_dir) / f"{self.name}_vocab.json"
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump({
                'token2idx': self.token2idx,
                'special_tokens': self.special_tokens
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"词表已保存到: {vocab_path}")
    
    @classmethod
    def load(cls, vocab_dir: str, name: str) -> 'Vocab':
        """从文件加载词表.
        
        Args:
            vocab_dir: 词表目录
            name: 词表名称
            
        Returns:
            加载的词表对象
        """
        vocab_path = Path(vocab_dir) / f"{name}_vocab.json"
        with open(vocab_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        vocab = cls(name, data['special_tokens'])
        vocab.token2idx = {k: int(v) for k, v in data['token2idx'].items()}
        vocab.idx2token = {int(k): v for k, v in 
                          {v: k for k, v in vocab.token2idx.items()}.items()}
        logger.info(f"已加载词表: {vocab_path}")
        return vocab


class SC2MutationPreprocessor:
    """星际2突变数据预处理器."""
    
    def __init__(self, 
                 processed_dir: str,
                 max_mutations: int = 10,
                 raw_data_path: str = None):  # 保持参数兼容性
        """初始化预处理器.
        
        Args:
            processed_dir: 处理后数据保存目录
            max_mutations: 最大突变因子数量
            raw_data_path: 原始数据文件路径（可选，用于兼容旧代码）
        """
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_mutations = max_mutations
        
        # 词表
        self.map_vocab = Vocab('map')
        self.commander_vocab = Vocab('commander')
        self.mutation_vocab = Vocab(
            'mutation', special_tokens=['[PAD]'])
        self.ai_vocab = Vocab('ai')
        
        # 元数据
        self.metadata = {
            'max_mutations': max_mutations,
            'num_classes': 5,
            'class_weights': None
        }
    
    def _safe_eval(self, s):
        """安全地解析字符串列表.
        
        Args:
            s: 输入字符串，格式如 "'item1', 'item2', 'item3'"
            
        Returns:
            解析后的列表
        """
        if pd.isna(s):
            return []
        try:
            # 清理字符串并分割
            items = [item.strip().strip("'").strip('"') for item in s.split(',')]
            # 过滤空值
            return [item for item in items if item]
        except:
            logger.warning(f"解析失败: {s}")
            return []

    def _build_vocabs(self, df: pd.DataFrame):
        """构建所有特征的词表.
        
        Args:
            df: 原始数据DataFrame
        """
        logger.info("开始构建词表...")
        
        # 地图词表
        for map_name in tqdm(df['map_name'].unique(), desc="构建地图词表"):
            if pd.notna(map_name):
                self.map_vocab.add_token(map_name)
        
        # 指挥官词表
        commanders = set()
        for cmd_pair in tqdm(df['commanders'], desc="构建指挥官词表"):
            if pd.notna(cmd_pair):
                cmds = self._safe_eval(cmd_pair)
                commanders.update(cmds)
        for cmd in commanders:
            if cmd:  # 过滤空值
                self.commander_vocab.add_token(cmd)
        
        # 突变因子词表
        mutations = set()
        for mutation_list in tqdm(df['mutation_factors'], desc="构建突变因子词表"):
            if pd.notna(mutation_list):
                factors = self._safe_eval(mutation_list)
                mutations.update(factors)
        for mutation in mutations:
            if mutation:  # 过滤空值
                self.mutation_vocab.add_token(mutation)
        
        # AI词表
        for ai in tqdm(df['enemy_ai'].unique(), desc="构建AI词表"):
            if pd.notna(ai):
                self.ai_vocab.add_token(ai)
        
        logger.info(f"词表构建完成:")
        logger.info(f"- 地图数量: {len(self.map_vocab)}")
        logger.info(f"- 指挥官数量: {len(self.commander_vocab)}")
        logger.info(f"- 突变因子数量: {len(self.mutation_vocab)}")
        logger.info(f"- AI类型数量: {len(self.ai_vocab)}")
    
    def _convert_features(self, 
                         df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """转换特征为模型输入格式.
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            特征字典和标签数组
        """
        logger.info("开始转换特征...")
        num_samples = len(df)
        
        # 初始化特征数组
        features = {
            'map_ids': np.zeros(num_samples, dtype=np.int64),
            'commander_ids': np.zeros((num_samples, 2), dtype=np.int64),
            'mutation_ids': np.zeros(
                (num_samples, self.max_mutations), dtype=np.int64),
            'mutation_mask': np.zeros(
                (num_samples, self.max_mutations), dtype=np.float32),
            'ai_ids': np.zeros(num_samples, dtype=np.int64),
            'is_verified': np.zeros(num_samples, dtype=np.int64)
        }
        
        # 获取[PAD]和[UNK]的索引
        pad_idx = self.mutation_vocab.token2idx['[PAD]']
        unk_map_idx = self.map_vocab.token2idx.get('[UNK]', 0)
        unk_commander_idx = self.commander_vocab.token2idx.get('[UNK]', 0)
        unk_mutation_idx = self.mutation_vocab.token2idx.get('[UNK]', pad_idx)
        unk_ai_idx = self.ai_vocab.token2idx.get('[UNK]', 0)
        
        # 转换特征
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="转换特征")):
            # 地图ID
            if pd.notna(row['map_name']):
                features['map_ids'][idx] = self.map_vocab.token2idx.get(
                    row['map_name'], unk_map_idx)
            
            # 指挥官ID
            if pd.notna(row['commanders']):
                commanders = self._safe_eval(row['commanders'])
                for j, cmd in enumerate(commanders[:2]):
                    if cmd:
                        features['commander_ids'][idx, j] = (
                            self.commander_vocab.token2idx.get(cmd, unk_commander_idx))
            
            # 突变因子ID和mask
            if pd.notna(row['mutation_factors']):
                mutations = self._safe_eval(row['mutation_factors'])
                # 只取前max_mutations个突变因子
                for j, mutation in enumerate(mutations[:self.max_mutations]):
                    if mutation:
                        features['mutation_ids'][idx, j] = (
                            self.mutation_vocab.token2idx.get(mutation, unk_mutation_idx))
                        features['mutation_mask'][idx, j] = 1.0
                # 将剩余位置填充为[PAD]
                features['mutation_ids'][idx, len(mutations):] = pad_idx
            else:
                # 如果没有突变因子，全部填充为[PAD]
                features['mutation_ids'][idx, :] = pad_idx
            
            # AI ID
            if pd.notna(row['enemy_ai']):
                features['ai_ids'][idx] = self.ai_vocab.token2idx.get(
                    row['enemy_ai'], unk_ai_idx)
            
            # 是否验证
            features['is_verified'][idx] = 1 if pd.notna(row['is_verified']) and row['is_verified'] == 1 else 0
        
        # 转换标签
        # 先将difficulty_score转换为数值类型
        df['difficulty_score'] = pd.to_numeric(df['difficulty_score'], errors='coerce')
        # 检查是否有无效值
        invalid_scores = df['difficulty_score'].isna()
        if invalid_scores.any():
            logger.warning(f"发现{invalid_scores.sum()}个无效的难度分数")
            # 使用众数填充无效值
            mode_score = df['difficulty_score'].mode()[0]
            df.loc[invalid_scores, 'difficulty_score'] = mode_score
        
        labels = df['difficulty_score'].values.astype(np.int64) - 1  # 转换为0-4
        
        logger.info("特征转换完成")
        return features, labels
    
    def process(self, df: pd.DataFrame):
        """处理数据.
        
        Args:
            df: 原始数据DataFrame
        """
        # 构建词表
        self._build_vocabs(df)
        
        # 保存词表
        vocab_dir = self.processed_dir / 'vocabs'
        vocab_dir.mkdir(parents=True, exist_ok=True)
        self.map_vocab.save(vocab_dir)
        self.commander_vocab.save(vocab_dir)
        self.mutation_vocab.save(vocab_dir)
        self.ai_vocab.save(vocab_dir)
        
        # 转换特征
        features, labels = self._convert_features(df)
        
        # 计算类别权重
        class_counts = Counter(labels)
        total_samples = len(labels)
        self.metadata['class_weights'] = [
            total_samples / (len(class_counts) * count)
            for i, count in sorted(class_counts.items())
        ]
        
        # 保存处理后的数据
        np.savez(
            self.processed_dir / 'processed_data.npz',
            **features,
            labels=labels
        )
        
        # 保存元数据
        with open(self.processed_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info("数据处理完成!")
        logger.info(f"- 样本总数: {total_samples}")
        logger.info("- 类别分布:")
        for label, count in sorted(class_counts.items()):
            percentage = count / total_samples * 100
            logger.info(f"  类别 {label}: {count} 样本 ({percentage:.2f}%)")


def main():
    """主函数."""
    # 设置路径
    raw_data_paths = [
        "data/raw/【实验数据】mutation_tasks_初版生成数据.csv",  # 生成数据
        "data/raw/【实验数据】高玩实际测试的高质量数据_评测集.csv"  # 高玩测试数据
    ]
    processed_dir = "data/processed"
    
    # 读取并合并数据
    dfs = []
    for path in raw_data_paths:
        logger.info(f"读取数据文件: {path}")
        df = pd.read_csv(
            path,
            encoding='utf-8',
            quoting=csv.QUOTE_ALL,  # 使用引号包围所有字段
            escapechar='\\',  # 使用反斜杠作为转义字符
            on_bad_lines='skip'  # 跳过错误行
        )
        # 确保高玩测试数据的is_verified为1
        if "高玩实际测试" in path:
            df['is_verified'] = 1
        dfs.append(df)
    
    # 合并数据
    merged_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"合并后的数据集大小: {len(merged_df)}")
    
    # 创建预处理器并处理数据
    preprocessor = SC2MutationPreprocessor(
        processed_dir=processed_dir
    )
    
    # 处理数据
    preprocessor.process(merged_df)


if __name__ == "__main__":
    main() 