"""检查数据中的ID映射."""

import pandas as pd
from src.data.preprocess import Vocab

def check_data():
    # 加载词表
    map_vocab = Vocab.load('data/processed/vocabs', 'map')
    print("地图词表:")
    for token, idx in map_vocab.token2idx.items():
        print(f"{idx}: {token}")
    
    # 读取数据
    df = pd.read_csv('data/processed/sc2_mutations_duo.csv')
    print("\n数据中的地图名称:")
    for map_name in df['map_name'].unique():
        if map_name not in map_vocab.token2idx:
            print(f"警告: 地图 '{map_name}' 不在词表中!")
        else:
            print(f"正常: {map_name} -> {map_vocab.token2idx[map_name]}")

if __name__ == '__main__':
    check_data() 