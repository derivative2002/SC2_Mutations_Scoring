"""修复CSV文件格式问题."""

import pandas as pd
import csv

def fix_csv_file():
    # 读取CSV文件，使用更严格的解析设置
    df = pd.read_csv(
        'data/processed/sc2_mutations_duo.csv',
        quoting=csv.QUOTE_ALL,  # 对所有字段使用引号
        escapechar='\\',        # 使用反斜杠作为转义字符
        encoding='utf-8',       # 明确指定编码
        on_bad_lines='warn'     # 对问题行发出警告
    )
    
    # 确保所有列都存在
    required_columns = [
        'task_id', 'map_name', 'mutation_factors', 'commanders', 
        'enemy_ai', 'difficulty_score', 'notes', 'is_verified', 
        'verifier', 'battle_experience', '是否为裂隙突变', 'commander_count'
    ]
    
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    # 只保留需要的列并按顺序排列
    df = df[required_columns]
    
    # 保存修复后的文件
    df.to_csv(
        'data/processed/sc2_mutations_duo_fixed.csv',
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar='\\',
        encoding='utf-8'
    )
    
    print('文件已修复，行数:', len(df))
    print('列名:', list(df.columns))
    print('唯一地图:', sorted(df['map_name'].unique()))
    
    # 检查数据完整性
    print('\n数据完整性检查:')
    print('map_name为空的行数:', df['map_name'].isna().sum())
    print('commanders为空的行数:', df['commanders'].isna().sum())
    print('mutation_factors为空的行数:', df['mutation_factors'].isna().sum())
    print('enemy_ai为空的行数:', df['enemy_ai'].isna().sum())
    print('difficulty_score为空的行数:', df['difficulty_score'].isna().sum())

if __name__ == '__main__':
    fix_csv_file() 