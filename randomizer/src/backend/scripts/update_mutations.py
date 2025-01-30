"""更新突变因子数据脚本。"""
import json
from pathlib import Path

def load_mutation_vocab() -> dict:
    """加载突变因子词表。"""
    vocab_path = Path('data/processed/vocabs/mutation_vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab['token2idx']

def update_mutations():
    """更新突变因子数据。"""
    # 获取图片目录
    image_dir = Path('SC2_Co-op_Resource/src/resources/images/mutations')
    output_file = Path('randomizer/resources/data/mutations.json')
    
    # 加载现有词表
    token2idx = load_mutation_vocab()
    
    # 获取所有突变因子名称和ID
    mutations = []
    for image_file in image_dir.glob('*.png'):
        name = image_file.stem
        if name == '.DS_Store':
            continue
            
        # 使用词表中的ID，如果不存在则跳过
        if name not in token2idx:
            print(f'警告：突变因子 "{name}" 不在词表中')
            continue
            
        mutations.append({
            'name': name,
            'id': str(token2idx[name]),  # 使用数字ID
            'image': str(image_file.relative_to(Path('SC2_Co-op_Resource/src/resources')))
        })
    
    # 按名称排序
    mutations.sort(key=lambda x: x['name'])
    
    # 保存映射表
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mutations, f, ensure_ascii=False, indent=4)
    
    print(f'已更新 {len(mutations)} 个突变因子数据')

if __name__ == '__main__':
    update_mutations() 
