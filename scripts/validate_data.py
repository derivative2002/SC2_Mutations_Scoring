"""数据验证脚本。"""
import json
import yaml
from pathlib import Path
import os


def validate_image_paths(data_file: Path, image_dir: Path):
    """验证图片路径是否存在。"""
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    errors = []
    for item in data:
        image_path = item['image']
        # 从完整路径中提取文件名
        file_name = Path(image_path).name
        full_path = image_dir / file_name
        
        if not full_path.exists():
            errors.append(f"图片不存在: {file_name}")
    
    return errors


def validate_rules(rules_file: Path, mutations_file: Path):
    """验证规则文件中的突变名称是否存在。"""
    # 加载突变数据
    with open(mutations_file, 'r', encoding='utf-8') as f:
        mutations = json.load(f)
    mutation_names = {m['name'] for m in mutations}
    
    # 加载规则数据
    with open(rules_file, 'r', encoding='utf-8') as f:
        rules = yaml.safe_load(f)
    
    errors = []
    
    # 检查互斥规则
    for rule in rules.get('incompatible_pairs', []):
        if rule['mutation1'] not in mutation_names:
            errors.append(f"互斥规则中的突变不存在: {rule['mutation1']}")
        if rule['mutation2'] not in mutation_names:
            errors.append(f"互斥规则中的突变不存在: {rule['mutation2']}")
    
    # 检查依赖规则
    for rule in rules.get('required_pairs', []):
        if rule['prerequisite'] not in mutation_names:
            errors.append(f"依赖规则中的前置突变不存在: {rule['prerequisite']}")
        if rule['dependent'] not in mutation_names:
            errors.append(f"依赖规则中的依赖突变不存在: {rule['dependent']}")
    
    return errors


def validate_data_consistency():
    """验证数据一致性。"""
    root_dir = Path(__file__).parent.parent
    resources_dir = root_dir / 'resources'
    
    # 验证地图数据
    maps_errors = validate_image_paths(
        resources_dir / 'data/raw/maps/maps.json',
        resources_dir / 'images/maps'
    )
    
    # 验证指挥官数据
    commanders_errors = validate_image_paths(
        resources_dir / 'data/raw/commanders/commanders.json',
        resources_dir / 'images/commanders'
    )
    
    # 验证突变数据
    mutations_errors = validate_image_paths(
        resources_dir / 'data/raw/mutations/mutations.json',
        resources_dir / 'images/mutations'
    )
    
    # 验证规则数据
    rules_errors = validate_rules(
        resources_dir / 'data/rules/rules.yaml',
        resources_dir / 'data/raw/mutations/mutations.json'
    )
    
    # 汇总错误
    all_errors = {
        'maps': maps_errors,
        'commanders': commanders_errors,
        'mutations': mutations_errors,
        'rules': rules_errors
    }
    
    return all_errors


def main():
    """主函数。"""
    print("开始验证数据...")
    
    errors = validate_data_consistency()
    has_error = False
    
    for category, category_errors in errors.items():
        if category_errors:
            has_error = True
            print(f"\n{category} 错误:")
            for error in category_errors:
                print(f"  - {error}")
    
    if not has_error:
        print("\n数据验证通过！")
    else:
        print("\n数据验证失败！")
        exit(1)


if __name__ == "__main__":
    main() 