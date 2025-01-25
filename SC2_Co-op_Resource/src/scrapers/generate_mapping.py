import os
import csv
from PIL import Image

def generate_mapping():
    """生成因子名称和图片的映射表"""
    # 获取项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # 图片目录
    mutations_dir = os.path.join(project_root, "src", "resources", "images", "mutations")
    maps_dir = os.path.join(project_root, "src", "resources", "images", "maps")
    
    # 输出目录
    output_dir = os.path.join(project_root, "src", "resources", "mappings")
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建映射表
    mutation_mappings = []
    map_mappings = []
    
    # 处理突变因子图片
    if os.path.exists(mutations_dir):
        for filename in sorted(os.listdir(mutations_dir)):
            if filename.endswith('.png'):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(mutations_dir, filename)
                
                try:
                    # 读取图片获取尺寸
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                    mutation_mappings.append({
                        '因子名称': name,
                        '图片文件名': filename,
                        '图片路径': f'src/resources/images/mutations/{filename}',
                        '图片尺寸': f'{width}x{height}'
                    })
                except Exception as e:
                    print(f"处理图片失败：{filename}，错误：{str(e)}")
    
    # 处理地图图片
    if os.path.exists(maps_dir):
        for filename in sorted(os.listdir(maps_dir)):
            if filename.endswith('.png'):
                name = os.path.splitext(filename)[0]
                img_path = os.path.join(maps_dir, filename)
                
                try:
                    # 读取图片获取尺寸
                    with Image.open(img_path) as img:
                        width, height = img.size
                        
                    map_mappings.append({
                        '地图名称': name,
                        '图片文件名': filename,
                        '图片路径': f'src/resources/images/maps/{filename}',
                        '图片尺寸': f'{width}x{height}'
                    })
                except Exception as e:
                    print(f"处理图片失败：{filename}，错误：{str(e)}")
    
    # 保存突变因子映射表
    if mutation_mappings:
        mutations_csv = os.path.join(output_dir, 'mutation_mappings.csv')
        with open(mutations_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['因子名称', '图片文件名', '图片路径', '图片尺寸'])
            writer.writeheader()
            writer.writerows(mutation_mappings)
        print(f"突变因子映射表已保存到：{mutations_csv}")
        print(f"共 {len(mutation_mappings)} 个突变因子")
    
    # 保存地图映射表
    if map_mappings:
        maps_csv = os.path.join(output_dir, 'map_mappings.csv')
        with open(maps_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['地图名称', '图片文件名', '图片路径', '图片尺寸'])
            writer.writeheader()
            writer.writerows(map_mappings)
        print(f"地图映射表已保存到：{maps_csv}")
        print(f"共 {len(map_mappings)} 个地图")

if __name__ == "__main__":
    generate_mapping() 