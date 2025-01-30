"""突变组合生成示例。"""
from randomizer.src.rules.generator import MutationGenerator
from randomizer.src.models.scorer import get_scorer
from randomizer.src.config import Config


def main():
    """主函数。"""
    # 初始化配置
    config = Config()
    
    # 获取评分器
    scorer = get_scorer()
    
    # 创建生成器
    generator = MutationGenerator('solo', scorer, config)
    
    # 生成突变组合
    result = generator.generate(
        target_difficulty=3.0,
        map_name="虚空降临",
        commanders=["雷诺"],
        tolerance=0.5
    )
    
    # 打印结果
    print(f"地图: {result.map_name}")
    print(f"指挥官: {result.commanders}")
    print(f"突变: {result.mutations}")
    print(f"难度: {result.difficulty:.2f}")


if __name__ == "__main__":
    main() 