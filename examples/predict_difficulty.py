"""难度评分示例。"""
from scoring.src.models.scorer import MutationScorer


def main():
    """主函数。"""
    # 初始化评分器
    scorer = MutationScorer()
    
    # 准备输入数据
    map_name = "虚空降临"
    commanders = ["雷诺"]
    mutations = ["丧尸大战", "行尸走肉"]
    
    # 预测难度
    score = scorer.predict(
        map_name=map_name,
        commanders=commanders,
        mutations=mutations
    )
    
    # 打印结果
    print(f"地图: {map_name}")
    print(f"指挥官: {commanders}")
    print(f"突变: {mutations}")
    print(f"预测难度: {score:.2f}")


if __name__ == "__main__":
    main() 