"""生成器单元测试。"""
import pytest
from unittest.mock import Mock, patch

from randomizer.src.backend.rules.generator import MutationGenerator
from randomizer.src.backend.config import Config
from randomizer.src.backend.models.scorer import MutationScorer


@pytest.fixture
def mock_config():
    """创建模拟配置。"""
    config = Mock(spec=Config)
    
    # 设置基本数据
    config.get_maps.return_value = ["虚空降临", "虚空撕裂", "亡者之夜"]
    config.get_commanders.return_value = ["雷诺", "凯瑞甘", "阿塔尼斯"]
    config.get_mutations.return_value = [
        "丧尸大战", "行尸走肉", "虚空裂隙", "暗无天日", 
        "生命吸取", "默哀", "强磁雷场", "闪避机动"
    ]
    
    # 设置规则
    config.get_incompatible_pairs.return_value = [
        ("虚空裂隙", "暗无天日"),
        ("生命吸取", "默哀"),
        ("强磁雷场", "闪避机动")
    ]
    config.get_required_pairs.return_value = [
        ("丧尸大战", "行尸走肉")
    ]
    
    return config


@pytest.fixture
def mock_scorer():
    """创建模拟评分器。"""
    scorer = Mock(spec=MutationScorer)
    
    # 设置预测行为
    def mock_predict(map_name, commanders, mutations, ai_type='standard'):
        # 根据突变数量返回难度分数
        base_score = len(mutations) * 0.8
        if "丧尸大战" in mutations and "行尸走肉" in mutations:
            base_score += 0.5  # 增加依赖组合的难度
        if "虚空裂隙" in mutations and "暗无天日" not in mutations:
            base_score += 0.3  # 增加单个强力突变的难度
        return min(5.0, max(1.0, base_score))
    
    scorer.predict.side_effect = mock_predict
    return scorer


def test_initialization(mock_config, mock_scorer):
    """测试生成器初始化。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    
    assert generator.mode == 'solo'
    assert generator.max_mutations == 4
    assert generator.min_mutations == 2
    
    generator = MutationGenerator('duo', mock_scorer, mock_config)
    assert generator.mode == 'duo'
    assert generator.max_mutations == 8
    assert generator.min_mutations == 4


def test_weighted_sample(mock_config, mock_scorer):
    """测试权重采样。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    
    # 测试基本采样
    mutations = generator._weighted_sample(
        generator.mutations, 
        num_samples=2
    )
    assert len(mutations) == 2
    assert len(set(mutations)) == 2  # 确保没有重复
    
    # 测试考虑当前突变的采样
    current_mutations = ["丧尸大战"]
    for _ in range(5):  # 多次尝试，因为是随机采样
        mutations = generator._weighted_sample(
            generator.mutations,
            num_samples=2,
            current_mutations=current_mutations
        )
        if "行尸走肉" in mutations:
            break
    assert "行尸走肉" in mutations  # 应该倾向于选择依赖项


def test_is_valid_addition(mock_config, mock_scorer):
    """测试突变添加验证。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    
    # 测试互斥规则
    current_mutations = ["虚空裂隙"]
    assert not generator._is_valid_addition("暗无天日", current_mutations)
    
    # 测试依赖规则
    current_mutations = []
    assert not generator._is_valid_addition("行尸走肉", current_mutations)
    
    # 测试有效添加
    current_mutations = ["丧尸大战"]
    assert generator._is_valid_addition("行尸走肉", current_mutations)


def test_optimize_combination(mock_config, mock_scorer):
    """测试组合优化。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    generator.current_map = "虚空降临"
    generator.current_commanders = ["雷诺"]
    
    # 测试向上优化
    initial_mutations = ["丧尸大战", "行尸走肉"]  # 基础分数: 2.1
    target_difficulty = 3.0
    
    mutations, difficulty = generator._optimize_combination(
        initial_mutations,
        target_difficulty,
        tolerance=0.5,
        max_iterations=10
    )
    
    assert len(mutations) > len(initial_mutations)  # 应该添加更多突变
    assert abs(difficulty - target_difficulty) <= 0.5  # 难度应该在容忍范围内
    
    # 测试向下优化
    initial_mutations = ["丧尸大战", "行尸走肉", "虚空裂隙", "强磁雷场"]  # 基础分数: 3.7
    target_difficulty = 2.5
    
    mutations, difficulty = generator._optimize_combination(
        initial_mutations,
        target_difficulty,
        tolerance=0.5,
        max_iterations=10
    )
    
    assert len(mutations) < len(initial_mutations)  # 应该移除一些突变
    assert abs(difficulty - target_difficulty) <= 0.5  # 难度应该在容忍范围内


def test_generate(mock_config, mock_scorer):
    """测试生成完整突变组合。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    
    # 测试指定难度的生成
    result = generator.generate(
        target_difficulty=3.0,
        map_name="虚空降临",
        commanders=["雷诺"],
        tolerance=0.5
    )
    
    assert 2 <= len(result.mutations) <= 4
    assert abs(result.difficulty - 3.0) <= 0.5
    assert result.map_name == "虚空降临"
    assert result.commanders == ["雷诺"]
    
    # 测试随机生成
    result = generator.generate(target_difficulty=3.0)
    
    assert 2 <= len(result.mutations) <= 4
    assert abs(result.difficulty - 3.0) <= 0.5
    assert result.map_name in mock_config.get_maps()
    assert len(result.commanders) == 1
    assert result.commanders[0] in mock_config.get_commanders()


def test_generate_invalid_input(mock_config, mock_scorer):
    """测试无效输入处理。"""
    generator = MutationGenerator('solo', mock_scorer, mock_config)
    
    # 测试无效难度
    with pytest.raises(ValueError):
        generator.generate(target_difficulty=0.5)
    
    with pytest.raises(ValueError):
        generator.generate(target_difficulty=5.5)
    
    # 测试无效指挥官数量
    with pytest.raises(ValueError):
        generator.generate(
            target_difficulty=3.0,
            commanders=["雷诺", "凯瑞甘"]  # solo模式不允许两个指挥官
        ) 