"""评分器单元测试。"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import torch

from randomizer.src.backend.models.scorer import MutationScorer
from randomizer.src.backend.exceptions import ModelError


@pytest.fixture
def mock_vocab():
    """创建模拟词表。"""
    vocab = MagicMock()
    vocab.pad_id = 0
    
    # 设置词表映射
    vocab_map = {
        "虚空降临": 1,
        "雷诺": 1,
        "凯瑞甘": 2,
        "丧尸大战": 1,
        "行尸走肉": 2,
        "standard": 1
    }
    vocab.__getitem__.side_effect = lambda x: vocab_map.get(x, 0)
    vocab.__len__.return_value = 10
    
    return vocab


@pytest.fixture
def mock_model():
    """创建模拟模型。"""
    model = Mock()
    
    def mock_forward(**kwargs):
        # 返回一个简单的logits张量
        return torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]])
    
    model.forward.side_effect = mock_forward
    model.eval.return_value = None
    
    return model


@pytest.fixture
def mock_config():
    """创建模拟配置。"""
    config = Mock()
    config.settings.model.vocab_dir = "dummy/path"
    config.settings.model.weights_path = "dummy/path/model.pt"
    config.settings.model.network = {
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1
    }
    return config


@patch('randomizer.src.backend.models.scorer.Config')
@patch('randomizer.src.backend.models.scorer.Vocab')
@patch('randomizer.src.backend.models.scorer.BaseScorer')
def test_initialization(MockBaseScorer, MockVocab, MockConfig, 
                       mock_vocab, mock_model, mock_config):
    """测试评分器初始化。"""
    # 设置模拟对象
    MockConfig.return_value = mock_config
    MockVocab.load.return_value = mock_vocab
    MockBaseScorer.return_value = mock_model
    
    # 初始化评分器
    scorer = MutationScorer(cache_size=100, test_mode=True)
    
    # 验证初始化
    assert scorer.cache_size == 100
    assert len(scorer.cache) == 0
    assert scorer.model is None  # 测试模式下不加载模型
    assert scorer.test_mode is True


def test_cache_management(mock_vocab, mock_model, mock_config):
    """测试缓存管理。"""
    with patch('randomizer.src.backend.models.scorer.Config', return_value=mock_config), \
         patch('randomizer.src.backend.models.scorer.Vocab') as MockVocab, \
         patch('randomizer.src.backend.models.scorer.BaseScorer', return_value=mock_model):
        
        MockVocab.load.return_value = mock_vocab
        scorer = MutationScorer(cache_size=2, test_mode=True)
        
        # 添加测试数据
        key1 = "test_key1"
        key2 = "test_key2"
        key3 = "test_key3"
        
        scorer._manage_cache(key1, 3.0)
        assert len(scorer.cache) == 1
        assert scorer.cache[key1] == 3.0
        
        scorer._manage_cache(key2, 4.0)
        assert len(scorer.cache) == 2
        assert key1 in scorer.cache
        assert key2 in scorer.cache
        
        # 测试缓存大小限制
        scorer._manage_cache(key3, 5.0)
        assert len(scorer.cache) == 2
        assert key1 not in scorer.cache  # 最早的项应该被移除
        assert key2 in scorer.cache
        assert key3 in scorer.cache


@patch('randomizer.src.backend.models.scorer.Config')
@patch('randomizer.src.backend.models.scorer.Vocab')
@patch('randomizer.src.backend.models.scorer.BaseScorer')
def test_predict(MockBaseScorer, MockVocab, MockConfig,
                mock_vocab, mock_model, mock_config):
    """测试预测功能。"""
    # 设置模拟对象
    MockConfig.return_value = mock_config
    MockVocab.load.return_value = mock_vocab
    MockBaseScorer.return_value = mock_model
    
    scorer = MutationScorer(test_mode=True)
    
    # 测试正常预测
    score = scorer.predict(
        map_name="虚空降临",
        commanders=["雷诺"],
        mutations=["丧尸大战", "行尸走肉"]
    )
    
    assert isinstance(score, float)
    assert 1.0 <= score <= 5.0
    
    # 测试缓存命中
    cached_score = scorer.predict(
        map_name="虚空降临",
        commanders=["雷诺"],
        mutations=["丧尸大战", "行尸走肉"]
    )
    
    assert score == cached_score
    assert mock_model.forward.call_count == 0  # 测试模式下不调用模型


@patch('randomizer.src.backend.models.scorer.Config')
@patch('randomizer.src.backend.models.scorer.Vocab')
@patch('randomizer.src.backend.models.scorer.BaseScorer')
def test_predict_errors(MockBaseScorer, MockVocab, MockConfig,
                       mock_vocab, mock_model, mock_config):
    """测试预测错误处理。"""
    # 设置模拟对象
    MockConfig.return_value = mock_config
    MockVocab.load.return_value = mock_vocab
    MockBaseScorer.return_value = mock_model
    
    scorer = MutationScorer(test_mode=True)
    
    # 测试空突变列表
    score = scorer.predict(
        map_name="虚空降临",
        commanders=["雷诺"],
        mutations=[]
    )
    assert score == 1.0
    
    # 测试最大突变数量
    score = scorer.predict(
        map_name="虚空降临",
        commanders=["雷诺"],
        mutations=["丧尸大战"] * 10
    )
    assert score == 5.0


def test_cache_key_generation(mock_vocab, mock_model, mock_config):
    """测试缓存键生成。"""
    with patch('randomizer.src.backend.models.scorer.Config', return_value=mock_config), \
         patch('randomizer.src.backend.models.scorer.Vocab') as MockVocab, \
         patch('randomizer.src.backend.models.scorer.BaseScorer', return_value=mock_model):
        
        MockVocab.load.return_value = mock_vocab
        scorer = MutationScorer(test_mode=True)
        
        # 测试相同输入生成相同的键
        key1 = scorer._generate_cache_key(
            map_name="虚空降临",
            commanders=["雷诺", "凯瑞甘"],
            mutations=["丧尸大战", "行尸走肉"],
            ai_type="standard"
        )
        
        key2 = scorer._generate_cache_key(
            map_name="虚空降临",
            commanders=["雷诺", "凯瑞甘"],
            mutations=["丧尸大战", "行尸走肉"],
            ai_type="standard"
        )
        
        assert key1 == key2
        
        # 测试顺序无关性
        key3 = scorer._generate_cache_key(
            map_name="虚空降临",
            commanders=["凯瑞甘", "雷诺"],  # 顺序改变
            mutations=["行尸走肉", "丧尸大战"],  # 顺序改变
            ai_type="standard"
        )
        
        assert key1 == key3 