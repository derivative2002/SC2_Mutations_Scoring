"""配置管理单元测试。"""
import pytest
import json
import yaml
from pathlib import Path
from unittest.mock import mock_open, patch, MagicMock

from randomizer.src.backend.config import Config, MutationRule, DependencyRule
from randomizer.src.backend.exceptions import ConfigError


@pytest.fixture
def mock_json_data():
    """模拟JSON数据。"""
    return {
        "maps": [
            {
                "name": "虚空降临",
                "id": "void_launch",
                "description": "保护并发射虚空水晶塔。"
            }
        ],
        "commanders": [
            {
                "name": "雷诺",
                "id": "raynor",
                "race": "terran"
            }
        ],
        "mutations": [
            {
                "name": "丧尸大战",
                "id": "32",
                "image": "images/mutations/丧尸大战.png"
            }
        ]
    }


@pytest.fixture
def mock_yaml_data():
    """模拟YAML数据。"""
    return {
        "incompatible_pairs": [
            {
                "mutation1": "虚空裂隙",
                "mutation2": "暗无天日",
                "reason": "视野影响叠加",
                "example": "示例说明"
            }
        ],
        "required_pairs": [
            {
                "prerequisite": "丧尸大战",
                "dependent": "行尸走肉",
                "reason": "机制依赖",
                "example": "示例说明"
            }
        ]
    }


def test_config_initialization(mock_json_data, mock_yaml_data):
    """测试配置初始化。"""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load") as mock_yaml_load:
                # 设置模拟返回值
                mock_json_load.side_effect = [
                    mock_json_data["maps"],
                    mock_json_data["commanders"],
                    mock_json_data["mutations"]
                ]
                mock_yaml_load.return_value = mock_yaml_data
                
                config = Config()
                
                # 验证文件读取
                assert mock_file.call_count >= 4  # 至少读取4个文件
                
                # 验证数据加载
                assert len(config.maps) == 1
                assert len(config.commanders) == 1
                assert len(config.mutations) == 1
                assert len(config.rules.incompatible_pairs) == 1
                assert len(config.rules.required_pairs) == 1


def test_get_maps(mock_json_data):
    """测试获取地图列表。"""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load"):
                mock_json_load.side_effect = [
                    mock_json_data["maps"],
                    mock_json_data["commanders"],
                    mock_json_data["mutations"]
                ]
                
                config = Config()
                maps = config.get_maps()
                
                assert isinstance(maps, list)
                assert len(maps) == 1
                assert maps[0] == "虚空降临"


def test_get_commanders(mock_json_data):
    """测试获取指挥官列表。"""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load"):
                mock_json_load.side_effect = [
                    mock_json_data["maps"],
                    mock_json_data["commanders"],
                    mock_json_data["mutations"]
                ]
                
                config = Config()
                commanders = config.get_commanders()
                
                assert isinstance(commanders, list)
                assert len(commanders) == 1
                assert commanders[0] == "雷诺"


def test_get_mutations(mock_json_data):
    """测试获取突变因子列表。"""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load"):
                mock_json_load.side_effect = [
                    mock_json_data["maps"],
                    mock_json_data["commanders"],
                    mock_json_data["mutations"]
                ]
                
                config = Config()
                mutations = config.get_mutations()
                
                assert isinstance(mutations, list)
                assert len(mutations) == 1
                assert mutations[0] == "丧尸大战"


def test_get_rule_description(mock_json_data, mock_yaml_data):
    """测试获取规则说明。"""
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load") as mock_yaml_load:
                mock_json_load.side_effect = [
                    mock_json_data["maps"],
                    mock_json_data["commanders"],
                    mock_json_data["mutations"]
                ]
                mock_yaml_load.return_value = mock_yaml_data
                
                config = Config()
                
                # 测试互斥规则
                desc = config.get_rule_description("虚空裂隙", "暗无天日")
                assert "互斥" in desc
                assert "视野影响叠加" in desc
                
                # 测试依赖规则
                desc = config.get_rule_description("丧尸大战", "行尸走肉")
                assert "依赖" in desc
                assert "机制依赖" in desc
                
                # 测试无规则
                desc = config.get_rule_description("未知突变1", "未知突变2")
                assert desc == ""


def test_error_handling():
    """测试错误处理。"""
    # 测试文件不存在
    with patch("builtins.open") as mock_file:
        mock_file.side_effect = FileNotFoundError()
        
        with pytest.raises(ConfigError) as exc_info:
            Config()
        assert "无法加载配置文件" in str(exc_info.value)
    
    # 测试JSON格式错误
    with patch("builtins.open", mock_open()):
        with patch("json.load") as mock_json_load:
            mock_json_load.side_effect = json.JSONDecodeError("", "", 0)
            
            with pytest.raises(ConfigError) as exc_info:
                Config()
            assert "JSON格式错误" in str(exc_info.value)
    
    # 测试YAML格式错误
    with patch("builtins.open", mock_open()) as mock_file:
        with patch("json.load") as mock_json_load:
            with patch("yaml.safe_load") as mock_yaml_load:
                # 设置JSON加载成功
                mock_json_load.side_effect = [
                    {"maps": []},
                    {"commanders": []},
                    {"mutations": []}
                ]
                # 设置YAML加载失败
                mock_yaml_load.side_effect = yaml.YAMLError()
                
                with pytest.raises(ConfigError) as exc_info:
                    Config()
                assert "YAML格式错误" in str(exc_info.value) 