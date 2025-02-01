"""数据加载模块."""

from typing import List, Dict, Any
from ..config import Config

_config = Config()

def load_maps() -> List[Dict[str, Any]]:
    """加载地图数据."""
    return _config.get_maps()

def load_commanders() -> List[Dict[str, Any]]:
    """加载指挥官数据."""
    return _config.get_commanders()

def load_mutations() -> List[Dict[str, Any]]:
    """加载突变因子数据."""
    return _config.get_mutations()

def load_rules() -> Dict[str, Any]:
    """加载规则数据."""
    return _config.get_rules() 