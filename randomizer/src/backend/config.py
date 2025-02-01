"""配置模块."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from .exceptions import ConfigError

class Config:
    """配置类."""
    
    def __init__(self):
        """初始化配置."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.data_dir = self.root_dir / 'resources' / 'data'
        self.raw_data_dir = self.data_dir / 'raw'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # 加载配置文件
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict[str, Any]:
        """加载配置文件."""
        try:
            settings_path = self.root_dir / 'config' / 'settings.json'
            if not settings_path.exists():
                return {}
            
            with open(settings_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载配置文件失败: {str(e)}")
            
    def get_maps(self) -> List[Dict[str, Any]]:
        """获取地图列表."""
        try:
            maps_path = self.raw_data_dir / 'maps.json'
            with open(maps_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载地图列表失败: {str(e)}")
            
    def get_commanders(self) -> List[Dict[str, Any]]:
        """获取指挥官列表."""
        try:
            commanders_path = self.raw_data_dir / 'commanders.json'
            with open(commanders_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载指挥官列表失败: {str(e)}")
            
    def get_mutations(self) -> List[Dict[str, Any]]:
        """获取突变因子列表."""
        try:
            mutations_path = self.raw_data_dir / 'mutations.json'
            with open(mutations_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载突变因子列表失败: {str(e)}")
            
    def get_rules(self) -> Dict[str, Any]:
        """获取规则列表."""
        try:
            rules_path = self.raw_data_dir / 'mutations' / 'rules.json'
            with open(rules_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载规则列表失败: {str(e)}")
            
    def get_generation_rules(self) -> Dict[str, Any]:
        """获取生成规则."""
        try:
            rules_path = self.raw_data_dir / 'mutations' / 'generation_rules.json'
            with open(rules_path) as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"加载生成规则失败: {str(e)}")
            
    def get_incompatible_pairs(self) -> List[tuple[str, str]]:
        """获取互斥突变因子对."""
        rules = self.get_rules()
        return [(rule["mutation1"], rule["mutation2"]) 
                for rule in rules.get("incompatible_pairs", [])]
                
    def get_required_pairs(self) -> List[tuple[str, str]]:
        """获取依赖突变因子对."""
        rules = self.get_rules()
        return [(rule["prerequisite"], rule["dependent"]) 
                for rule in rules.get("required_pairs", [])]
                
    def get_rule_description(self, m1: str, m2: str) -> Optional[str]:
        """获取规则描述."""
        rules = self.get_rules()
        
        # 检查互斥规则
        for rule in rules.get("incompatible_pairs", []):
            if ((rule["mutation1"] == m1 and rule["mutation2"] == m2) or
                (rule["mutation1"] == m2 and rule["mutation2"] == m1)):
                return rule.get("description")
                
        # 检查依赖规则
        for rule in rules.get("required_pairs", []):
            if ((rule["prerequisite"] == m1 and rule["dependent"] == m2) or
                (rule["prerequisite"] == m2 and rule["dependent"] == m1)):
                return rule.get("description")
                
        return None 