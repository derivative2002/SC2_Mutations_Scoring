"""配置模块."""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from types import SimpleNamespace

from .exceptions import ConfigError
from .logger import logger

class Config:
    """配置类."""
    
    def __init__(self):
        """初始化配置."""
        self.root_dir = Path(__file__).resolve().parent.parent.parent.parent
        self.data_dir = self.root_dir / 'resources' / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.maps_dir = self.raw_dir / 'maps'
        self.commanders_dir = self.raw_dir / 'commanders'
        self.mutations_dir = self.raw_dir / 'mutations'
        self.processed_data_dir = self.data_dir / 'processed'
        
        # 验证目录结构
        self._verify_directories()
        
        # 加载配置文件
        self.settings = self._load_settings()
        
        # 记录配置类型和内容
        logger.info(f"配置类型: {type(self.settings)}")
        logger.info(f"配置内容: {self.settings}")
        
    def _dict_to_namespace(self, d: Dict) -> SimpleNamespace:
        """将字典转换为对象."""
        if not isinstance(d, dict):
            return d
        for k, v in d.items():
            if isinstance(v, dict):
                d[k] = self._dict_to_namespace(v)
        return SimpleNamespace(**d)
        
    def _verify_directories(self):
        """验证目录结构是否正确."""
        required_dirs = [
            self.data_dir,
            self.raw_dir,
            self.maps_dir,
            self.commanders_dir,
            self.mutations_dir,
            self.processed_data_dir
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                logger.warning(f"目录不存在: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"已创建目录: {dir_path}")
        
    def _load_settings(self) -> Dict[str, Any]:
        """加载配置文件."""
        try:
            settings_path = self.root_dir / 'config' / 'settings.json'
            logger.info(f"尝试加载配置文件: {settings_path}")
            if not settings_path.exists():
                logger.warning(f"配置文件不存在: {settings_path}")
                return {}
            
            with open(settings_path, encoding='utf-8') as f:
                settings = json.load(f)
                logger.info(f"成功加载配置: {settings}")
                return settings
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise ConfigError(f"加载配置文件失败: {str(e)}")
            
    def get_maps(self) -> List[Dict[str, Any]]:
        """获取地图列表."""
        try:
            maps_path = self.maps_dir / 'maps.json'
            if not maps_path.exists():
                logger.error(f"地图文件不存在: {maps_path}")
                raise ConfigError(f"地图文件不存在: {maps_path}")
            with open(maps_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载地图列表失败: {str(e)}")
            raise ConfigError(f"加载地图列表失败: {str(e)}")
            
    def get_commanders(self) -> List[Dict[str, Any]]:
        """获取指挥官列表."""
        try:
            commanders_path = self.commanders_dir / 'commanders.json'
            if not commanders_path.exists():
                logger.error(f"指挥官文件不存在: {commanders_path}")
                raise ConfigError(f"指挥官文件不存在: {commanders_path}")
            with open(commanders_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载指挥官列表失败: {str(e)}")
            raise ConfigError(f"加载指挥官列表失败: {str(e)}")
            
    def get_mutations(self) -> List[Dict[str, Any]]:
        """获取突变因子列表."""
        try:
            mutations_path = self.mutations_dir / 'mutations.json'
            if not mutations_path.exists():
                logger.error(f"突变因子文件不存在: {mutations_path}")
                raise ConfigError(f"突变因子文件不存在: {mutations_path}")
            with open(mutations_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载突变因子列表失败: {str(e)}")
            raise ConfigError(f"加载突变因子列表失败: {str(e)}")
            
    def get_rules(self) -> Dict[str, Any]:
        """获取规则列表."""
        try:
            rules_path = self.mutations_dir / 'rules.json'
            if not rules_path.exists():
                logger.error(f"规则文件不存在: {rules_path}")
                raise ConfigError(f"规则文件不存在: {rules_path}")
            with open(rules_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载规则列表失败: {str(e)}")
            raise ConfigError(f"加载规则列表失败: {str(e)}")
            
    def get_generation_rules(self) -> Dict[str, Any]:
        """获取生成规则."""
        try:
            rules_path = self.mutations_dir / 'generation_rules.json'
            if not rules_path.exists():
                logger.error(f"生成规则文件不存在: {rules_path}")
                raise ConfigError(f"生成规则文件不存在: {rules_path}")
            with open(rules_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载生成规则失败: {str(e)}")
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
    
    def get_map_details(self, map_name: str) -> Dict[str, Any]:
        """获取地图详情."""
        try:
            maps = self.get_maps()
            for map_data in maps:
                if map_data["name"] == map_name:
                    return {k: v for k, v in map_data.items() if k != "name"}
            return {}
        except Exception as e:
            logger.error(f"获取地图详情失败: {str(e)}")
            return {}
    
    def get_commander_details(self, commander_name: str) -> Dict[str, Any]:
        """获取指挥官详情."""
        try:
            commanders = self.get_commanders()
            for commander_data in commanders:
                if commander_data["name"] == commander_name:
                    return {k: v for k, v in commander_data.items() if k != "name"}
            return {}
        except Exception as e:
            logger.error(f"获取指挥官详情失败: {str(e)}")
            return {}
    
    def get_mutation_details(self, mutation_name: str) -> Dict[str, Any]:
        """获取突变因子详情."""
        try:
            mutations = self.get_mutations()
            for mutation_data in mutations:
                if mutation_data["name"] == mutation_name:
                    return {k: v for k, v in mutation_data.items() if k != "name"}
            return {}
        except Exception as e:
            logger.error(f"获取突变因子详情失败: {str(e)}")
            return {} 