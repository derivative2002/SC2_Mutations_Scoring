"""配置管理模块."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import yaml
from pydantic import BaseModel, Field

from .exceptions import ConfigError
from .logger import logger


class AppSettings(BaseModel):
    """应用设置."""
    name: str = "SC2突变评分"
    version: str = "1.0.0"
    debug: bool = False


class ServerSettings(BaseModel):
    """服务器设置."""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = False


class ModelSettings(BaseModel):
    """模型设置."""
    network: Dict[str, Any] = Field(default_factory=lambda: {
        "map_dim": 64,
        "commander_dim": 96,
        "mutation_dim": 96,
        "ai_dim": 64,
        "hidden_dims": [256, 128, 64],
        "num_classes": 5,
        "dropout": 0.3,
        "embed_dropout": 0.2,
        "use_batch_norm": True
    })


class Settings(BaseModel):
    """全局设置."""
    app: AppSettings = Field(default_factory=AppSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)


class Config:
    """配置管理器."""
    _instance: Optional['Config'] = None
    
    def __new__(cls) -> 'Config':
        """单例模式."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化配置."""
        if not getattr(self, '_initialized', False):
            self.settings = Settings()
            self._load_config()
            self._load_resources()
            self._initialized = True
    
    def _load_config(self):
        """加载配置文件."""
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    self.settings = Settings.parse_obj(config_data)
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
                raise ConfigError(f"加载配置文件失败: {str(e)}")
    
    def _load_resources(self):
        """加载资源文件."""
        try:
            # 加载地图数据
            maps_file = Path("resources/data/raw/maps/maps.json")
            if maps_file.exists():
                with open(maps_file, 'r', encoding='utf-8') as f:
                    maps_data = json.load(f)
                    self.maps = []
                    self.map_details = {}
                    for map_info in maps_data:
                        self.maps.append(map_info["name"])
                        self.map_details[map_info["name"]] = {
                            "id": map_info["id"],
                            "description": map_info["description"],
                            "image": map_info["image"]
                        }
            else:
                logger.warning("地图数据文件不存在，使用默认数据")
                self.maps = [
                    "虚空启航", "死亡摇篮", "虚空降临", "锁链末日", "虚空通道",
                    "死亡之夜", "黑暗杀戮", "亡者归来", "虚空恐惧", "死亡审判"
                ]
                self.map_details = {}
            
            # 加载指挥官数据
            commanders_file = Path("resources/data/raw/commanders/commanders.json")
            if commanders_file.exists():
                with open(commanders_file, 'r', encoding='utf-8') as f:
                    commanders_data = json.load(f)
                    self.commanders = []
                    self.commander_details = {}
                    for commander_info in commanders_data:
                        self.commanders.append(commander_info["name"])
                        self.commander_details[commander_info["name"]] = {
                            "id": commander_info["id"],
                            "race": commander_info["race"],
                            "description": commander_info["description"],
                            "image": commander_info["image"]
                        }
            else:
                logger.warning("指挥官数据文件不存在，使用默认数据")
                self.commanders = [
                    "雷诺", "凯瑞甘", "阿塔尼斯", "斯旺", "扎加拉", "沃拉尊",
                    "诺娃", "斯图科夫", "阿拉纳克", "阿巴瑟", "卡拉克斯", "芬尼克斯",
                    "德霍拉", "泰凯斯", "塞特曼", "门格斯克", "泰勒斯", "蒙斯克"
                ]
                self.commander_details = {}
            
            # 加载突变因子数据
            mutations_file = Path("resources/data/raw/mutations/mutations.json")
            if mutations_file.exists():
                with open(mutations_file, 'r', encoding='utf-8') as f:
                    mutations_data = json.load(f)
                    self.mutations = []
                    self.mutation_details = {}
                    for mutation_info in mutations_data:
                        self.mutations.append(mutation_info["name"])
                        self.mutation_details[mutation_info["name"]] = {
                            "id": mutation_info["id"],
                            "description": mutation_info["description"],
                            "image": mutation_info["image"]
                        }
            else:
                logger.warning("突变因子数据文件不存在，使用默认数据")
                self.mutations = [
                    "丧尸大战", "行尸走肉", "虚空裂隙", "暗无天日", "生命吸取",
                    "默哀", "强磁雷场", "闪避机动"
                ]
                self.mutation_details = {}
            
            # 加载规则数据
            rules_file = Path("resources/data/rules/rules.yaml")
            if rules_file.exists():
                with open(rules_file, 'r', encoding='utf-8') as f:
                    self.rules = yaml.safe_load(f)
            else:
                logger.warning("规则数据文件不存在，使用默认数据")
                self.rules = {
                    "incompatible_pairs": [
                        {
                            "mutation1": "虚空裂隙",
                            "mutation2": "暗无天日",
                            "reason": "视野影响叠加，会导致游戏体验极差"
                        },
                        {
                            "mutation1": "生命吸取",
                            "mutation2": "默哀",
                            "reason": "治疗效果相互抵消"
                        }
                    ],
                    "required_pairs": [
                        {
                            "prerequisite": "丧尸大战",
                            "dependent": "行尸走肉",
                            "reason": "机制相互配合"
                        },
                        {
                            "prerequisite": "强磁雷场",
                            "dependent": "闪避机动",
                            "reason": "增加战术深度"
                        }
                    ]
                }
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {str(e)}")
            raise ConfigError(f"JSON格式错误: {str(e)}")
        except yaml.YAMLError as e:
            logger.error(f"YAML格式错误: {str(e)}")
            raise ConfigError(f"YAML格式错误: {str(e)}")
        except Exception as e:
            logger.error(f"加载资源文件失败: {str(e)}")
            raise ConfigError(f"加载资源文件失败: {str(e)}")
    
    def get_maps(self) -> List[str]:
        """获取可用地图列表."""
        return self.maps
    
    def get_map_details(self, map_name: str) -> Dict[str, Any]:
        """获取地图详细信息."""
        return self.map_details.get(map_name, {})
    
    def get_commanders(self) -> List[str]:
        """获取可用指挥官列表."""
        return self.commanders
    
    def get_commander_details(self, commander_name: str) -> Dict[str, Any]:
        """获取指挥官详细信息."""
        return self.commander_details.get(commander_name, {})
    
    def get_mutations(self) -> List[str]:
        """获取可用突变因子列表."""
        return self.mutations
    
    def get_mutation_details(self, mutation_name: str) -> Dict[str, Any]:
        """获取突变因子详细信息."""
        return self.mutation_details.get(mutation_name, {})
    
    def get_incompatible_pairs(self) -> List[Tuple[str, str]]:
        """获取互斥突变对列表."""
        return [(rule["mutation1"], rule["mutation2"]) 
                for rule in self.rules.get("incompatible_pairs", [])]
    
    def get_required_pairs(self) -> List[Tuple[str, str]]:
        """获取依赖突变对列表."""
        return [(rule["prerequisite"], rule["dependent"]) 
                for rule in self.rules.get("required_pairs", [])]
    
    def get_rule_description(self, mutation1: str, mutation2: str) -> str:
        """获取规则说明.
        
        Args:
            mutation1: 第一个突变因子
            mutation2: 第二个突变因子
            
        Returns:
            规则说明文本
        """
        # 检查互斥规则
        for rule in self.rules.get("incompatible_pairs", []):
            if ((rule["mutation1"] == mutation1 and rule["mutation2"] == mutation2) or
                (rule["mutation1"] == mutation2 and rule["mutation2"] == mutation1)):
                return f"互斥规则：{rule['reason']}"
        
        # 检查依赖规则
        for rule in self.rules.get("required_pairs", []):
            if (rule["prerequisite"] == mutation1 and rule["dependent"] == mutation2):
                return f"依赖规则：{rule['reason']}"
            elif (rule["prerequisite"] == mutation2 and rule["dependent"] == mutation1):
                return f"依赖规则：{rule['reason']}"
        
        return "" 