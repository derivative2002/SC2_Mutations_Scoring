"""配置管理模块。"""
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import yaml
from pydantic import BaseModel

from .exceptions import ConfigError
from .logger import logger


class AppSettings(BaseModel):
    """应用程序设置。"""
    name: str = "SC2 Mutations Randomizer"
    version: str = "0.1.0"
    description: str = "星际争霸2合作任务突变组合生成器"
    debug: bool = False


class ModelSettings(BaseModel):
    """模型设置。"""
    vocab_dir: str = "resources/model/vocab"
    weights_path: str = "resources/model/weights/model.pt"
    network: Dict = {
        "embed_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": 0.1
    }


class Settings(BaseModel):
    """全局设置。"""
    app: AppSettings = AppSettings()
    model: ModelSettings = ModelSettings()


class MutationRule(BaseModel):
    """突变规则。"""
    mutation1: str
    mutation2: str
    reason: str
    example: str


class DependencyRule(BaseModel):
    """依赖规则。"""
    prerequisite: str
    dependent: str
    reason: str
    example: str


class Rules(BaseModel):
    """规则集合。"""
    incompatible_pairs: List[MutationRule] = []
    required_pairs: List[DependencyRule] = []


class Config:
    """配置管理类。"""
    
    def __init__(self):
        """初始化配置。"""
        self.settings = Settings()
        self.resource_dir = Path(__file__).parent.parent.parent / 'resources' / 'data'
        self._load_resources()
    
    def _load_resources(self):
        """加载资源文件。"""
        try:
            # 加载地图数据
            with open(self.resource_dir / 'maps.json', 'r', encoding='utf-8') as f:
                self.maps = json.load(f)
            
            # 加载指挥官数据
            with open(self.resource_dir / 'commanders.json', 'r', encoding='utf-8') as f:
                self.commanders = json.load(f)
            
            # 加载突变因子数据
            with open(self.resource_dir / 'mutations.json', 'r', encoding='utf-8') as f:
                self.mutations = json.load(f)
            
            # 加载规则数据
            with open(self.resource_dir / 'rules.yaml', 'r', encoding='utf-8') as f:
                rules_data = yaml.safe_load(f)
                
                # 转换规则格式
                incompatible_pairs = []
                for pair in rules_data.get('incompatible_pairs', []):
                    incompatible_pairs.append(MutationRule(
                        mutation1=pair['mutation1'],
                        mutation2=pair['mutation2'],
                        reason=pair['reason'],
                        example=pair['example']
                    ))
                
                required_pairs = []
                for pair in rules_data.get('required_pairs', []):
                    required_pairs.append(DependencyRule(
                        prerequisite=pair['prerequisite'],
                        dependent=pair['dependent'],
                        reason=pair['reason'],
                        example=pair['example']
                    ))
                
                self.rules = Rules(
                    incompatible_pairs=incompatible_pairs,
                    required_pairs=required_pairs
                )
            
            logger.info("配置加载成功")
            
        except FileNotFoundError as e:
            logger.error(f"无法加载配置文件: {str(e)}")
            raise ConfigError(f"无法加载配置文件: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误: {str(e)}")
            raise ConfigError(f"JSON格式错误: {str(e)}")
        except yaml.YAMLError as e:
            logger.error(f"YAML格式错误: {str(e)}")
            raise ConfigError(f"YAML格式错误: {str(e)}")
        except Exception as e:
            logger.error(f"配置加载失败: {str(e)}")
            raise ConfigError(f"配置加载失败: {str(e)}")
    
    def get_maps(self) -> List[str]:
        """获取地图列表。"""
        return [map_data['name'] for map_data in self.maps]
    
    def get_commanders(self) -> List[str]:
        """获取指挥官列表。"""
        return [commander['name'] for commander in self.commanders]
    
    def get_mutations(self) -> List[str]:
        """获取突变因子列表。"""
        return [mutation['name'] for mutation in self.mutations]
    
    def get_incompatible_pairs(self) -> List[Tuple[str, str]]:
        """获取互斥突变对。"""
        return [(rule.mutation1, rule.mutation2) 
                for rule in self.rules.incompatible_pairs]
    
    def get_required_pairs(self) -> List[Tuple[str, str]]:
        """获取依赖突变对。"""
        return [(rule.prerequisite, rule.dependent) 
                for rule in self.rules.required_pairs]
    
    def get_rule_description(self, mutation1: str, mutation2: str) -> str:
        """获取规则说明。
        
        Args:
            mutation1: 第一个突变因子
            mutation2: 第二个突变因子
            
        Returns:
            规则说明
        """
        # 检查互斥规则
        for rule in self.rules.incompatible_pairs:
            if ((rule.mutation1 == mutation1 and rule.mutation2 == mutation2) or
                (rule.mutation1 == mutation2 and rule.mutation2 == mutation1)):
                return f"互斥规则：{rule.reason}。{rule.example}"
        
        # 检查依赖规则
        for rule in self.rules.required_pairs:
            if ((rule.prerequisite == mutation1 and rule.dependent == mutation2) or
                (rule.prerequisite == mutation2 and rule.dependent == mutation1)):
                return f"依赖规则：{rule.reason}。{rule.example}"
        
        return "" 