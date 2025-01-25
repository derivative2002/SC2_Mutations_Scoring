"""词表构建脚本."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VocabBuilder:
    """词表构建器."""
    
    def __init__(self, resource_dir: str = "SC2_Co-op_Resource"):
        """初始化词表构建器.
        
        Args:
            resource_dir: 资源文件目录
        """
        self.resource_dir = Path(resource_dir)
        self.maps: List[str] = []
        self.commanders: List[str] = []
        self.mutations: List[str] = []
        self._load_resources()
    
    def _load_resources(self):
        """从资源文件加载标准数据."""
        try:
            # 加载地图数据
            map_file = self.resource_dir / "src/resources/mappings/map_mappings.csv"
            with open(map_file, 'r', encoding='utf-8') as f:
                # 跳过标题行
                next(f)
                self.maps = [line.split(',')[0] for line in f]
            logger.info(f"加载了 {len(self.maps)} 个地图")
            
            # 加载突变因子数据
            mutation_file = self.resource_dir / "src/resources/mappings/mutation_mappings.csv"
            with open(mutation_file, 'r', encoding='utf-8') as f:
                # 跳过标题行
                next(f)
                self.mutations = [line.split(',')[0] for line in f]
            logger.info(f"加载了 {len(self.mutations)} 个突变因子")
            
            # 从指挥官图片目录获取指挥官名称
            commander_dir = self.resource_dir / "src/resources/images/commanders"
            self.commanders = [f.stem for f in commander_dir.glob("*.png")]
            logger.info(f"加载了 {len(self.commanders)} 个指挥官")
            
        except Exception as e:
            logger.error(f"加载资源文件失败: {str(e)}")
            raise
    
    def build_vocab(self, 
                   name: str, 
                   tokens: List[str], 
                   special_tokens: Optional[List[str]] = None) -> Dict:
        """构建词表.
        
        Args:
            name: 词表名称
            tokens: token列表
            special_tokens: 特殊token列表
            
        Returns:
            词表字典
        """
        special_tokens = special_tokens or ["[PAD]", "[UNK]"]
        token2idx = {}
        
        # 添加特殊token
        for token in special_tokens:
            if token not in token2idx:
                token2idx[token] = len(token2idx)
        
        # 添加普通token
        for token in tokens:
            if token and token not in token2idx:  # 过滤空值
                token2idx[token] = len(token2idx)
        
        return {
            "token2idx": token2idx,
            "special_tokens": special_tokens
        }
    
    def save_vocab(self, vocab: Dict, output_dir: str, name: str):
        """保存词表到文件.
        
        Args:
            vocab: 词表字典
            output_dir: 输出目录
            name: 词表名称
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{name}_vocab.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
        logger.info(f"词表已保存到: {output_path}")
    
    def build_all_vocabs(self, output_dir: str):
        """构建并保存所有词表.
        
        Args:
            output_dir: 输出目录
        """
        # 构建地图词表
        map_vocab = self.build_vocab("map", self.maps)
        self.save_vocab(map_vocab, output_dir, "map")
        
        # 构建指挥官词表
        commander_vocab = self.build_vocab("commander", self.commanders)
        self.save_vocab(commander_vocab, output_dir, "commander")
        
        # 构建突变因子词表
        mutation_vocab = self.build_vocab("mutation", self.mutations)
        self.save_vocab(mutation_vocab, output_dir, "mutation")
        
        # 构建AI词表（使用标准的AI类型）
        ai_types = [
            # 第一行
            "大师机械",
            "步战机甲",
            "袭扰炮击",
            "卡莱的希望",
            "风暴迫临",
            "暗影袭扰",
            "艾尔先锋",
            "族长之军",
            # 第二行
            "旧世步兵团",
            "突击团",
            "旧世机械团",
            "战争机械团",
            "暗影科技团",
            "帝国战斗群",
            # 第三行
            "肆虐扩散",
            "滋生腐化",
            "爆炸威胁",
            "侵略虫群",
            "遮天蔽日"
        ]
        ai_vocab = self.build_vocab("ai", ai_types)
        self.save_vocab(ai_vocab, output_dir, "ai")
        
        logger.info("所有词表构建完成")

def main():
    """主函数."""
    # 构建词表
    builder = VocabBuilder()
    builder.build_all_vocabs("data/processed/vocabs")

if __name__ == "__main__":
    main() 