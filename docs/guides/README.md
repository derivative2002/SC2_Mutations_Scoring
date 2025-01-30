# SC2 Mutations 使用指南

## 快速开始

### 安装

1. 克隆项目:
```bash
git clone https://github.com/your-username/sc2-mutations.git
cd sc2-mutations
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 安装开发依赖(可选):
```bash
pip install -r requirements-dev.txt
```

### 使用随机生成器

```python
from randomizer.src.rules.generator import MutationGenerator
from randomizer.src.models.scorer import get_scorer

# 初始化评分器和生成器
scorer = get_scorer()
generator = MutationGenerator('solo', scorer)

# 生成突变组合
result = generator.generate(
    target_difficulty=3.0,
    map_name="虚空降临",
    commanders=["雷诺"]
)

print(f"生成的突变: {result.mutations}")
print(f"预测难度: {result.difficulty}")
```

### 使用评分模型

```python
from scoring.src.models.scorer import MutationScorer

# 初始化评分器
scorer = MutationScorer()

# 预测难度
score = scorer.predict(
    map_name="虚空降临",
    commanders=["雷诺"],
    mutations=["丧尸大战", "行尸走肉"]
)

print(f"预测难度分数: {score}")
```

## 配置说明

### 随机生成器配置

配置文件位于 `configs/generator.yaml`:

```yaml
mode:
  solo:
    max_mutations: 4
    min_mutations: 2
  duo:
    max_mutations: 8
    min_mutations: 4
```

### 评分模型配置

配置文件位于 `configs/model.yaml`:

```yaml
model:
  vocab_dir: "resources/model/vocab"
  weights_path: "resources/model/weights/model.pt"
  network:
    embed_dim: 64
    hidden_dim: 128
    num_layers: 2
    dropout: 0.1
```

## 常见问题

1. 如何添加新的突变规则?
2. 如何调整难度评分标准?
3. 如何扩展支持的地图和指挥官?

详细解答请参考[技术文档](../technical/README.md)。 