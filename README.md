# SC2 Mutations Scoring

星际争霸2合作模式突变难度评分系统。基于深度学习模型,通过分析地图、指挥官组合、突变因子和敌方AI等特征,预测任务难度等级(1-5分)。

## 项目结构

```
SC2_Mutations_Scoring/          # 项目根目录
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包
├── configs/                   # 配置文件目录
│   ├── __init__.py
│   ├── model_config.py       # 模型配置
│   └── train_config.py       # 训练配置
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   │   └── sc2_mutations_raw.csv
│   └── processed/            # 处理后数据
│       ├── sc2_mutations_duo.csv
│       └── metadata.json
├── src/                      # 源代码目录
│   ├── __init__.py
│   ├── data/                # 数据处理相关
│   │   ├── __init__.py
│   │   ├── dataset.py      # 数据集类
│   │   └── preprocess.py   # 数据预处理
│   ├── models/             # 模型相关
│   │   ├── __init__.py
│   │   ├── embeddings.py   # Embedding层
│   │   └── networks.py     # 网络结构
│   ├── training/           # 训练相关
│   │   ├── __init__.py
│   │   ├── trainer.py      # 训练器
│   │   └── metrics.py      # 评估指标
│   └── utils/              # 工具函数
│       ├── __init__.py
│       └── common.py       # 通用函数
├── scripts/                 # 脚本目录
│   ├── train.py            # 训练脚本
│   └── predict.py          # 预测脚本
└── tests/                  # 测试目录
    ├── __init__.py
    └── test_models.py      # 模型测试
```

## 编码规范

本项目遵循 Google Python Style Guide 和 PEP 8 规范。

### 命名规范

1. 文件名
- 使用小写字母
- 单词间用下划线连接
- 例如: `model_config.py`, `data_loader.py`

2. 类名
- 使用驼峰命名法(CamelCase)
- 每个单词首字母大写
- 例如: `MutationScorer`, `DataLoader`

3. 函数名
- 使用小写字母
- 单词间用下划线连接
- 例如: `train_model()`, `process_data()`

4. 变量名
- 使用小写字母
- 单词间用下划线连接
- 例如: `batch_size`, `learning_rate`

5. 常量名
- 使用大写字母
- 单词间用下划线连接
- 例如: `MAX_EPOCHS`, `DEFAULT_BATCH_SIZE`

### 代码格式

1. 缩进
- 使用4个空格进行缩进
- 不使用制表符(Tab)

2. 行长度
- 最大行长度为80个字符
- 超过时使用括号进行换行

3. 导入顺序
- 标准库导入
- 相关第三方导入
- 本地应用/库特定导入

4. 空行
- 顶级函数和类定义用两个空行分隔
- 类中的方法定义用一个空行分隔

5. 注释
- 使用docstring记录模块、函数、类的文档
- 注释应该是完整的句子
- 行内注释在代码后使用两个空格分隔

### 类型注解

- 使用Python类型注解
- 复杂类型使用typing模块
- 例如:
```python
from typing import List, Dict, Optional

def process_data(data: List[Dict]) -> Optional[np.ndarray]:
    pass
```

## 依赖管理

项目使用requirements.txt管理依赖包。主要依赖包括:

- pytorch>=1.9.0
- numpy>=1.19.2
- pandas>=1.3.0
- scikit-learn>=0.24.2

## 开发流程

1. 创建新分支进行开发
2. 编写单元测试
3. 运行测试确保通过
4. 提交代码前进行代码格式检查
5. 创建Pull Request进行代码审查

## 模型架构

详见 `docs/model_architecture.md`

## 训练流程

详见 `docs/training_pipeline.md`

## License

MIT License 