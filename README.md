# SC2 突变评级预测模型

这个项目使用图神经网络（GNN）和注意力机制来预测星际争霸2合作模式中突变任务的难度评级。

## 项目结构

```
SC2_Mutations_Scoring/
├── data/                      # 数据目录
│   └── raw/                   # 原始数据
│       └── train.csv          # 训练数据
├── src/                       # 源代码
│   ├── config/                # 配置文件
│   │   └── default_config.py  # 默认配置
│   ├── data/                  # 数据处理
│   │   └── preprocess.py      # 数据预处理
│   ├── models/                # 模型定义
│   │   └── gnn_attention.py   # GNN和注意力模型
│   └── utils/                 # 工具函数
│       ├── logger.py          # 日志工具
│       ├── progress.py        # 进度条工具
│       └── visualization.py   # 可视化工具
├── model/                     # 模型保存目录
│   ├── checkpoints/          # 检查点
│   └── best_model.pth        # 最佳模型
├── logs/                      # 日志目录
├── visualizations/            # 可视化输出目录
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 环境配置

1. 创建并激活conda环境：
```bash
conda create -n sc2_env python=3.9
conda activate sc2_env
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据说明

训练数据包含以下字段：
- 序号：任务编号
- 突变名称：任务名称
- 地图：任务地图
- 因子1-4：突变因子
- 评级：难度评级（1-10）

## 模型架构

模型包含以下主要组件：
1. 图神经网络（GNN）：处理突变因子之间的关系
2. 注意力机制：处理地图对突变难度的影响
3. 多层感知机：综合特征进行最终预测

## 运行说明

1. 训练模型：
```bash
python src/main.py
```

2. 评估模型：
```bash
python src/evaluate.py
```

3. 预测新数据：
```bash
python src/predict.py --input_file path/to/input.csv
```

## 可视化

模型训练过程中会生成以下可视化：
1. 训练损失和验证准确率曲线
2. 模型结构图
3. 注意力权重热力图

可视化结果保存在 `visualizations/` 目录下。

## 日志

训练和评估日志保存在 `logs/` 目录下，包含：
1. 训练过程日志
2. 评估结果日志
3. 预测输出日志 