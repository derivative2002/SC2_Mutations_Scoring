# SC2 Mutations Scoring

星际争霸2合作模式突变难度评估系统。基于深度学习模型,通过分析地图、指挥官组合、突变因子和敌方AI等特征,预测任务难度等级(1-5分)。

## 项目结构

```
.
├── configs/            # 配置文件
├── data/              # 数据目录
│   ├── processed/     # 处理后的数据
│   └── raw/          # 原始数据
├── experiments/       # 实验结果
├── scripts/          # 脚本文件
└── src/              # 源代码
    ├── data/         # 数据处理
    ├── models/       # 模型定义
    └── training/     # 训练相关
```

## 环境配置

1. 创建虚拟环境：
```bash
conda create -n sc2_env python=3.9
conda activate sc2_env
```

2. 安装依赖：
```bash
pip install -e .
```

## 使用说明

1. 数据预处理：
```bash
python src/data/preprocess.py
```

2. 训练模型：
```bash
python scripts/train.py configs/focal_loss.yaml
```

3. 预测：
```bash
python scripts/predict.py --checkpoint experiments/focal_loss_v2/best_acc_checkpoint.pt
```

## 模型架构

- 使用MLP（多层感知器）作为基础模型
- 支持地图、指挥官、突变因子和AI类型的特征嵌入
- 使用Focal Loss处理类别不平衡问题

## 许可证

MIT License 