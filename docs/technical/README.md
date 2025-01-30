# SC2 Mutations 技术文档

## 项目结构

```
sc2_mutations/
├── docs/                      # 项目文档
│   ├── api/                   # API文档
│   ├── guides/                # 使用指南
│   └── technical/            # 技术文档
├── randomizer/               # 突变生成器模块
│   ├── src/
│   │   ├── api/             # API接口
│   │   ├── config/          # 配置管理
│   │   ├── models/          # 模型封装
│   │   └── rules/           # 规则引擎
│   └── tests/               # 生成器测试
├── scoring/                  # 评分模型模块
│   ├── src/
│   │   ├── data/            # 数据处理
│   │   ├── models/          # 模型定义
│   │   └── training/        # 训练相关
│   └── tests/               # 评分模型测试
├── resources/               # 资源文件
│   ├── data/               # 游戏数据
│   └── model/              # 模型文件
├── scripts/                # 工具脚本
├── configs/                # 配置文件
└── examples/               # 示例代码
```

## 核心模块说明

### 随机生成器

#### 配置管理 (`randomizer.src.config`)

负责加载和管理配置信息，包括:
- 地图数据
- 指挥官数据
- 突变因子数据
- 规则数据

#### 规则引擎 (`randomizer.src.rules`)

实现突变组合的生成逻辑:
- 突变因子互斥规则
- 突变因子依赖规则
- 难度平衡算法

#### 模型封装 (`randomizer.src.models`)

封装评分模型的调用接口:
- 模型加载
- 预测接口
- 缓存管理

### 评分模型

#### 数据处理 (`scoring.src.data`)

实现数据预处理流程:
- 数据清洗
- 特征工程
- 词表构建

#### 模型定义 (`scoring.src.models`)

定义神经网络模型结构:
- 嵌入层
- 注意力机制
- 预测层

#### 训练模块 (`scoring.src.training`)

实现模型训练流程:
- 数据加载
- 训练循环
- 验证评估
- 模型保存

## 开发指南

### 添加新的突变规则

1. 在 `resources/data/rules.yaml` 中添加规则定义
2. 更新 `randomizer.src.rules.generator` 中的规则处理逻辑

### 调整难度评分标准

1. 修改 `scoring.src.models.networks` 中的模型结构
2. 在 `scoring.src.training.trainer` 中调整训练参数
3. 重新训练模型

### 扩展支持的地图和指挥官

1. 在 `resources/data/` 下的相应JSON文件中添加数据
2. 更新词表文件
3. 如果需要，重新训练模型

## 测试

### 单元测试

```bash
pytest randomizer/tests/unit/
pytest scoring/tests/unit/
```

### 集成测试

```bash
pytest randomizer/tests/integration/
pytest scoring/tests/integration/
```

### 系统测试

```bash
pytest randomizer/tests/system/
pytest scoring/tests/system/
```

## 部署

### 环境要求

- Python 3.9+
- PyTorch 1.9+
- FastAPI
- Redis (可选，用于缓存)

### 部署步骤

1. 准备环境
2. 配置文件
3. 启动服务
4. 监控和日志

详细部署文档请参考[部署指南](deployment.md)。 