# SC2 Mutations

星际争霸2合作任务突变组合生成器

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

### 使用示例

1. 生成突变组合:
```bash
python examples/generate_mutations.py
```

2. 预测难度分数:
```bash
python examples/predict_difficulty.py
```

## 文档

- [API文档](docs/api/README.md)
- [使用指南](docs/guides/README.md)
- [技术文档](docs/technical/README.md)
- [部署指南](docs/technical/deployment.md)

## 开发

### 测试

```bash
# 运行所有测试
pytest

# 运行特定模块测试
pytest randomizer/tests/
pytest scoring/tests/

# 生成覆盖率报告
pytest --cov=randomizer/src --cov=scoring/src --cov-report=html
```

### 代码风格

```bash
# 运行代码格式化
black .

# 运行代码检查
flake8
```

## 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

- 项目维护者: [Your Name](mailto:your.email@example.com)
- 项目主页: https://github.com/your-username/sc2-mutations 