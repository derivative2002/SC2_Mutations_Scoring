# 开发者指南

本指南面向希望了解系统技术细节或参与开发的开发者。

## 目录

1. [系统架构](architecture.md)
2. [开发环境搭建](setup.md)
3. [代码结构](codebase.md)
4. [API设计](api_design.md)
5. [模型说明](models.md)
6. [测试指南](testing.md)

## 技术栈

### 后端
- Python 3.8+
- FastAPI
- PyTorch 1.8+
- SQLAlchemy
- Pydantic

### 前端
- Vue.js 3.x
- Element Plus
- Axios
- Vite

### 数据库
- SQLite（开发）
- PostgreSQL（生产）

## 项目结构

```
sc2_mutations/
├── scoring/                  # 评分模型模块
│   ├── src/
│   │   ├── data/            # 数据处理
│   │   ├── models/          # 模型定义
│   │   └── training/        # 训练相关
│   └── tests/               # 单元测试
├── randomizer/              # 突变生成器模块
│   ├── src/
│   │   ├── api/            # API接口
│   │   ├── models/         # 数据模型
│   │   └── rules/          # 规则引擎
│   └── tests/              # 单元测试
└── docs/                    # 项目文档
```

## 开发流程

1. **环境配置**
   - 克隆代码库
   - 安装依赖
   - 配置开发环境

2. **开发规范**
   - 代码风格：PEP 8
   - 提交规范：Angular风格
   - 分支策略：Git Flow

3. **测试要求**
   - 单元测试覆盖率 > 80%
   - 集成测试
   - 性能测试

## API文档

详细的API文档请参考：
- [API参考](../api_reference/README.md)
- [接口规范](api_design.md)
- [认证授权](authentication.md)

## 模型开发

1. **数据处理**
   - 数据收集
   - 预处理
   - 特征工程

2. **模型训练**
   - 模型架构
   - 训练流程
   - 评估方法

3. **部署和优化**
   - 模型导出
   - 性能优化
   - 监控和更新

## 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交变更
4. 发起Pull Request

## 常见问题

开发中常见的问题和解决方案请参考：
[常见问题](troubleshooting.md) 