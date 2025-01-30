# SC2 Mutations API 文档

## 概述

本文档描述了SC2 Mutations项目的API接口。

## 模块

### 随机生成器 API

- `randomizer.api.mutations`: 突变组合生成接口
- `randomizer.api.health`: 健康检查接口

### 评分模型 API

- `scoring.models.predict`: 难度评分预测接口
- `scoring.models.train`: 模型训练接口

## 详细说明

### 突变组合生成

#### 生成突变组合

```http
POST /api/mutations/generate
```

请求参数:
- `target_difficulty`: 目标难度 (1.0-5.0)
- `map_name`: 地图名称 (可选)
- `commanders`: 指挥官列表 (可选)
- `mode`: 游戏模式 ('solo'/'duo')

响应:
```json
{
    "mutations": ["突变1", "突变2", ...],
    "difficulty": 3.5,
    "map_name": "虚空降临",
    "commanders": ["雷诺"]
}
```

### 难度评分

#### 预测难度分数

```http
POST /api/score/predict
```

请求参数:
- `map_name`: 地图名称
- `commanders`: 指挥官列表
- `mutations`: 突变列表
- `ai_type`: AI类型 (可选)

响应:
```json
{
    "score": 3.5
}
``` 