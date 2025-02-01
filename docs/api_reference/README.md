# API参考文档

本文档详细说明了SC2突变评分系统提供的所有API接口。

## API概述

系统提供RESTful API，支持以下功能：
- 突变组合生成
- 难度评分预测
- 数据查询和统计

## 基础信息

- 基础URL：`/api/mutations`
- 支持格式：JSON
- 认证方式：Bearer Token

## 接口列表

### 突变相关

#### 获取突变列表
```http
GET /api/mutations/mutations
```

**响应示例：**
```json
{
  "mutations": [
    {
      "id": "mutation_1",
      "name": "虚空裂隙",
      "description": "敌人单位可以通过虚空裂隙传送"
    }
  ]
}
```

#### 获取地图列表
```http
GET /api/mutations/maps
```

#### 获取指挥官列表
```http
GET /api/mutations/commanders
```

### 评分相关

#### 预测难度分数
```http
POST /api/mutations/score
```

**请求体：**
```json
{
  "map": "聚铁成兵",
  "commanders": ["斯台特曼", "扎加拉"],
  "mutations": ["风暴英雄", "虚空裂隙", "同化体"],
  "ai": "滋生腐化"
}
```

**响应示例：**
```json
{
  "score": 4,
  "difficulty": "困难",
  "analysis": {
    "factors": [
      {
        "name": "虚空裂隙",
        "impact": 0.8
      }
    ]
  }
}
```

## 错误码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 401 | 未授权 |
| 404 | 资源不存在 |
| 500 | 服务器错误 |

## 使用示例

### Python
```python
import requests

url = "http://localhost:8000/api/mutations/score"
data = {
    "map": "聚铁成兵",
    "commanders": ["斯台特曼", "扎加拉"],
    "mutations": ["风暴英雄", "虚空裂隙", "同化体"],
    "ai": "滋生腐化"
}

response = requests.post(url, json=data)
print(response.json())
```

### JavaScript
```javascript
fetch('/api/mutations/score', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    map: "聚铁成兵",
    commanders: ["斯台特曼", "扎加拉"],
    mutations: ["风暴英雄", "虚空裂隙", "同化体"],
    ai: "滋生腐化"
  }),
})
.then(response => response.json())
.then(data => console.log(data));
```

## 注意事项

1. 所有请求需要包含适当的认证信息
2. 请求体和响应都使用UTF-8编码
3. 日期时间格式遵循ISO 8601标准
4. 建议实现请求重试和错误处理机制

## 更新日志

- 2024-01-30: 初始版本发布
- 2024-02-01: 添加新的分析接口 