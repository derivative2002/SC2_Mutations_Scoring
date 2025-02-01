# 部署指南

本指南详细说明了如何部署和维护SC2突变评分系统。

## 目录

1. [环境要求](requirements.md)
2. [安装步骤](installation.md)
3. [配置说明](configuration.md)
4. [运维指南](maintenance.md)
5. [监控方案](monitoring.md)

## 系统要求

### 硬件要求
- CPU: 4核+
- 内存: 8GB+
- 存储: 50GB+
- 网络: 100Mbps+

### 软件要求
- 操作系统: Ubuntu 20.04+ / CentOS 8+
- Python 3.8+
- NVIDIA GPU (推荐)
- Docker 20.10+
- Docker Compose 2.0+

## 部署步骤

### 1. 准备环境

```bash
# 安装系统依赖
apt update && apt upgrade -y
apt install -y python3-pip python3-venv nginx

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 配置服务

1. **配置文件**
```bash
# 复制配置模板
cp config/settings.example.json config/settings.json

# 修改配置
vim config/settings.json
```

2. **环境变量**
```bash
# 创建环境变量文件
cp .env.example .env

# 设置必要的环境变量
vim .env
```

### 3. 启动服务

#### 使用Docker（推荐）

```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d
```

#### 直接运行

```bash
# 启动后端服务
python -m randomizer.src.backend.main

# 启动前端服务
cd randomizer/src/frontend
npm install
npm run build
```

## 配置说明

### 1. 应用配置

主要配置文件：`config/settings.json`

```json
{
  "app": {
    "name": "SC2突变评分",
    "version": "1.0.0",
    "debug": false
  },
  "model": {
    "network": {
      "map_dim": 64,
      "commander_dim": 96,
      "mutation_dim": 96,
      "ai_dim": 64,
      "hidden_dims": [256, 128, 64],
      "num_classes": 5
    }
  }
}
```

### 2. 服务配置

Nginx配置示例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 维护指南

### 1. 日常维护

- 日志检查
- 性能监控
- 数据备份
- 系统更新

### 2. 故障处理

常见问题及解决方案：

1. 服务无响应
```bash
# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

2. 内存占用过高
```bash
# 检查内存使用
docker stats
```

### 3. 更新部署

```bash
# 拉取最新代码
git pull

# 重新构建
docker-compose build

# 重启服务
docker-compose down
docker-compose up -d
```

## 监控方案

### 1. 系统监控
- CPU使用率
- 内存使用
- 磁盘空间
- 网络流量

### 2. 应用监控
- 请求响应时间
- 错误率
- API调用量
- 模型预测性能

### 3. 告警设置
- 服务器资源告警
- 应用性能告警
- 错误日志告警

## 备份策略

1. **数据备份**
```bash
# 备份配置
cp -r config/settings.json backups/

# 备份模型文件
cp -r resources/model/ backups/
```

2. **定时备份**
```bash
# 添加定时任务
crontab -e

# 每天凌晨3点备份
0 3 * * * /path/to/backup.sh
```

## 安全建议

1. 使用HTTPS
2. 配置防火墙
3. 定期更新依赖
4. 实施访问控制
5. 监控异常访问 