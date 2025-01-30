# SC2 Mutations 部署指南

## 环境准备

### 系统要求

- CPU: 2核心以上
- 内存: 4GB以上
- 磁盘: 10GB以上
- 操作系统: Linux/macOS/Windows

### Python环境

1. 安装Python 3.9+:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.9 python3.9-dev python3.9-venv

# macOS
brew install python@3.9

# Windows
# 从Python官网下载安装包
```

2. 创建虚拟环境:
```bash
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 依赖安装

1. 基础依赖:
```bash
pip install -r requirements.txt
```

2. 开发依赖(可选):
```bash
pip install -r requirements-dev.txt
```

## 配置文件

### 1. 应用配置

创建 `configs/app.yaml`:
```yaml
app:
  name: "SC2 Mutations Randomizer"
  version: "0.1.0"
  description: "星际争霸2合作任务突变组合生成器"
  debug: false
```

### 2. 模型配置

创建 `configs/model.yaml`:
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

### 3. 生成器配置

创建 `configs/generator.yaml`:
```yaml
mode:
  solo:
    max_mutations: 4
    min_mutations: 2
  duo:
    max_mutations: 8
    min_mutations: 4
```

## 启动服务

### 1. 开发环境

```bash
uvicorn randomizer.src.backend.main:app --reload --port 8000
```

### 2. 生产环境

1. 使用Gunicorn(Linux/macOS):
```bash
gunicorn randomizer.src.backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

2. 使用Supervisor管理进程:

创建 `/etc/supervisor/conf.d/sc2mutations.conf`:
```ini
[program:sc2mutations]
command=/path/to/venv/bin/gunicorn randomizer.src.backend.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
directory=/path/to/project
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/sc2mutations/error.log
stdout_logfile=/var/log/sc2mutations/access.log
```

启动服务:
```bash
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start sc2mutations
```

## 监控和日志

### 1. 日志配置

日志文件位于 `logs/` 目录:
- `app.log`: 应用日志
- `error.log`: 错误日志
- `access.log`: 访问日志

### 2. 监控指标

1. 应用指标:
- API请求数
- 响应时间
- 错误率
- 缓存命中率

2. 系统指标:
- CPU使用率
- 内存使用率
- 磁盘使用率
- 网络流量

### 3. 告警设置

1. 错误率告警:
- 错误率 > 1%: 警告
- 错误率 > 5%: 严重

2. 响应时间告警:
- P95 > 500ms: 警告
- P95 > 1000ms: 严重

3. 系统资源告警:
- CPU > 80%: 警告
- 内存 > 80%: 警告
- 磁盘 > 80%: 警告

## 备份和恢复

### 1. 数据备份

1. 配置文件:
```bash
cp -r configs/ backup/configs/
```

2. 模型文件:
```bash
cp -r resources/model/ backup/model/
```

3. 日志文件:
```bash
cp -r logs/ backup/logs/
```

### 2. 数据恢复

1. 恢复配置:
```bash
cp -r backup/configs/ configs/
```

2. 恢复模型:
```bash
cp -r backup/model/ resources/model/
```

## 故障排除

### 1. 常见问题

1. 服务无法启动:
- 检查端口占用
- 检查配置文件
- 检查日志文件

2. 模型加载失败:
- 检查模型文件路径
- 检查模型版本兼容性
- 检查GPU支持

3. API响应慢:
- 检查数据库连接
- 检查缓存状态
- 检查系统资源

### 2. 日志分析

使用以下命令分析日志:
```bash
# 查看错误日志
tail -f logs/error.log

# 统计错误类型
grep "ERROR" logs/error.log | cut -d" " -f5- | sort | uniq -c | sort -nr

# 分析响应时间
grep "Request completed" logs/access.log | awk '{print $NF}' | sort -n | awk '{sum+=$1} END {print "avg:", sum/NR}'
```

### 3. 性能优化

1. 应用层面:
- 优化缓存策略
- 调整批处理大小
- 使用异步处理

2. 系统层面:
- 调整系统参数
- 优化网络配置
- 升级硬件资源 