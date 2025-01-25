# StarCraft II Co-op Maps Resources

这个项目用于收集和管理星际争霸2合作任务相关的资源。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── scrapers/          # 爬虫脚本
│   │   ├── commander_scraper.py    # 指挥官图标爬虫
│   │   ├── mutation_scraper.py     # 突变因子图标爬虫
│   │   └── map_scraper.py         # 地图预览图爬虫
│   └── resources/         # 资源文件
│       └── images/        # 图片资源
│           ├── commanders/    # 指挥官图标
│           ├── mutations/     # 突变因子图标
│           └── maps/         # 地图预览图
├── tests/                 # 测试目录
├── requirements.txt       # 项目依赖
└── README.md             # 项目说明
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 爬取指挥官图标
```bash
python src/scrapers/commander_scraper.py
```

### 爬取突变因子图标
```bash
python src/scrapers/mutation_scraper.py
```

### 爬取地图预览图
```bash
python src/scrapers/map_scraper.py
```

## 注意事项

- 需要安装 Chrome 浏览器
- 需要安装对应版本的 ChromeDriver
- 图片会保存在 src/resources/images 目录下对应的子目录中 