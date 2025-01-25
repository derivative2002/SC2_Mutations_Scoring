from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .routers import predictions

app = FastAPI(
    title="SC2 Mutations AI Platform",
    description="星际争霸2合作模式突变难度评估与题目生成平台",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该设置具体的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含路由
app.include_router(predictions.router)

# 健康检查
@app.get("/")
async def root():
    return {"status": "ok", "message": "SC2 Mutations AI Platform is running"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 
