"""基本功能测试。"""
import pytest
from fastapi.testclient import TestClient

from randomizer.src.backend.config import Config
from randomizer.src.backend.app import create_app


def test_config_loading():
    """测试配置加载。"""
    config = Config()
    
    assert config.settings.app.name == "SC2 Mutations Randomizer"
    assert config.settings.app.version == "0.1.0"
    assert config.settings.app.debug is False
    
    assert len(config.get_maps()) > 0
    assert len(config.get_commanders()) > 0
    assert len(config.get_mutations()) > 0
    assert len(config.get_incompatible_pairs()) > 0
    assert len(config.get_required_pairs()) > 0


def test_api_health():
    """测试API健康状况。"""
    app = create_app()
    client = TestClient(app)
    
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_error_handling():
    """测试错误处理。"""
    app = create_app()
    client = TestClient(app)
    
    # 测试404错误
    response = client.get("/not_found")
    assert response.status_code == 404
    
    # 测试无效参数
    response = client.post("/api/mutations/generate", json={
        "target_difficulty": 0  # 无效的难度值
    })
    assert response.status_code == 422 