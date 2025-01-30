"""API集成测试."""

import pytest
from fastapi.testclient import TestClient

from randomizer.src.backend.app import create_app
from randomizer.src.backend.config import Config
from randomizer.src.backend.models.scorer import MutationScorer


@pytest.fixture
def app():
    """创建测试应用."""
    return create_app()


@pytest.fixture
def client(app):
    """创建测试客户端."""
    return TestClient(app)


def test_get_maps(client):
    """测试获取地图列表."""
    response = client.get("/api/mutations/maps")
    assert response.status_code == 200
    maps = response.json()
    assert isinstance(maps, list)
    assert len(maps) > 0
    assert all(isinstance(m, str) for m in maps)


def test_get_commanders(client):
    """测试获取指挥官列表."""
    response = client.get("/api/mutations/commanders")
    assert response.status_code == 200
    commanders = response.json()
    assert isinstance(commanders, list)
    assert len(commanders) > 0
    assert all(isinstance(c, str) for c in commanders)


def test_get_mutations(client):
    """测试获取突变因子列表."""
    response = client.get("/api/mutations/mutations")
    assert response.status_code == 200
    mutations = response.json()
    assert isinstance(mutations, list)
    assert len(mutations) > 0
    assert all(isinstance(m, str) for m in mutations)


def test_generate_mutations_solo(client):
    """测试生成单人模式突变组合."""
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo"
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data["map_name"], str)
    assert isinstance(data["commanders"], list)
    assert len(data["commanders"]) == 1
    assert isinstance(data["mutations"], list)
    assert 2 <= len(data["mutations"]) <= 4  # solo模式突变数量范围
    assert isinstance(data["difficulty"], float)
    assert 1.0 <= data["difficulty"] <= 5.0
    assert isinstance(data["rules"], list)


def test_generate_mutations_duo(client):
    """测试生成双人模式突变组合."""
    request_data = {
        "target_difficulty": 3.0,
        "mode": "duo"
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data["map_name"], str)
    assert isinstance(data["commanders"], list)
    assert len(data["commanders"]) == 2
    assert isinstance(data["mutations"], list)
    assert 4 <= len(data["mutations"]) <= 8  # duo模式突变数量范围
    assert isinstance(data["difficulty"], float)
    assert 1.0 <= data["difficulty"] <= 5.0
    assert isinstance(data["rules"], list)


def test_generate_mutations_with_map(client):
    """测试指定地图生成突变组合."""
    # 先获取可用地图
    maps_response = client.get("/api/mutations/maps")
    maps = maps_response.json()
    
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "map_name": maps[0]
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["map_name"] == maps[0]


def test_generate_mutations_with_commanders(client):
    """测试指定指挥官生成突变组合."""
    # 先获取可用指挥官
    commanders_response = client.get("/api/mutations/commanders")
    commanders = commanders_response.json()
    
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "commanders": [commanders[0]]
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert data["commanders"] == [commanders[0]]


def test_generate_mutations_invalid_difficulty(client):
    """测试无效难度值."""
    request_data = {
        "target_difficulty": 6.0,  # 超出范围
        "mode": "solo"
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 400
    
    data = response.json()
    assert data["code"] == "GENERATION_ERROR"
    assert "难度" in data["message"]


def test_generate_mutations_invalid_mode(client):
    """测试无效游戏模式."""
    request_data = {
        "target_difficulty": 3.0,
        "mode": "invalid"  # 无效模式
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 422  # FastAPI验证错误
    
    data = response.json()
    assert "mode" in str(data["detail"])


def test_generate_mutations_invalid_map(client):
    """测试无效地图名称."""
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "map_name": "不存在的地图"
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 400
    
    data = response.json()
    assert data["code"] == "GENERATION_ERROR"
    assert "地图" in data["message"]


def test_generate_mutations_invalid_commanders(client):
    """测试无效指挥官组合."""
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "commanders": ["不存在的指挥官"]
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 400
    
    data = response.json()
    assert data["code"] == "GENERATION_ERROR"
    assert "指挥官" in data["message"]


def test_generate_mutations_wrong_commander_count(client):
    """测试错误的指挥官数量."""
    # 先获取可用指挥官
    commanders_response = client.get("/api/mutations/commanders")
    commanders = commanders_response.json()
    
    request_data = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "commanders": commanders[:2]  # solo模式指定2个指挥官
    }
    
    response = client.post("/api/mutations/generate", json=request_data)
    assert response.status_code == 422  # FastAPI验证错误
    
    data = response.json()
    assert "commanders" in str(data["detail"]) 