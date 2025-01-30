"""API端到端测试."""

import pytest
import requests

# 测试服务器URL
BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def setup():
    """测试前的设置."""
    # 这里可以添加启动服务器的代码
    # 但通常在实际测试中，我们会假设服务器已经运行
    yield
    # 这里可以添加清理代码


def test_api_health():
    """测试API健康状况."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_openapi_docs():
    """测试OpenAPI文档."""
    response = requests.get(f"{BASE_URL}/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_complete_flow():
    """测试完整的生成流程."""
    # 1. 获取可用地图
    maps_response = requests.get(f"{BASE_URL}/api/mutations/maps")
    assert maps_response.status_code == 200
    maps = maps_response.json()
    assert len(maps) > 0
    
    # 2. 获取可用指挥官
    commanders_response = requests.get(f"{BASE_URL}/api/mutations/commanders")
    assert commanders_response.status_code == 200
    commanders = commanders_response.json()
    assert len(commanders) > 0
    
    # 3. 获取可用突变因子
    mutations_response = requests.get(f"{BASE_URL}/api/mutations/mutations")
    assert mutations_response.status_code == 200
    mutations = mutations_response.json()
    assert len(mutations) > 0
    
    # 4. 生成突变组合（solo模式）
    solo_request = {
        "target_difficulty": 3.0,
        "mode": "solo",
        "map_name": maps[0],
        "commanders": [commanders[0]]
    }
    
    solo_response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        json=solo_request
    )
    assert solo_response.status_code == 200
    solo_result = solo_response.json()
    
    assert solo_result["map_name"] == maps[0]
    assert solo_result["commanders"] == [commanders[0]]
    assert 2 <= len(solo_result["mutations"]) <= 4
    
    # 5. 生成突变组合（duo模式）
    duo_request = {
        "target_difficulty": 4.0,
        "mode": "duo",
        "map_name": maps[0],
        "commanders": commanders[:2]
    }
    
    duo_response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        json=duo_request
    )
    assert duo_response.status_code == 200
    duo_result = duo_response.json()
    
    assert duo_result["map_name"] == maps[0]
    assert duo_result["commanders"] == commanders[:2]
    assert 4 <= len(duo_result["mutations"]) <= 8


def test_error_handling():
    """测试错误处理."""
    # 1. 无效的请求体
    invalid_json_response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        data="invalid json"
    )
    assert invalid_json_response.status_code == 422
    
    # 2. 缺少必需字段
    missing_field_response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        json={"mode": "solo"}  # 缺少target_difficulty
    )
    assert missing_field_response.status_code == 422
    
    # 3. 字段验证错误
    validation_error_response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        json={
            "target_difficulty": 0.5,  # 低于最小值
            "mode": "solo"
        }
    )
    assert validation_error_response.status_code == 422


def test_concurrent_requests():
    """测试并发请求."""
    import concurrent.futures
    
    def make_request():
        return requests.post(
            f"{BASE_URL}/api/mutations/generate",
            json={
                "target_difficulty": 3.0,
                "mode": "solo"
            }
        )
    
    # 并发发送10个请求
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in futures]
    
    # 验证所有请求都成功
    assert all(r.status_code == 200 for r in responses)
    
    # 验证返回的结果都是有效的
    results = [r.json() for r in responses]
    assert all(2 <= len(r["mutations"]) <= 4 for r in results)  # solo模式
    assert all(1.0 <= r["difficulty"] <= 5.0 for r in results)


def test_performance():
    """测试性能."""
    import time
    
    # 测试生成接口的响应时间
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/mutations/generate",
        json={
            "target_difficulty": 3.0,
            "mode": "solo"
        }
    )
    end_time = time.time()
    
    assert response.status_code == 200
    assert end_time - start_time < 1.0  # 响应时间应小于1秒 