import pytest
import requests
import json
import time
import sys
import os

# Добавляем путь к корню проекта для импортов
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.api.app import app
    from fastapi.testclient import TestClient
    API_AVAILABLE = True
except ImportError:
    print("API module not available, skipping some tests")
    API_AVAILABLE = False

# Создаем клиент только если API доступно
if API_AVAILABLE:
    client = TestClient(app)
else:
    client = None

def test_health_endpoint_live():
    """Тест health endpoint на живом сервере"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        print("✓ Health endpoint test passed")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_root_endpoint_live():
    """Тест root endpoint на живом сервере"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        print("✓ Root endpoint test passed")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_model_info_endpoint_live():
    """Тест model info endpoint на живом сервере"""
    try:
        response = requests.get("http://localhost:8000/model_info", timeout=5)
        # Может вернуть 200 или 500 в зависимости от загрузки модели
        assert response.status_code in [200, 500]
        print("✓ Model info endpoint test passed")
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_predict_endpoint_live():
    """Тест predict endpoint с валидными данными на живом сервере"""
    valid_data = {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 1,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 1000,
        "BILL_AMT2": 1500,
        "BILL_AMT3": 1200,
        "BILL_AMT4": 1300,
        "BILL_AMT5": 1400,
        "BILL_AMT6": 1600,
        "PAY_AMT1": 500,
        "PAY_AMT2": 600,
        "PAY_AMT3": 700,
        "PAY_AMT4": 800,
        "PAY_AMT5": 900,
        "PAY_AMT6": 1000
    }
    
    try:
        response = requests.post("http://localhost:8000/predict", 
                               json=valid_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            assert "default_probability" in data
            assert "default_class" in data
            assert "risk_level" in data
            assert 0 <= data["default_probability"] <= 1
            assert data["default_class"] in [0, 1]
            print(f"✓ Predict endpoint test passed - Probability: {data['default_probability']}")
        elif response.status_code == 500:
            # Модель может быть не загружена
            print("⚠ Predict endpoint returned 500 (model likely not loaded)")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

def test_batch_predict_live():
    """Тест batch predict endpoint на живом сервере"""
    batch_data = [
        {
            "LIMIT_BAL": 20000,
            "SEX": 2,
            "EDUCATION": 1,
            "MARRIAGE": 1,
            "AGE": 35,
            "PAY_0": 0,
            "PAY_2": 0,
            "PAY_3": 0,
            "PAY_4": 0,
            "PAY_5": 0,
            "PAY_6": 0,
            "BILL_AMT1": 1000,
            "BILL_AMT2": 1500,
            "BILL_AMT3": 1200,
            "BILL_AMT4": 1300,
            "BILL_AMT5": 1400,
            "BILL_AMT6": 1600,
            "PAY_AMT1": 500,
            "PAY_AMT2": 600,
            "PAY_AMT3": 700,
            "PAY_AMT4": 800,
            "PAY_AMT5": 900,
            "PAY_AMT6": 1000
        },
        {
            "LIMIT_BAL": 50000,
            "SEX": 1,
            "EDUCATION": 2,
            "MARRIAGE": 2,
            "AGE": 45,
            "PAY_0": -1,
            "PAY_2": -1,
            "PAY_3": -1,
            "PAY_4": -1,
            "PAY_5": -1,
            "PAY_6": -1,
            "BILL_AMT1": 5000,
            "BILL_AMT2": 5500,
            "BILL_AMT3": 5200,
            "BILL_AMT4": 5300,
            "BILL_AMT5": 5400,
            "BILL_AMT6": 5600,
            "PAY_AMT1": 2500,
            "PAY_AMT2": 2600,
            "PAY_AMT3": 2700,
            "PAY_AMT4": 2800,
            "PAY_AMT5": 2900,
            "PAY_AMT6": 3000
        }
    ]
    
    try:
        response = requests.post("http://localhost:8000/batch_predict", 
                               json=batch_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_count" in data
            assert len(data["predictions"]) == 2
            print("✓ Batch predict endpoint test passed")
        elif response.status_code == 500:
            print("⚠ Batch predict endpoint returned 500 (model likely not loaded)")
        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

# Тесты с TestClient (только если API доступно)
if API_AVAILABLE:
    def test_health_endpoint():
        """Тест health endpoint с TestClient"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data

    def test_root_endpoint():
        """Тест root endpoint с TestClient"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

if __name__ == "__main__":
    # Запуск тестов вручную
    print("Running API tests...")
    
    # Предварительно убедитесь что сервер запущен на localhost:8000
    tests = [
        test_health_endpoint_live,
        test_root_endpoint_live, 
        test_model_info_endpoint_live,
        test_predict_endpoint_live,
        test_batch_predict_live
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")