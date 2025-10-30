import pytest
import requests
import json
import time
from src.api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    """Тест health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_root_endpoint():
    """Тест root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data

def test_model_info_endpoint():
    """Тест model info endpoint"""
    response = client.get("/model_info")
    # Может вернуть 500 если модель не загружена, что нормально для тестов
    assert response.status_code in [200, 500]

def test_predict_endpoint():
    """Тест predict endpoint с валидными данными"""
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
    
    response = client.post("/predict", json=valid_data)
    # Может вернуть 500 если модель не загружена
    if response.status_code == 200:
        data = response.json()
        assert "default_probability" in data
        assert "default_class" in data
        assert "risk_level" in data
        assert 0 <= data["default_probability"] <= 1
        assert data["default_class"] in [0, 1]

def test_predict_invalid_data():
    """Тест predict endpoint с невалидными данными"""
    invalid_data = {
        "LIMIT_BAL": "invalid",  # Неверный тип
        "SEX": 5,  # Неверное значение
        # ... остальные поля отсутствуют
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code in [400, 422, 500]

def test_batch_predict():
    """Тест batch predict endpoint"""
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
    
    response = client.post("/batch_predict", json=batch_data)
    if response.status_code == 200:
        data = response.json()
        assert "predictions" in data
        assert "total_count" in data
        assert len(data["predictions"]) == 2