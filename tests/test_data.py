import pytest
import pandas as pd
import numpy as np
import sys
import os

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocess import clean_data, create_features

def test_clean_data():
    """Тест очистки данных"""
    # Создание тестовых данных
    test_data = pd.DataFrame({
        'ID': [1, 2, 3],
        'LIMIT_BAL': [10000, 20000, 30000],
        'SEX': [1, 2, 1],
        'default.payment.next.month': [0, 1, 0]
    })
    
    cleaned_data = clean_data(test_data)
    
    # Проверка удаления ID колонки
    assert 'ID' not in cleaned_data.columns
    # Проверка переименования целевой переменной
    assert 'default_payment_next_month' in cleaned_data.columns

def test_create_features():
    """Тест создания фичей"""
    # Создаем полный набор данных как в реальном датасете
    test_data = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000, 30000],
        'SEX': [1, 2, 1],
        'EDUCATION': [1, 2, 3],
        'MARRIAGE': [1, 2, 1],
        'AGE': [25, 35, 45],
        'PAY_0': [0, 1, -1],
        'PAY_2': [0, 1, -1],
        'PAY_3': [0, 1, -1],
        'PAY_4': [0, 1, -1],
        'PAY_5': [0, 1, -1],
        'PAY_6': [0, 1, -1],
        'BILL_AMT1': [1000, 2000, 3000],
        'BILL_AMT2': [1500, 2500, 3500],
        'BILL_AMT3': [1200, 2200, 3200],
        'BILL_AMT4': [1300, 2300, 3300],
        'BILL_AMT5': [1400, 2400, 3400],
        'BILL_AMT6': [1600, 2600, 3600],
        'PAY_AMT1': [500, 1000, 1500],
        'PAY_AMT2': [600, 1100, 1600],
        'PAY_AMT3': [700, 1200, 1700],
        'PAY_AMT4': [800, 1300, 1800],
        'PAY_AMT5': [900, 1400, 1900],
        'PAY_AMT6': [1000, 1500, 2000],
        'default_payment_next_month': [0, 1, 0]
    })
    
    featured_data = create_features(test_data)
    
    # Проверка создания новых фичей
    assert 'PAY_MEAN' in featured_data.columns
    assert 'BILL_AMT_MEAN' in featured_data.columns
    assert 'AGE_BINNED' in featured_data.columns
    assert 'PAYMENT_RATIO' in featured_data.columns
    
    # Проверка корректности вычислений - PAY_MEAN для первой строки
    # PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 = [0, 0, 0, 0, 0, 0] -> mean = 0.0
    pay_mean_value = featured_data['PAY_MEAN'].iloc[0]
    assert abs(pay_mean_value - 0.0) < 0.001, f"Expected PAY_MEAN ≈ 0.0, got {pay_mean_value}"
    
    # AGE_BINNED для 25 лет -> bin 0
    assert featured_data['AGE_BINNED'].iloc[0] == 0
    
    # Проверка что нет NaN значений
    assert not featured_data.isnull().any().any()

def test_create_features_with_missing_columns():
    """Тест создания фичей с отсутствующими колонками"""
    # Минимальный набор данных
    test_data = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000],
        'AGE': [25, 35],
        'default_payment_next_month': [0, 1]
    })
    
    # Функция должна обработать данные даже с отсутствующими колонками
    featured_data = create_features(test_data)
    
    # Проверка что функция не падает и возвращает DataFrame
    assert isinstance(featured_data, pd.DataFrame)
    assert len(featured_data) == 2

def test_data_validation():
    """Тест валидации данных"""
    from data.validation import validate_dataset
    
    # Создание корректных тестовых данных
    valid_data = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000, 30000],
        'SEX': [1, 2, 1],
        'EDUCATION': [1, 2, 3],
        'MARRIAGE': [1, 2, 1],
        'AGE': [25, 35, 45],
        'PAY_0': [0, 1, -1],
        'PAY_2': [0, 1, -1],
        'PAY_3': [0, 1, -1],
        'PAY_4': [0, 1, -1],
        'PAY_5': [0, 1, -1],
        'PAY_6': [0, 1, -1],
        'BILL_AMT1': [1000, 2000, 3000],
        'BILL_AMT2': [1500, 2500, 3500],
        'BILL_AMT3': [1200, 2200, 3200],
        'BILL_AMT4': [1300, 2300, 3300],
        'BILL_AMT5': [1400, 2400, 3400],
        'BILL_AMT6': [1600, 2600, 3600],
        'PAY_AMT1': [500, 1000, 1500],
        'PAY_AMT2': [600, 1100, 1600],
        'PAY_AMT3': [700, 1200, 1700],
        'PAY_AMT4': [800, 1300, 1800],
        'PAY_AMT5': [900, 1400, 1900],
        'PAY_AMT6': [1000, 1500, 2000],
        'default_payment_next_month': [0, 1, 0]
    })
    
    # Добавление фичей
    valid_data = create_features(valid_data)
    
    # Валидация должна пройти успешно
    report = validate_dataset(valid_data, "test")
    assert report['success'] == True

def test_feature_engineering_consistency():
    """Тест консистентности feature engineering"""
    # Два одинаковых набора данных
    data1 = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000],
        'AGE': [25, 35],
        'PAY_0': [0, 1],
        'PAY_2': [0, 1],
        'PAY_3': [0, 1],
        'PAY_4': [0, 1],
        'PAY_5': [0, 1],
        'PAY_6': [0, 1],
        'BILL_AMT1': [1000, 2000],
        'BILL_AMT2': [1500, 2500],
        'BILL_AMT3': [1200, 2200],
        'BILL_AMT4': [1300, 2300],
        'BILL_AMT5': [1400, 2400],
        'BILL_AMT6': [1600, 2600],
        'PAY_AMT1': [500, 1000],
        'PAY_AMT2': [600, 1100],
        'PAY_AMT3': [700, 1200],
        'PAY_AMT4': [800, 1300],
        'PAY_AMT5': [900, 1400],
        'PAY_AMT6': [1000, 1500],
        'default_payment_next_month': [0, 1]
    })
    
    data2 = data1.copy()
    
    # Применение feature engineering
    data1_featured = create_features(data1)
    data2_featured = create_features(data2)
    
    # Проверка консистентности
    for col in data1_featured.columns:
        if col in data2_featured.columns:
            assert data1_featured[col].equals(data2_featured[col]), f"Inconsistency in {col}"

def debug_create_features():
    """Функция для отладки - посмотрим что реально вычисляется"""
    test_data = pd.DataFrame({
        'LIMIT_BAL': [10000],
        'SEX': [1],
        'EDUCATION': [1],
        'MARRIAGE': [1],
        'AGE': [25],
        'PAY_0': [0],
        'PAY_2': [0],
        'PAY_3': [0],
        'PAY_4': [0],
        'PAY_5': [0],
        'PAY_6': [0],
        'BILL_AMT1': [1000],
        'BILL_AMT2': [1500],
        'BILL_AMT3': [1200],
        'BILL_AMT4': [1300],
        'BILL_AMT5': [1400],
        'BILL_AMT6': [1600],
        'PAY_AMT1': [500],
        'PAY_AMT2': [600],
        'PAY_AMT3': [700],
        'PAY_AMT4': [800],
        'PAY_AMT5': [900],
        'PAY_AMT6': [1000],
        'default_payment_next_month': [0]
    })
    
    featured_data = create_features(test_data)
    print("Debug - PAY columns found:", [col for col in test_data.columns if col.startswith('PAY_')])
    print("Debug - PAY_MEAN value:", featured_data['PAY_MEAN'].iloc[0])
    print("Debug - All PAY values:", [test_data[col].iloc[0] for col in ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']])

if __name__ == "__main__":
    # Сначала запустим отладку
    debug_create_features()
    
    # Затем тесты
    pytest.main([__file__, "-v"])