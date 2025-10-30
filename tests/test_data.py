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
    test_data = pd.DataFrame({
        'LIMIT_BAL': [10000, 20000, 30000],
        'AGE': [25, 35, 45],
        'PAY_0': [0, 1, -1],
        'PAY_2': [0, 1, -1],
        'BILL_AMT1': [1000, 2000, 3000],
        'BILL_AMT2': [1500, 2500, 3500],
        'PAY_AMT1': [500, 1000, 1500],
        'PAY_AMT2': [600, 1100, 1600],
        'default_payment_next_month': [0, 1, 0]
    })
    
    featured_data = create_features(test_data)
    
    # Проверка создания новых фичей
    assert 'PAY_MEAN' in featured_data.columns
    assert 'BILL_AMT_MEAN' in featured_data.columns
    assert 'AGE_BINNED' in featured_data.columns
    assert 'PAYMENT_RATIO' in featured_data.columns
    
    # Проверка корректности вычислений
    assert featured_data['PAY_MEAN'].iloc[0] == 0.0  # (0+0)/2
    assert featured_data['AGE_BINNED'].iloc[0] == 0  # 25 -> bin 0

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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])