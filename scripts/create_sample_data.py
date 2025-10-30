"""
Скрипт для создания sample данных для тестирования
"""
import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Создание sample данных"""
    np.random.seed(42)
    
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'LIMIT_BAL': np.random.uniform(10000, 500000, n_samples).astype(int),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
        'AGE': np.random.randint(20, 70, n_samples),
        'PAY_0': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_2': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_3': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_4': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_5': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'PAY_6': np.random.choice([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], n_samples),
        'BILL_AMT1': np.random.uniform(0, 500000, n_samples).astype(int),
        'BILL_AMT2': np.random.uniform(0, 500000, n_samples).astype(int),
        'BILL_AMT3': np.random.uniform(0, 500000, n_samples).astype(int),
        'BILL_AMT4': np.random.uniform(0, 500000, n_samples).astype(int),
        'BILL_AMT5': np.random.uniform(0, 500000, n_samples).astype(int),
        'BILL_AMT6': np.random.uniform(0, 500000, n_samples).astype(int),
        'PAY_AMT1': np.random.uniform(0, 50000, n_samples).astype(int),
        'PAY_AMT2': np.random.uniform(0, 50000, n_samples).astype(int),
        'PAY_AMT3': np.random.uniform(0, 50000, n_samples).astype(int),
        'PAY_AMT4': np.random.uniform(0, 50000, n_samples).astype(int),
        'PAY_AMT5': np.random.uniform(0, 50000, n_samples).astype(int),
        'PAY_AMT6': np.random.uniform(0, 50000, n_samples).astype(int),
    })
    
    # Создание целевой переменной с некоторой логикой
    default_proba = 1 / (1 + np.exp(-(
        0.000001 * sample_data['LIMIT_BAL'] +
        0.1 * (sample_data['PAY_0'] > 0) +
        0.1 * (sample_data['PAY_2'] > 0) +
        0.05 * (sample_data['AGE'] > 50) -
        0.2 * (sample_data['EDUCATION'] == 1) +
        np.random.normal(0, 0.5, n_samples)
    )))
    
    sample_data['default_payment_next_month'] = (default_proba > 0.5).astype(int)
    
    return sample_data

def main():
    """Основная функция"""
    print("Creating sample data...")
    
    # Создание директорий
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Создание sample данных
    sample_data = create_sample_data()
    
    # Сохранение
    sample_data.to_csv('data/raw/default_of_credit_card_clients.csv', index=False)
    sample_data.to_csv('data/processed/train.csv', index=False)
    
    print(f"Sample data created with shape: {sample_data.shape}")
    print("Default rate:", sample_data['default_payment_next_month'].mean())

if __name__ == "__main__":
    main()