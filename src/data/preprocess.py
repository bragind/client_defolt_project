import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

def load_data(file_path):
    """Загрузка исходных данных"""
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """Очистка данных"""
    # Удаление ID колонки если существует
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)
    
    # Проверка и исправление имен колонок
    df.columns = df.columns.str.strip()
    
    # Исправление целевой переменной
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default_payment_next_month'})
    
    return df

def create_features(df):
    """Feature Engineering"""
    # Создание агрегированных признаков из истории платежей
    pay_columns = [col for col in df.columns if col.startswith('PAY_')]
    if pay_columns:
        df['PAY_MEAN'] = df[pay_columns].mean(axis=1)
        df['PAY_STD'] = df[pay_columns].std(axis=1)
        df['PAY_MAX'] = df[pay_columns].max(axis=1)
        df['PAY_MIN'] = df[pay_columns].min(axis=1)
    
    # Агрегация сумм счетов
    bill_columns = [col for col in df.columns if col.startswith('BILL_AMT')]
    if bill_columns:
        df['BILL_AMT_MEAN'] = df[bill_columns].mean(axis=1)
        df['BILL_AMT_STD'] = df[bill_columns].std(axis=1)
        df['BILL_AMT_TOTAL'] = df[bill_columns].sum(axis=1)
    
    # Агрегация сумм платежей
    pay_amt_columns = [col for col in df.columns if col.startswith('PAY_AMT')]
    if pay_amt_columns:
        df['PAY_AMT_MEAN'] = df[pay_amt_columns].mean(axis=1)
        df['PAY_AMT_TOTAL'] = df[pay_amt_columns].sum(axis=1)
    
    # Биннинг возраста
    if 'AGE' in df.columns:
        df['AGE_BINNED'] = pd.cut(
            df['AGE'], 
            bins=[20, 30, 40, 50, 60, 80], 
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
    
    # Создание отношения платежей к счетам
    if 'BILL_AMT_TOTAL' in df.columns and 'PAY_AMT_TOTAL' in df.columns:
        df['PAYMENT_RATIO'] = df['PAY_AMT_TOTAL'] / (df['BILL_AMT_TOTAL'] + 1)  # +1 чтобы избежать деления на 0
    
    # Создание отношения кредитного лимита к общим счетам
    if 'LIMIT_BAL' in df.columns and 'BILL_AMT_TOTAL' in df.columns:
        df['CREDIT_UTILIZATION'] = df['BILL_AMT_TOTAL'] / (df['LIMIT_BAL'] + 1)
    
    return df

def main():
    """Основная функция подготовки данных"""
    print("Starting data preprocessing...")
    
    # Создание директорий если не существуют
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Загрузка данных
    input_path = 'data/raw/default_of_credit_card_clients.csv'
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        print("Please download the dataset from UCI repository and place it in data/raw/")
        sys.exit(1)
    
    df = load_data(input_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Очистка данных
    df = clean_data(df)
    print("Data cleaning completed")
    
    # Feature Engineering
    df = create_features(df)
    print("Feature engineering completed")
    print(f"New data shape: {df.shape}")
    
    # Сохранение полного processed dataset
    df.to_csv('data/processed/full_data.csv', index=False)
    
    # Разделение на train/test
    target_col = 'default_payment_next_month'
    if target_col not in df.columns:
        # Попробуем найти целевую переменную с другим именем
        possible_targets = [col for col in df.columns if 'default' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            print("Error: Could not find target column")
            sys.exit(1)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Сохранение train и test данных
    train_df = X_train.copy()
    train_df[target_col] = y_train
    
    test_df = X_test.copy()
    test_df[target_col] = y_test
    
    train_df.to_csv('data/processed/train.csv', index=False)
    test_df.to_csv('data/processed/test.csv', index=False)
    
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()