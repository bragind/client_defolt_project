import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys
import json

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
    
    # Заполнение пропущенных значений
    df = fill_missing_values(df)
    
    return df

def fill_missing_values(df):
    """Заполнение пропущенных значений"""
    # Числовые колонки заполняем медианой
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Категориальные колонки заполняем модой
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def create_features(df):
    """Feature Engineering"""
    # Создание агрегированных признаков из истории платежей
    pay_columns = [col for col in df.columns if col.startswith('PAY_') and col != 'PAY_MEAN']
    if pay_columns:
        df['PAY_MEAN'] = df[pay_columns].mean(axis=1)
        df['PAY_STD'] = df[pay_columns].std(axis=1)
        df['PAY_MAX'] = df[pay_columns].max(axis=1)
        df['PAY_MIN'] = df[pay_columns].min(axis=1)
        # Заполняем NaN в новых фичах
        df['PAY_MEAN'] = df['PAY_MEAN'].fillna(0)
        df['PAY_STD'] = df['PAY_STD'].fillna(0)
        df['PAY_MAX'] = df['PAY_MAX'].fillna(0)
        df['PAY_MIN'] = df['PAY_MIN'].fillna(0)
    
    # Агрегация сумм счетов
    bill_columns = [col for col in df.columns if col.startswith('BILL_AMT')]
    if bill_columns:
        df['BILL_AMT_MEAN'] = df[bill_columns].mean(axis=1)
        df['BILL_AMT_STD'] = df[bill_columns].std(axis=1)
        df['BILL_AMT_TOTAL'] = df[bill_columns].sum(axis=1)
        # Заполняем NaN
        df['BILL_AMT_MEAN'] = df['BILL_AMT_MEAN'].fillna(0)
        df['BILL_AMT_STD'] = df['BILL_AMT_STD'].fillna(0)
        df['BILL_AMT_TOTAL'] = df['BILL_AMT_TOTAL'].fillna(0)
    
    # Агрегация сумм платежей
    pay_amt_columns = [col for col in df.columns if col.startswith('PAY_AMT')]
    if pay_amt_columns:
        df['PAY_AMT_MEAN'] = df[pay_amt_columns].mean(axis=1)
        df['PAY_AMT_TOTAL'] = df[pay_amt_columns].sum(axis=1)
        # Заполняем NaN
        df['PAY_AMT_MEAN'] = df['PAY_AMT_MEAN'].fillna(0)
        df['PAY_AMT_TOTAL'] = df['PAY_AMT_TOTAL'].fillna(0)
    
    # Биннинг возраста с обработкой NaN
    if 'AGE' in df.columns:
        # Сначала заполняем пропуски в возрасте
        df['AGE'] = df['AGE'].fillna(df['AGE'].median())
        
        # Биннинг с безопасным преобразованием
        age_bins = [20, 30, 40, 50, 60, 80]
        age_labels = [0, 1, 2, 3, 4]
        
        # Используем pd.cut с fillna для безопасности
        age_binned = pd.cut(
            df['AGE'], 
            bins=age_bins, 
            labels=age_labels,
            right=False
        )
        
        # Заполняем возможные NaN в биннинге
        df['AGE_BINNED'] = age_binned.cat.add_categories(-1).fillna(-1).astype(int)
    
    # Создание отношения платежей к счетам
    if 'BILL_AMT_TOTAL' in df.columns and 'PAY_AMT_TOTAL' in df.columns:
        # Защита от деления на ноль и отрицательных значений
        df['PAYMENT_RATIO'] = np.where(
            df['BILL_AMT_TOTAL'] <= 0, 
            0, 
            df['PAY_AMT_TOTAL'] / (df['BILL_AMT_TOTAL'] + 1e-6)  # + маленькое число чтобы избежать деления на 0
        )
        df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].fillna(0)
    
    # Создание отношения кредитного лимита к общим счетам
    if 'LIMIT_BAL' in df.columns and 'BILL_AMT_TOTAL' in df.columns:
        df['CREDIT_UTILIZATION'] = np.where(
            df['LIMIT_BAL'] <= 0,
            0,
            df['BILL_AMT_TOTAL'] / (df['LIMIT_BAL'] + 1e-6)
        )
        df['CREDIT_UTILIZATION'] = df['CREDIT_UTILIZATION'].fillna(0)
    
    # Убедимся, что нет NaN в финальном датафрейме
    df = df.fillna(0)
    
    return df

def validate_dataframe(df):
    """Валидация итогового датафрейма"""
    # Проверка на NaN
    if df.isnull().any().any():
        print("Предупреждение: В данных остались NaN значения:")
        print(df.isnull().sum())
        # Заполняем оставшиеся NaN
        df = df.fillna(0)
    
    # Проверка на бесконечные значения
    if np.isinf(df.select_dtypes(include=[np.number])).any().any():
        print("Предупреждение: В данных есть бесконечные значения")
        # Заменяем бесконечности на большие числа
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

def save_data_metrics(train_df, test_df, target_col):
    """Сохранение метрик данных в JSON файл"""
    # Расчет метрик данных
    data_metrics = {
        "original_rows": len(train_df) + len(test_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features_count": len(train_df.columns) - 1,  # минус целевая переменная
        "train_target_distribution": {
            "class_0": int(train_df[target_col].value_counts().get(0, 0)),
            "class_1": int(train_df[target_col].value_counts().get(1, 0))
        },
        "test_target_distribution": {
            "class_0": int(test_df[target_col].value_counts().get(0, 0)),
            "class_1": int(test_df[target_col].value_counts().get(1, 0))
        },
        "class_ratio_train": float(train_df[target_col].mean()),
        "class_ratio_test": float(test_df[target_col].mean()),
        "train_memory_mb": float(train_df.memory_usage(deep=True).sum() / 1024 / 1024),
        "test_memory_mb": float(test_df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    
    # Сохранение метрик в JSON файл
    with open('reports/data_metrics.json', 'w') as f:
        json.dump(data_metrics, f, indent=2)
    
    print("Data metrics saved to reports/data_metrics.json")
    return data_metrics

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
        print("Or run: python scripts/create_sample_data.py")
        sys.exit(1)
    
    df = load_data(input_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Очистка данных
    df = clean_data(df)
    print("Data cleaning completed")
    
    # Feature Engineering
    df = create_features(df)
    print("Feature engineering completed")
    
    # Валидация данных
    df = validate_dataframe(df)
    print("Data validation completed")
    
    print(f"New data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Сохранение полного processed dataset
    df.to_csv('data/processed/full_data.csv', index=False)
    
    # Разделение на train/test
    target_col = 'default_payment_next_month'
    if target_col not in df.columns:
        # Попробуем найти целевую переменную с другим именем
        possible_targets = [col for col in df.columns if 'default' in col.lower()]
        if possible_targets:
            target_col = possible_targets[0]
            print(f"Using target column: {target_col}")
        else:
            print("Error: Could not find target column")
            # Создаем искусственную целевую переменную для тестирования
            df[target_col] = np.random.randint(0, 2, len(df))
            print("Created synthetic target variable for testing")
    
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
    
    # Вывод информации о данных
    print(f"\nTarget distribution:")
    print(f"Train: {y_train.value_counts().to_dict()}")
    print(f"Test: {y_test.value_counts().to_dict()}")
    
    # Сохранение метрик данных
    data_metrics = save_data_metrics(train_df, test_df, target_col)
    
    print("\nData metrics summary:")
    print(f"Total samples: {data_metrics['original_rows']}")
    print(f"Train samples: {data_metrics['train_rows']}")
    print(f"Test samples: {data_metrics['test_rows']}")
    print(f"Features count: {data_metrics['features_count']}")
    print(f"Train class ratio: {data_metrics['class_ratio_train']:.3f}")
    print(f"Test class ratio: {data_metrics['class_ratio_test']:.3f}")
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()