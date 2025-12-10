# client_defolt_project/data/make_dataset.py

import pandas as pd
import os
from pathlib import Path

# Определяем корень проекта как родительский каталог для `data/`
PROJECT_ROOT = Path(__file__).parent.parent

def load_data(filepath):
    """Загружает данные из CSV."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Выполняет первичную очистку данных."""
    # Удаляем дубликаты
    df = df.drop_duplicates()
    
    # Приводим названия столбцов к нижнему регистру
    df.columns = df.columns.str.lower()
    
    # Переименовываем целевую переменную
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})
    
    return df

def save_data(df, output_path):
    """Сохраняет данные в CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Используем абсолютные пути относительно PROJECT_ROOT
    input_file = PROJECT_ROOT / "data" / "raw" / "UCI_Credit_Card.csv"
    output_train = PROJECT_ROOT / "data" / "processed" / "train.csv"
    output_test = PROJECT_ROOT / "data" / "processed" / "test.csv"
    
    print("Загрузка данных...")
    df = load_data(input_file)
    print(f"Исходный размер: {df.shape}")
    
    print("Очистка данных...")
    df_clean = clean_data(df)
    print(f"После очистки: {df_clean.shape}")
    
    # Разделяем на train/test (80/20)
    train_df = df_clean.sample(frac=0.8, random_state=42)
    test_df = df_clean.drop(train_df.index)
    
    print(f"Размер train: {train_df.shape}")
    print(f"Размер test: {test_df.shape}")
    
    save_data(train_df, output_train)
    save_data(test_df, output_test)
    
    print("Данные успешно сохранены.")