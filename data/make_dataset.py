import pandas as pd
import os

def load_data(filepath):
    """Загружает данные из CSV."""
    df = pd.read_csv(filepath)
    return df

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
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    input_file = "../data/raw/UCI_Credit_Card.csv"
    output_train = "../data/processed/train.csv"
    output_test = "../data/processed/test.csv"
    
    print("Загрузка данных...")
    df = load_data(input_file)
    print(f"Исходный размер: {df.shape}")
    
    print("Очистка данных...")
    df_clean = clean_data(df)
    print(f"После очистки: {df_clean.shape}")
    
    # Разделяем на train/test (простое разделение 80/20)
    train_df = df_clean.sample(frac=0.8, random_state=42)
    test_df = df_clean.drop(train_df.index)
    
    print(f"Размер train: {train_df.shape}")
    print(f"Размер test: {test_df.shape}")
    
    save_data(train_df, output_train)
    save_data(test_df, output_test)
    
    print("Данные успешно сохранены.")