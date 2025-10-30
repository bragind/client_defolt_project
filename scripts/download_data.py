"""
Скрипт для загрузки реальных данных с UCI repository
"""
import pandas as pd
import numpy as np
import os
from urllib.request import urlretrieve

def download_uci_data():
    """Загрузка данных с UCI"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
    output_path = "data/raw/default_of_credit_card_clients.xls"
    
    print("Downloading data from UCI...")
    os.makedirs('data/raw', exist_ok=True)
    
    try:
        urlretrieve(url, output_path)
        print("Download completed!")
        
        # Чтение Excel файла
        df = pd.read_excel(output_path, header=1)
        
        # Сохранение как CSV
        csv_path = "data/raw/default_of_credit_card_clients.csv"
        df.to_csv(csv_path, index=False)
        print(f"Data saved to {csv_path}")
        print(f"Data shape: {df.shape}")
        
        return df
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

if __name__ == "__main__":
    download_uci_data()