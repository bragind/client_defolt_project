# src/models/prepare_onnx_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import traceback
import sys

def main():
    try:
        print("🔍 Starting ONNX data preparation...")
        
        # Проверяем наличие train.csv
        train_path = "data/processed/train.csv"
        if not Path(train_path).exists():
            print(f"❌ ERROR: {train_path} not found!")
            print("💡 Please run: python src/data/preprocess.py")
            sys.exit(1)
        
        print(f"✅ Loading {train_path}...")
        train_df = pd.read_csv(train_path)
        print(f"📊 Loaded {len(train_df)} rows")

        # Определяем целевую переменную
        target_col = "default_payment_next_month"
        if target_col not in train_df.columns:
            print(f"❌ ERROR: Target column '{target_col}' not found in data!")
            print(f"Available columns: {list(train_df.columns)}")
            sys.exit(1)

        X = train_df.drop(columns=[target_col])
        print(f"🧮 Feature matrix shape: {X.shape}")

        # Создаём папку
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Сохраняем
        np.save("data/processed/sample_input.npy", X.values[:1].astype(np.float32))
        np.save("data/processed/test_batch.npy", X.values[:1000].astype(np.float32))

        print("✅ ONNX data files created successfully!")
        print("   → data/processed/sample_input.npy")
        print("   → data/processed/test_batch.npy")

    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()