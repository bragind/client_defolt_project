import pandas as pd
import numpy as np
import joblib
import os
import json
import sys

# Добавляем путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.preprocess import create_features

def test_model_simple():
    """Простой тест модели"""
    print("=== SIMPLE MODEL TEST ===")
    
    # 1. Проверка что модель существует
    if not os.path.exists('models/best_model.pkl'):
        print("❌ Model file not found. Please train the model first.")
        return False
    
    # 2. Загрузка модели
    try:
        model = joblib.load('models/best_model.pkl')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # 3. Проверка структуры
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, 'named_steps'):
        print(f"Pipeline steps: {list(model.named_steps.keys())}")
        classifier = model.named_steps['classifier']
        print(f"Classifier: {type(classifier).__name__}")
    
    # 4. Тестовое предсказание
    test_data_raw = pd.DataFrame({
        'LIMIT_BAL': [20000],
        'SEX': [2],
        'EDUCATION': [1],
        'MARRIAGE': [1],
        'AGE': [35],
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
        'PAY_AMT6': [1000]
    })
    
    try:
        # Создаем фичи
        test_data = create_features(test_data_raw)
        
        prediction = model.predict(test_data)[0]
        probability = model.predict_proba(test_data)[0]
        
        print(f"✓ Prediction successful")
        print(f"  Predicted class: {prediction} ({'No Default' if prediction == 0 else 'Default'})")
        print(f"  Probabilities: [No Default: {probability[0]:.3f}, Default: {probability[1]:.3f}]")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # 5. Проверка метрик обучения
    if os.path.exists('reports/training_metrics.json'):
        with open('reports/training_metrics.json', 'r') as f:
            metrics = json.load(f)
        print(f"✓ Training metrics found")
        print(f"  Best model: {metrics.get('best_model', 'Unknown')}")
        print(f"  Best ROC-AUC: {metrics.get('best_score', 'Unknown'):.4f}")
    
    print("\n🎉 Model test completed successfully!")
    return True

if __name__ == "__main__":
    test_model_simple()