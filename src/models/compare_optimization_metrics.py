# src/models/compare_optimization_metrics.py
import torch
import numpy as np
import time
import json
from pathlib import Path
import sys
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import roc_auc_score, accuracy_score

# Добавляем путь для импорта модели
sys.path.append(str(Path(__file__).parent))
from train_nn import CreditScoringNN

def load_test_data_with_labels():
    """Загружает тестовые данные И метки для оценки качества"""
    test_df = pd.read_csv('data/processed/test.csv')
    target_col = 'default_payment_next_month'
    X = test_df.drop(columns=[target_col]).values.astype(np.float32)
    y = test_df[target_col].values
    return X, y

def evaluate_pytorch_model(model, X, y):
    """Оценка качества и производительности PyTorch-модели"""
    print("  PyTorch: оценка качества...")
    model.eval()
    with torch.no_grad():
        y_proba = model(torch.from_numpy(X)).numpy().flatten()
        y_pred = (y_proba > 0.5).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y, y_proba)),
        'accuracy': float(accuracy_score(y, y_pred))
    }
    
    # Производительность
    print("  PyTorch: замер скорости...")
    times = []
    with torch.no_grad():
        for _ in range(10):  # warmup
            _ = model(torch.from_numpy(X[:10]))
        for _ in range(50):
            start = time.perf_counter()
            _ = model(torch.from_numpy(X))
            times.append(time.perf_counter() - start)
    
    metrics['mean_inference_time_ms'] = float(np.mean(times) * 1000)
    metrics['throughput'] = float(len(X) / np.mean(times))
    
    return metrics

def evaluate_onnx_model(model_path, X, y):
    """Оценка качества и производительности ONNX-модели"""
    print(f"  ONNX ({Path(model_path).name}): оценка качества...")
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    y_proba = []
    for i in range(len(X)):
        inp = X[i:i+1]
        out = session.run(None, {input_name: inp})[0][0, 0]
        y_proba.append(out)
    y_proba = np.array(y_proba)
    y_pred = (y_proba > 0.5).astype(int)
    
    metrics = {
        'roc_auc': float(roc_auc_score(y, y_proba)),
        'accuracy': float(accuracy_score(y, y_pred))
    }
    
    # Производительность
    print(f"  ONNX ({Path(model_path).name}): замер скорости...")
    times = []
    for _ in range(10):  # warmup
        _ = session.run(None, {input_name: X[:10]})
    for _ in range(50):
        start = time.perf_counter()
        _ = session.run(None, {input_name: X})
        times.append(time.perf_counter() - start)
    
    metrics['mean_inference_time_ms'] = float(np.mean(times) * 1000)
    metrics['throughput'] = float(len(X) / np.mean(times))
    
    return metrics

def get_model_size(model_path):
    """Возвращает размер модели в KB"""
    return Path(model_path).stat().st_size / 1024

def main():
    print("📊 Сравнение метрик до и после оптимизации")
    
    # Загрузка данных
    X, y = load_test_data_with_labels()
    print(f"✅ Загружено {len(X)} тестовых сэмплов")
    
    # Инициализация PyTorch-модели
    input_size = X.shape[1]
    pytorch_model = CreditScoringNN(input_size)
    pytorch_model.load_state_dict(torch.load('models/credit_scoring_nn.pth', map_location='cpu'))
    
    # Пути к моделям
    models = {
        'pytorch': {
            'path': 'models/credit_scoring_nn.pth',
            'evaluate': lambda: evaluate_pytorch_model(pytorch_model, X, y)
        },
        'onnx': {
            'path': 'models/credit_scoring_nn.onnx',
            'evaluate': lambda: evaluate_onnx_model('models/credit_scoring_nn.onnx', X, y)
        },
        'quantized': {
            'path': 'models/credit_scoring_nn_quantized.onnx',
            'evaluate': lambda: evaluate_onnx_model('models/credit_scoring_nn_quantized.onnx', X, y)
        }
    }
    
    # Сбор метрик
    results = {}
    for name, config in models.items():
        print(f"\n--- {name.upper()} ---")
        if not Path(config['path']).exists():
            print(f"⚠️  Модель не найдена: {config['path']}")
            results[name] = None
            continue
        
        metrics = config['evaluate']()
        metrics['model_size_kb'] = get_model_size(config['path'])
        results[name] = metrics
    
    # Сохранение
    Path('reports/metrics').mkdir(parents=True, exist_ok=True)
    with open('reports/metrics/optimization_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Вывод сводки
    print("\n" + "="*80)
    print("СВОДКА: СРАВНЕНИЕ МЕТРИК")
    print("="*80)
    print(f"{'Модель':<15} {'ROC-AUC':<10} {'Accuracy':<10} {'Размер (KB)':<12} {'Инф. (мс)':<12} {'Пропуск-ть'}")
    print("-"*80)
    
    for name, metrics in results.items():
        if metrics is None:
            print(f"{name:<15} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12} {'N/A'}")
            continue
        print(f"{name:<15} "
              f"{metrics['roc_auc']:<10.4f} "
              f"{metrics['accuracy']:<10.4f} "
              f"{metrics['model_size_kb']:<12.1f} "
              f"{metrics['mean_inference_time_ms']:<12.2f} "
              f"{metrics['throughput']:.0f}")
    
    print("\n✅ Результаты сохранены в: reports/metrics/optimization_comparison.json")

if __name__ == "__main__":
    main()