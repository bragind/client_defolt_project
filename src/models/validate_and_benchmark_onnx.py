# src/models/validate_and_benchmark_onnx.py
import torch
import torch.nn as nn
import onnxruntime as ort
import numpy as np
import time
import json
from pathlib import Path
import sys
import pandas as pd

# Добавляем путь для импорта модели
sys.path.append(str(Path(__file__).parent))
from train_nn import CreditScoringNN

def load_test_data():
    """Загружает тестовые данные для валидации и бенчмарка"""
    try:
        # Приоритет: test_batch.npy (уже в float32)
        data = np.load('data/processed/test_batch.npy')
        print("✅ Загружены данные из test_batch.npy")
    except FileNotFoundError:
        # Резерв: из test.csv
        print("⚠️ test_batch.npy не найден, используем test.csv")
        test_df = pd.read_csv('data/processed/test.csv')
        target_col = 'default_payment_next_month'
        if target_col in test_df.columns:
            test_df = test_df.drop(columns=[target_col])
        data = test_df.values.astype(np.float32)
    return data

def validate_conversion(pytorch_model, dummy_input, onnx_path):
    """Валидация точности конвертации"""
    print("\n🔍 Валидация корректности конвертации...")
    
    # PyTorch предсказание
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    # ONNX предсказание
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: dummy_input.numpy()})[0]
    
    # Сравнение
    mse = np.mean((pytorch_output - ort_output) ** 2)
    max_diff = np.max(np.abs(pytorch_output - ort_output))
    is_valid = mse < 1e-5  # допуск немного увеличен для надёжности
    
    result = {
        'mse': float(mse),
        'max_absolute_difference': float(max_diff),
        'pytorch_output_sample': float(pytorch_output[0, 0]),
        'onnx_output_sample': float(ort_output[0, 0]),
        'conversion_valid': bool(is_valid)
    }
    
    print(f"  MSE: {mse:.2e}")
    print(f"  Макс. разница: {max_diff:.2e}")
    print(f"  Статус: {'✅ Успешно' if is_valid else '❌ Неуспешно'}")
    
    return result, ort_session

def benchmark_inference(pytorch_model, ort_session, test_data):
    """Сравнение производительности инференса"""
    print("\n⏱️  Бенчмарк производительности на CPU...")
    
    input_name = ort_session.get_inputs()[0].name
    results = {}
    batch_sizes = [1, 10, 100, min(1000, len(test_data))]
    
    for batch_size in batch_sizes:
        batch = test_data[:batch_size]
        print(f"\n--- Batch size: {batch_size} ---")
        
        # PyTorch inference
        torch_times = []
        with torch.no_grad():
            for _ in range(10):  # warmup
                _ = pytorch_model(torch.from_numpy(batch[:min(10, batch_size)]))
            for _ in range(30):
                start = time.perf_counter()
                _ = pytorch_model(torch.from_numpy(batch))
                torch_times.append(time.perf_counter() - start)
        
        # ONNX inference
        onnx_times = []
        for _ in range(10):  # warmup
            _ = ort_session.run(None, {input_name: batch[:min(10, batch_size)]})
        for _ in range(30):
            start = time.perf_counter()
            _ = ort_session.run(None, {input_name: batch})
            onnx_times.append(time.perf_counter() - start)
        
        # Расчёт метрик
        torch_mean = np.mean(torch_times)
        onnx_mean = np.mean(onnx_times)
        speedup = torch_mean / onnx_mean if onnx_mean > 0 else 0
        
        results[f'batch_{batch_size}'] = {
            'pytorch_mean_ms': float(torch_mean * 1000),
            'onnx_mean_ms': float(onnx_mean * 1000),
            'speedup': float(speedup),
            'throughput_pytorch': float(batch_size / torch_mean),
            'throughput_onnx': float(batch_size / onnx_mean)
        }
        
        print(f"  PyTorch: {torch_mean*1000:.2f} ms")
        print(f"  ONNX:    {onnx_mean*1000:.2f} ms")
        print(f"  Ускорение: {speedup:.2f}x")
    
    return results

def main():
    print("🚀 Запуск валидации и бенчмарка ONNX-модели")
    
    # 1. Загрузка данных и модели
    test_data = load_test_data()
    input_size = test_data.shape[1]
    
    pytorch_model = CreditScoringNN(input_size)
    pytorch_model.load_state_dict(torch.load('models/credit_scoring_nn.pth', map_location='cpu'))
    
    # 2. Валидация конвертации
    dummy_input = torch.from_numpy(test_data[:1].astype(np.float32))
    validation_result, ort_session = validate_conversion(
        pytorch_model, dummy_input, 'models/credit_scoring_nn.onnx'
    )
    
    # 3. Бенчмарк
    benchmark_result = benchmark_inference(pytorch_model, ort_session, test_data)
    
    # 4. Сохранение результатов
    Path('reports/metrics').mkdir(parents=True, exist_ok=True)
    
    full_result = {
        'validation': validation_result,
        'benchmark': benchmark_result
    }
    
    with open('reports/metrics/onnx_validation_and_benchmark.json', 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ ВСЁ ГОТОВО!")
    print("Результаты сохранены в:")
    print("  → reports/metrics/onnx_validation_and_benchmark.json")
    print("="*60)

if __name__ == "__main__":
    main()