# src/models/quantize_and_benchmark.py
import numpy as np
import json
import time
from pathlib import Path
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

def quantize_model():
    """Квантуем ONNX-модель"""
    print("🔍 Квантование модели...")
    input_model = "models/credit_scoring_nn.onnx"
    output_model = "models/credit_scoring_nn_quantized.onnx"
    
    # Динамическое квантование (для CPU)
    quantize_dynamic(
        input_model,
        output_model,
        weight_type=QuantType.QUInt8
    )
    
    # Размеры
    orig_size = Path(input_model).stat().st_size
    quant_size = Path(output_model).stat().st_size
    reduction = (1 - quant_size / orig_size) * 100
    
    print(f"✅ Квантованная модель сохранена: {output_model}")
    print(f"📦 Размер: {orig_size/1024:.1f} KB → {quant_size/1024:.1f} KB (-{reduction:.1f}%)")
    
    return {
        'original_size_bytes': orig_size,
        'quantized_size_bytes': quant_size,
        'size_reduction_percent': reduction
    }

def load_test_data():
    """Загрузка тестовых данных"""
    try:
        return np.load('data/processed/test_batch.npy').astype(np.float32)
    except FileNotFoundError:
        import pandas as pd
        df = pd.read_csv('data/processed/test.csv')
        if 'default_payment_next_month' in df.columns:
            df = df.drop(columns=['default_payment_next_month'])
        return df.values.astype(np.float32)

def validate_quantized_model():
    """Проверка точности квантованной модели"""
    print("\n🔍 Валидация точности квантованной модели...")
    test_data = load_test_data()[:100]  # 100 сэмплов для проверки
    
    # Загружаем оригинальную и квантованную модели
    orig_session = ort.InferenceSession("models/credit_scoring_nn.onnx")
    quant_session = ort.InferenceSession("models/credit_scoring_nn_quantized.onnx")
    
    input_name = orig_session.get_inputs()[0].name
    
    orig_outputs = []
    quant_outputs = []
    
    for i in range(len(test_data)):
        inp = test_data[i:i+1]
        orig_out = orig_session.run(None, {input_name: inp})[0][0, 0]
        quant_out = quant_session.run(None, {input_name: inp})[0][0, 0]
        orig_outputs.append(orig_out)
        quant_outputs.append(quant_out)
    
    orig_outputs = np.array(orig_outputs)
    quant_outputs = np.array(quant_outputs)
    
    mse = np.mean((orig_outputs - quant_outputs) ** 2)
    max_diff = np.max(np.abs(orig_outputs - quant_outputs))
    corr = np.corrcoef(orig_outputs, quant_outputs)[0, 1]
    
    is_valid = mse < 1e-3  # допуск выше, чем для конвертации
    
    result = {
        'mse': float(mse),
        'max_absolute_difference': float(max_diff),
        'correlation': float(corr),
        'quantization_valid': bool(is_valid)
    }
    
    print(f"  MSE: {mse:.2e}")
    print(f"  Макс. разница: {max_diff:.4f}")
    print(f"  Корреляция: {corr:.6f}")
    print(f"  Статус: {'✅ Успешно' if is_valid else '⚠️ Значительная потеря точности'}")
    
    return result

def benchmark_quantized_model():
    """Сравнение производительности: оригинальная vs квантованная"""
    print("\n⏱️  Бенчмарк квантованной модели...")
    test_data = load_test_data()
    
    orig_session = ort.InferenceSession("models/credit_scoring_nn.onnx")
    quant_session = ort.InferenceSession("models/credit_scoring_nn_quantized.onnx")
    input_name = orig_session.get_inputs()[0].name
    
    results = {}
    batch_sizes = [1, 10, 100, min(1000, len(test_data))]
    
    for batch_size in batch_sizes:
        batch = test_data[:batch_size]
        
        # Оригинальная модель
        for _ in range(5):  # warmup
            _ = orig_session.run(None, {input_name: batch[:min(10, batch_size)]})
        orig_times = []
        for _ in range(30):
            start = time.perf_counter()
            _ = orig_session.run(None, {input_name: batch})
            orig_times.append(time.perf_counter() - start)
        
        # Квантованная модель
        for _ in range(5):  # warmup
            _ = quant_session.run(None, {input_name: batch[:min(10, batch_size)]})
        quant_times = []
        for _ in range(30):
            start = time.perf_counter()
            _ = quant_session.run(None, {input_name: batch})
            quant_times.append(time.perf_counter() - start)
        
        orig_mean = np.mean(orig_times)
        quant_mean = np.mean(quant_times)
        speedup = orig_mean / quant_mean if quant_mean > 0 else 0
        
        results[f'batch_{batch_size}'] = {
            'original_mean_ms': float(orig_mean * 1000),
            'quantized_mean_ms': float(quant_mean * 1000),
            'speedup': float(speedup),
            'throughput_original': float(batch_size / orig_mean),
            'throughput_quantized': float(batch_size / quant_mean)
        }
        
        print(f"\nBatch {batch_size}:")
        print(f"  Оригинал: {orig_mean*1000:.2f} ms")
        print(f"  Квантованная: {quant_mean*1000:.2f} ms")
        print(f"  Ускорение: {speedup:.2f}x")
    
    return results

def main():
    print("🚀 Квантование ONNX-модели и бенчмарк")
    
    # 1. Квантование
    size_result = quantize_model()
    
    # 2. Валидация точности
    validation_result = validate_quantized_model()
    
    # 3. Бенчмарк производительности
    benchmark_result = benchmark_quantized_model()
    
    # 4. Сохранение результатов
    Path('reports/metrics').mkdir(parents=True, exist_ok=True)
    
    full_result = {
        'size': size_result,
        'validation': validation_result,
        'benchmark': benchmark_result
    }
    
    with open('reports/metrics/quantization_results.json', 'w') as f:
        json.dump(full_result, f, indent=2)
    
    print("\n" + "="*60)
    print("✅ КВАНТОВАНИЕ ЗАВЕРШЕНО!")
    print("Результаты сохранены в:")
    print("  → reports/metrics/quantization_results.json")
    print("="*60)

if __name__ == "__main__":
    main()