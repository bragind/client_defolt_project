import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import time
import json
from pathlib import Path

def quantize_onnx_model():
    """Динамическое квантование ONNX модели"""
    
    # Пути к моделям
    input_model_path = 'models/credit_scoring_nn.onnx'
    quantized_model_path = 'models/credit_scoring_nn_quantized.onnx'
    
    # Динамическое квантование
    quantize_dynamic(
        input_model_path,
        quantized_model_path,
        weight_type=QuantType.QUInt8  # 8-битное целое без знака
    )
    
    print(f"Квантованная модель сохранена: {quantized_model_path}")
    
    # Сравнение размеров моделей
    original_size = Path(input_model_path).stat().st_size / 1024  # KB
    quantized_size = Path(quantized_model_path).stat().st_size / 1024  # KB
    
    # Бенчмарк производительности
    benchmark_results = compare_performance(
        input_model_path, 
        quantized_model_path
    )
    
    # Сохранение результатов
    results = {
        'original_size_kb': original_size,
        'quantized_size_kb': quantized_size,
        'size_reduction_percent': ((original_size - quantized_size) / original_size) * 100,
        'performance_comparison': benchmark_results
    }
    
    with open('reports/metrics/quantization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Оригинальный размер: {original_size:.2f} KB")
    print(f"Квантованный размер: {quantized_size:.2f} KB")
    print(f"Сокращение размера: {results['size_reduction_percent']:.2f}%")
    
    return results

def compare_performance(original_model_path, quantized_model_path):
    """Сравнение производительности оригинальной и квантованной моделей"""
    
    import onnxruntime as ort
    test_data = np.load('data/processed/test_batch.npy')
    
    results = {}
    
    for model_name, model_path in [('original', original_model_path), 
                                  ('quantized', quantized_model_path)]:
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        # Прогрев
        for _ in range(10):
            session.run(None, {input_name: test_data[:10].astype(np.float32)})
        
        # Измерение времени инференса
        times = []
        for batch_size in [1, 10, 100, 1000]:
            batch = test_data[:batch_size]
            
            start_time = time.perf_counter()
            for _ in range(100):  # Многократные запуски для точности
                session.run(None, {input_name: batch.astype(np.float32)})
            elapsed = (time.perf_counter() - start_time) / 100
            
            times.append({
                'batch_size': batch_size,
                'inference_time_ms': elapsed * 1000,
                'throughput_samples_per_second': batch_size / elapsed
            })
        
        results[model_name] = times
    
    return results

if __name__ == "__main__":
    quantize_onnx_model()