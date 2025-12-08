import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import json

def convert_to_onnx():
    # Загрузка модели PyTorch
    from train_nn import CreditScoringNN
    
    # Определение размера входных данных
    sample_data = np.load('data/processed/sample_input.npy')
    input_size = sample_data.shape[1]
    
    # Создание и загрузка модели
    model = CreditScoringNN(input_size)
    model.load_state_dict(torch.load('models/credit_scoring_nn.pth'))
    model.eval()
    
    # Создание примера входных данных
    dummy_input = torch.randn(1, input_size, dtype=torch.float32)
    
    # Экспорт в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        'models/credit_scoring_nn.onnx',
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Модель успешно конвертирована в ONNX")
    
    # Валидация конвертации
    validate_onnx_conversion(model, dummy_input)

def validate_onnx_conversion(pytorch_model, dummy_input):
    """Валидация корректности конвертации"""
    
    # Получение предсказания PyTorch модели
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    # Получение предсказания ONNX модели
    ort_session = ort.InferenceSession('models/credit_scoring_nn.onnx')
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    # Сравнение результатов
    mse = np.mean((pytorch_output - ort_output) ** 2)
    
    validation_result = {
        'mse_between_predictions': float(mse),
        'pytorch_prediction': float(pytorch_output[0][0]),
        'onnx_prediction': float(ort_output[0][0]),
        'absolute_difference': float(abs(pytorch_output[0][0] - ort_output[0][0])),
        'conversion_valid': mse < 1e-6
    }
    
    # Сохранение результатов валидации
    with open('reports/metrics/onnx_validation.json', 'w') as f:
        json.dump(validation_result, f, indent=2)
    
    print(f"MSE между предсказаниями: {mse:.10f}")
    print(f"Конвертация {'успешна' if mse < 1e-6 else 'неуспешна'}")
    
    return validation_result

def benchmark_models():
    """Сравнение производительности PyTorch и ONNX моделей"""
    
    # Загрузка тестовых данных
    test_data = np.load('data/processed/test_batch.npy')
    
    # Загрузка моделей
    sample_data = np.load('data/processed/sample_input.npy')
    input_size = sample_data.shape[1]
    
    # PyTorch модель
    from train_nn import CreditScoringNN
    pytorch_model = CreditScoringNN(input_size)
    pytorch_model.load_state_dict(torch.load('models/credit_scoring_nn.pth'))
    pytorch_model.eval()
    
    # ONNX модель
    ort_session = ort.InferenceSession('models/credit_scoring_nn.onnx')
    input_name = ort_session.get_inputs()[0].name
    
    # Тестирование производительности на CPU
    results = {}
    
    for batch_size in [1, 10, 100, 1000]:
        batch = test_data[:batch_size]
        
        # PyTorch инференс
        start_time = time.time()
        with torch.no_grad():
            pytorch_input = torch.FloatTensor(batch)
            pytorch_output = pytorch_model(pytorch_input)
        pytorch_time = time.time() - start_time
        
        # ONNX инференс
        start_time = time.time()
        ort_output = ort_session.run(None, {input_name: batch.astype(np.float32)})
        onnx_time = time.time() - start_time
        
        results[f'batch_{batch_size}'] = {
            'pytorch_time_ms': pytorch_time * 1000,
            'onnx_time_ms': onnx_time * 1000,
            'speedup': pytorch_time / onnx_time if onnx_time > 0 else 0
        }
    
    # Сохранение результатов бенчмарка
    with open('reports/metrics/performance_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Результаты бенчмарка:")
    for batch, metrics in results.items():
        print(f"{batch}: PyTorch={metrics['pytorch_time_ms']:.2f}ms, "
              f"ONNX={metrics['onnx_time_ms']:.2f}ms, "
              f"Ускорение={metrics['speedup']:.2f}x")
    
    return results

if __name__ == "__main__":
    convert_to_onnx()
    benchmark_models()