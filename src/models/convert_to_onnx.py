# src/models/convert_to_onnx.py
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Импортируем модель
from train_nn import CreditScoringNN

def convert_to_onnx():
    # Загрузка размера входа из sample_input
    sample_data = np.load('data/processed/sample_input.npy')
    input_size = sample_data.shape[1]
    
    # Создание и загрузка модели
    model = CreditScoringNN(input_size)
    model.load_state_dict(torch.load('models/credit_scoring_nn.pth', map_location='cpu'))
    model.eval()
    
    # Создание dummy input
    dummy_input = torch.randn(1, input_size, dtype=torch.float32)
    
    # Экспорт в ONNX
    onnx_path = 'models/credit_scoring_nn.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
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
    
    print(f"✅ Модель успешно экспортирована в {onnx_path}")
    validate_onnx_conversion(model, dummy_input)

def validate_onnx_conversion(pytorch_model, dummy_input):
    """Проверка корректности конвертации"""
    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()
    
    ort_session = ort.InferenceSession('models/credit_scoring_nn.onnx')
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]
    
    mse = np.mean((pytorch_output - ort_output) ** 2)
    is_valid = mse < 1e-6
    
    result = {
        'mse': float(mse),
        'pytorch_output': float(pytorch_output[0][0]),
        'onnx_output': float(ort_output[0][0]),
        'valid': bool(is_valid)  # ← ИСПРАВЛЕНО
    }
    
    Path('reports/metrics').mkdir(parents=True, exist_ok=True)
    with open('reports/metrics/onnx_validation.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"🔍 Валидация: MSE = {mse:.2e} → {'✅ Успешно' if is_valid else '❌ Неуспешно'}")

if __name__ == "__main__":
    convert_to_onnx()