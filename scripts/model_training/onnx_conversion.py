# scripts/model_training/onnx_conversion.py
import onnx
import onnxruntime as ort
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

def validate_onnx_conversion(original_model_path, onnx_model_path, test_data_path=None):
    # Загрузка модели
    model = joblib.load(original_model_path)
    
    # Генерация тестовых данных или загрузка
    if test_data_path:
        data = pd.read_csv(test_data_path)
        X = data.drop(columns=['default']).values
    else:
        X, _ = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    # Предсказание оригинальной моделью
    y_pred_orig = model.predict_proba(X)
    
    # Конвертация в ONNX
    initial_type = [('float_input', FloatTensorType([None, X.shape[1])])]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    # Сохранение ONNX
    os.makedirs(os.path.dirname(onnx_model_path), exist_ok=True)
    with open(onnx_model_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    # Загрузка ONNX-модели
    sess = ort.InferenceSession(onnx_model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # Предсказание ONNX
    y_pred_onnx = sess.run([label_name], {input_name: X.astype(np.float32)})[0]
    
    # Сравнение
    diff = np.abs(y_pred_orig - y_pred_onnx)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"✅ Максимальная разница: {max_diff:.6f}")
    print(f"✅ Средняя разница: {mean_diff:.6f}")
    
    assert max_diff < 1e-4, "Конвертация некорректна!"
    print("🎉 Конвертация успешна!")

if __name__ == "__main__":
    validate_onnx_conversion(
        original_model_path="models/trained/credit_default_model.pkl",
        onnx_model_path="models/trained/model.onnx",
        test_data_path="data/processed/test.csv"
    )