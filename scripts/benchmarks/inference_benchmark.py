# scripts/benchmarks/inference_benchmark.py
import time
import numpy as np
import onnxruntime as ort
import joblib

def benchmark_inference(model_path, onnx_path, num_samples=10000, warmup=100):
    # Генерация данных
    X = np.random.randn(num_samples, 20).astype(np.float32)
    
    # Scikit-learn
    model = joblib.load(model_path)
    for _ in range(warmup):
        model.predict_proba(X[:1])
    start = time.perf_counter()
    for i in range(0, num_samples, 100):
        model.predict_proba(X[i:i+100])
    sk_time = time.perf_counter() - start
    print(f"⏱️  Scikit-learn: {sk_time:.2f} sec")
    
    # ONNX Runtime
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {input_name: X[:1]})
    start = time.perf_counter()
    for i in range(0, num_samples, 100):
        sess.run(None, {input_name: X[i:i+100]})
    onnx_time = time.perf_counter() - start
    print(f"⏱️  ONNX Runtime: {onnx_time:.2f} sec")
    
    speedup = sk_time / onnx_time
    print(f"🚀 Ускорение: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark_inference(
        model_path="models/trained/credit_default_model.pkl",
        onnx_path="models/trained/model.onnx"
    )