# src/benchmark/resource_benchmark.py
import psutil
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import subprocess
import sys

def get_system_info():
    """Получает информацию о системе"""
    return {
        "cpu_count": psutil.cpu_count(),
        "cpu_freq": psutil.cpu_freq().max if psutil.cpu_freq() else 0,
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "has_gpu": False,
        "gpu_info": ""
    }

def run_inference_benchmark(model_path, test_data, batch_size=1):
    """Запускает бенчмарк инференса и измеряет ресурсы"""
    # Загружаем ONNX модель
    import onnxruntime as ort
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    # Прогрев
    for _ in range(10):
        _ = session.run(None, {input_name: test_data[:min(10, len(test_data))]})
    
    # Замер ресурсов
    cpu_percentages = []
    memory_usages = []
    latencies = []
    
    iterations = 100
    for _ in range(iterations):
        # Замер до
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().used
        
        # Инференс
        start = time.perf_counter()
        batch = test_data[:batch_size]
        _ = session.run(None, {input_name: batch.astype(np.float32)})
        latency = time.perf_counter() - start
        
        # Замер после
        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().used
        
        latencies.append(latency)
        cpu_percentages.append((cpu_before + cpu_after) / 2)
        memory_usages.append((mem_after - mem_before) / (1024**2))  # MB
    
    return {
        "batch_size": batch_size,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000,
        "avg_cpu_percent": np.mean(cpu_percentages),
        "avg_memory_mb": np.mean(memory_usages),
        "throughput": batch_size / np.mean(latencies)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/credit_scoring_nn_quantized.onnx")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 10, 100])
    parser.add_argument("--output", default="resource_benchmark.json")
    args = parser.parse_args()
    
    # Загружаем данные
    test_df = pd.read_csv("data/processed/test.csv")
    target_col = "default_payment_next_month"
    if target_col in test_df.columns:
        test_df = test_df.drop(columns=[target_col])
    test_data = test_df.values.astype(np.float32)
    
    # Системная информация
    system_info = get_system_info()
    
    # Бенчмарк для разных batch sizes
    results = []
    for batch_size in args.batch_sizes:
        print(f"🔄 Тестирование batch_size={batch_size}...")
        result = run_inference_benchmark(args.model, test_data, batch_size)
        results.append(result)
    
    # Итоговый отчёт
    report = {
        "system_info": system_info,
        "model_path": args.model,
        "benchmarks": results
    }
    
    # Сохраняем
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Результаты сохранены в: {args.output}")
    
    # Вывод рекомендаций
    best_throughput = max(results, key=lambda x: x["throughput"])
    print(f"\n💡 Рекомендация: используйте batch_size={best_throughput['batch_size']} "
          f"для максимальной пропускной способности ({best_throughput['throughput']:.0f} req/sec)")

if __name__ == "__main__":
    main()