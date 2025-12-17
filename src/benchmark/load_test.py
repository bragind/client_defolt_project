# src/benchmark/load_test.py
import asyncio
import aiohttp
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

async def send_request(session, url, data):
    start = time.perf_counter()
    async with session.post(url, json=data) as resp:
        result = await resp.json()
        latency = time.perf_counter() - start
        return latency, resp.status

async def load_test(endpoint, test_data, rps, duration):
    """Нагрузочное тестирование с целевым RPS"""
    print(f"🚀 Запуск нагрузочного теста: {rps} RPS, {duration} сек")
    
    # Готовим данные
    sample = test_data[0].tolist()
    request_data = {"features": sample}
    
    latencies = []
    errors = 0
    start_time = time.perf_counter()
    request_count = 0
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.perf_counter() - start_time < duration:
            # Рассчитываем интервал для достижения RPS
            interval = 1.0 / rps
            tasks = []
            
            # Отправляем пакет запросов
            batch_size = max(1, int(rps * 0.1))  # 10% от RPS за раз
            for _ in range(batch_size):
                if time.perf_counter() - start_time >= duration:
                    break
                tasks.append(send_request(session, endpoint, request_data))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        errors += 1
                    else:
                        latency, status = result
                        if status == 200:
                            latencies.append(latency)
                        else:
                            errors += 1
                request_count += len(tasks)
            
            # Пауза для соблюдения RPS
            await asyncio.sleep(max(0, interval * batch_size - (time.perf_counter() - start_time)))
    
    # Вычисляем метрики
    if latencies:
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)
    else:
        p50 = p95 = p99 = avg_latency = 0
    
    actual_rps = request_count / (time.perf_counter() - start_time)
    
    result = {
        "target_rps": rps,
        "actual_rps": actual_rps,
        "duration_sec": duration,
        "total_requests": request_count,
        "successful_requests": len(latencies),
        "error_rate": errors / (request_count or 1),
        "avg_latency_ms": avg_latency * 1000,
        "p50_latency_ms": p50 * 1000,
        "p95_latency_ms": p95 * 1000,
        "p99_latency_ms": p99 * 1000
    }
    
    print(f"✅ RPS: {actual_rps:.1f}, Ошибки: {result['error_rate']:.2%}, "
          f"P95: {p95*1000:.1f}ms")
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000/predict")
    parser.add_argument("--rps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--output", default="load_test_results.json")
    args = parser.parse_args()
    
    # Загружаем тестовые данные
    test_df = pd.read_csv("data/processed/test.csv")
    target_col = "default_payment_next_month"
    if target_col in test_df.columns:
        test_df = test_df.drop(columns=[target_col])
    test_data = test_df.values.astype(np.float32)
    
    # Запускаем тест
    result = asyncio.run(load_test(args.endpoint, test_data, args.rps, args.duration))
    
    # Сохраняем
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n💾 Результаты сохранены в: {args.output}")

if __name__ == "__main__":
    main()