# src/benchmark/generate_report.py
import json
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

def load_results():
    """Загружает все результаты бенчмарков"""
    results = {}
    
    # Resource benchmark
    try:
        with open("reports/benchmark/resource_benchmark.json") as f:
            results["resource"] = json.load(f)
    except FileNotFoundError:
        results["resource"] = None
    
    # Load test results (предполагаем, что есть несколько файлов)
    load_test_files = list(Path("reports/benchmark").glob("load_test_rps_*.json"))
    results["load_tests"] = []
    for file in sorted(load_test_files):
        with open(file) as f:
            results["load_tests"].append(json.load(f))
    
    return results

def generate_markdown_report(results):
    """Генерирует отчёт в формате Markdown"""
    report = []
    report.append("# 📊 Benchmark Report: Credit Scoring Model")
    report.append(f"**Дата генерации**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Системная информация
    if results["resource"]:
        sys_info = results["resource"]["system_info"]
        report.append("## 💻 Системная конфигурация")
        report.append(f"- **CPU**: {sys_info['cpu_count']} ядер, {sys_info['cpu_freq']:.0f} MHz")
        report.append(f"- **Память**: {sys_info['memory_gb']:.1f} GB")
        report.append(f"- **GPU**: {'Да' if sys_info['has_gpu'] else 'Нет'}")
        report.append("")
    
    # Resource benchmark
    if results["resource"]:
        report.append("## 📈 Бенчмарк производительности")
        report.append("| Batch Size | Latency (ms) | P95 Latency | CPU (%) | Memory (MB) | Throughput (req/sec) |")
        report.append("|------------|--------------|-------------|---------|-------------|---------------------|")
        
        for bench in results["resource"]["benchmarks"]:
            report.append(f"| {bench['batch_size']} | {bench['avg_latency_ms']:.2f} | "
                         f"{bench['p95_latency_ms']:.2f} | {bench['avg_cpu_percent']:.1f} | "
                         f"{bench['avg_memory_mb']:.1f} | {bench['throughput']:.0f} |")
        report.append("")
    
    # Load testing
    if results["load_tests"]:
        report.append("## 🚀 Нагрузочное тестирование")
        report.append("| Target RPS | Actual RPS | Error Rate | P95 Latency (ms) |")
        report.append("|------------|------------|------------|------------------|")
        
        for test in results["load_tests"]:
            report.append(f"| {test['target_rps']} | {test['actual_rps']:.1f} | "
                         f"{test['error_rate']:.2%} | {test['p95_latency_ms']:.1f} |")
        report.append("")
    
    # Рекомендации
    report.append("## 🎯 Рекомендации для продакшена")
    
    if results["resource"]:
        best_bench = max(results["resource"]["benchmarks"], key=lambda x: x["throughput"])
        report.append(f"- **Оптимальный batch size**: {best_bench['batch_size']}")
        report.append(f"- **Ожидаемая пропускная способность**: {best_bench['throughput']:.0f} запросов/сек")
    
    if results["load_tests"]:
        stable_tests = [t for t in results["load_tests"] if t["error_rate"] < 0.01]
        if stable_tests:
            max_stable = max(stable_tests, key=lambda x: x["target_rps"])
            report.append(f"- **Максимальная стабильная нагрузка**: {max_stable['target_rps']} RPS")
        else:
            report.append("- **Рекомендуется снизить нагрузку** до уровня с ошибками < 1%")
    
    report.append("")
    report.append("## 📁 Используемые артефакты")
    if results["resource"]:
        report.append(f"- **Модель**: {results['resource']['model_path']}")
    report.append("- **Данные**: data/processed/test.csv")
    
    return "\n".join(report)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/benchmark/benchmark_report.md")
    args = parser.parse_args()
    
    # Загружаем результаты
    results = load_results()
    
    # Генерируем отчёт
    markdown_report = generate_markdown_report(results)
    
    # Сохраняем
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"✅ Отчёт сгенерирован: {args.output}")
    
    # Выводим краткую сводку в консоль
    if results["resource"]:
        best = max(results["resource"]["benchmarks"], key=lambda x: x["throughput"])
        print(f"💡 Оптимальная конфигурация: batch_size={best['batch_size']}, "
              f"throughput={best['throughput']:.0f} req/sec")

if __name__ == "__main__":
    main()