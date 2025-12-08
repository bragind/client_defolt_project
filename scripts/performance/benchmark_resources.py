import subprocess
import json
import time
from dataclasses import dataclass
from typing import Dict, List
import psutil

@dataclass
class ResourceConfig:
    cpu_limit: str
    memory_limit: str
    instance_type: str

def benchmark_different_configs():
    """Тестирование производительности на разных конфигурациях ресурсов"""
    
    configs = [
        ResourceConfig("100m", "128Mi", "small-cpu"),
        ResourceConfig("500m", "512Mi", "medium-cpu"),
        ResourceConfig("1000m", "1Gi", "large-cpu"),
        ResourceConfig("2000m", "2Gi", "xlarge-cpu"),
    ]
    
    results = {}
    
    for config in configs:
        print(f"Тестирование конфигурации: {config.instance_type}")
        
        # Запуск контейнера с указанными ресурсами
        container_id = start_container_with_resources(config)
        
        # Запуск нагрузочного тестирования
        performance_metrics = run_performance_test(container_id)
        
        # Сбор метрик ресурсов
        resource_metrics = collect_resource_metrics(container_id)
        
        # Остановка контейнера
        stop_container(container_id)
        
        results[config.instance_type] = {
            "config": config.__dict__,
            "performance": performance_metrics,
            "resource_usage": resource_metrics,
            "cost_effectiveness": calculate_cost_effectiveness(
                performance_metrics, 
                config
            )
        }
    
    # Сохранение результатов
    with open('reports/metrics/resource_benchmark.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Определение оптимальной конфигурации
    optimal_config = determine_optimal_configuration(results)
    
    print(f"\nОптимальная конфигурация для продакшена: {optimal_config}")
    
    return results, optimal_config

def calculate_cost_effectiveness(performance: Dict, config: ResourceConfig) -> float:
    """Расчет cost-effectiveness (запросов в секунду на единицу CPU)"""
    
    # Примерная стоимость ресурсов (условные единицы)
    cost_per_cpu = {
        "100m": 1.0,
        "500m": 4.0,
        "1000m": 7.0,
        "2000m": 13.0
    }
    
    rps = performance.get('requests_per_second', 0)
    cpu_cost = cost_per_cpu.get(config.cpu_limit, 1.0)
    
    return rps / cpu_cost

if __name__ == "__main__":
    benchmark_different_configs()