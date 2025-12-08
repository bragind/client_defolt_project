import locust
from locust import HttpUser, task, between
import numpy as np
import json
import time

class CreditScoringUser(HttpUser):
    wait_time = between(1, 5)
    
    def on_start(self):
        """Подготовка тестовых данных"""
        self.test_data = np.random.randn(1, 30).tolist()
        self.headers = {"Content-Type": "application/json"}
    
    @task
    def predict(self):
        """Запрос предсказания"""
        payload = {
            "features": self.test_data
        }
        
        with self.client.post("/predict", 
                            json=payload,
                            headers=self.headers,
                            catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

# Скрипт для запуска нагрузочного тестирования
def run_load_test():
    import subprocess
    import sys
    
    # Запуск Locust
    cmd = [
        "locust",
        "-f", "scripts/performance/load_testing.py",
        "--host", "http://localhost:8000",
        "--users", "100",
        "--spawn-rate", "10",
        "--run-time", "5m",
        "--headless"
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    run_load_test()