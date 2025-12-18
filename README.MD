# Прогнозирование дефолта по кредитной карте

End-to-end MLOps-пайплайн для прогнозирования вероятности дефолта по кредитной карте на основе датасета [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

> 💡 Проект реализует полный жизненный цикл ML: от валидации данных и экспериментов до развертывания модели и мониторинга — **включая нейросетевое обучение и развёртывание в VK Cloud**.

## 🚀 Основные возможности

✅ **Валидация данных** — Great Expectations  
🔄 **Feature Engineering** — автоматическое создание и преобразование признаков  
🧠 **Нейросетевое обучение** — поддержка MLP на PyTorch (через Scikit-learn-совместимый интерфейс)  
📊 **Трекинг экспериментов** — MLflow  
📦 **Версионирование** — DVC для данных и моделей  
🌐 **REST API** — FastAPI для инференса  
🔍 **Мониторинг** — обнаружение дрейфа данных  
🧪 **Тестирование** — полное покрытие `pytest`, `coverage`, `mypy`  
🐳 **Контейнеризация** — Docker + `docker-compose`  
⚙️ **CI/CD** — GitHub Actions  
☁️ **Облачное развертывание** — VK Cloud (Kubernetes + Container Registry)

## 📊 Результаты экспериментов

Сравнение моделей на hold-out выборке:

| Модель                | ROC-AUC | Precision | Recall | F1-Score |
|----------------------|---------|-----------|--------|----------|
| **CatBoost**         | 0.842   | 0.71      | 0.63   | 0.67     |
| **MLP (PyTorch)**    | 0.831   | 0.68      | 0.61   | 0.65     |

- **Порог классификации**: 0.42 (для максимизации F1)
- **Важнейшие признаки**: `PAY_0`, `LIMIT_BAL`, `BILL_AMT1`, `AGE`
- Подробные метрики: `reports/metrics/validation_metrics.json`
- Визуализации: `reports/figures/`

> Лучшей выбрана **CatBoost**, но нейросетевой пайплайн полностью поддерживается и может быть активирован через конфигурацию.

```text
## 📁 Структура проекта
├── data/ # Управление данными
│ ├── raw/ # Исходные данные
│ ├── processed/ # Обработанные данные
│ └── external/ # Внешние данные
├── src/ # Исходный код
│ ├── data/ # Валидация и препроцессинг
│ ├── models/ # ML-пайплайны (включая нейросети)
│ ├── api/ # FastAPI
│ └── monitoring/ # Дрейф и мониторинг
├── models/ # Обученные модели
│ ├── best_model.pkl
│ └── model_versions/
├── reports/ # Метрики и визуализации
├── notebooks/ # EDA и эксперименты
├── scripts/ # Вспомогательные скрипты
├── tests/ # Тесты
├── infrastructure/ # Terraform + K8s для VK Cloud
│ ├── environments
│ └── kubernets
├── .github/workflows/ # CI/CD
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements-*.txt
└── README.md
```text

## 🛠 Быстрый старт (локально)

### Требования
- Python ≥ 3.9
- Docker
- Git

### Установка

```bash
git clone https://github.com/bragind/client_defolt_project.git
cd client_defolt_project

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements-core.txt   # основные зависимости
pip install -r requirements-dev.txt    # для разработки
pip install -r requirements-docker.txt # для контейнеризации

### Запуск пайплайна

```bash
make data      # загрузка и препроцессинг
make train     # обучение модели
make serve     # запуск API локально
# ИЛИ
python scripts/run_pipeline.py

### Развертывание в VK Cloud

Проект готов к развёртыванию в VK Cloud Solutions:

Инфраструктура: infrastructure/terraform/ создаёт:
Managed Kubernetes кластер
VK Container Registry (cr.ru-7.vkcs.ru)
Object Storage для DVC и Terraform state
CI/CD: GitHub Actions автоматически:
Собирает Docker-образ
Пушит в VK CR
Применяет terraform apply
Деплоит в K8s через kubectl apply -k infrastructure/kubernetes/overlays/prod

### Ручное развёртывание

Настройте infrastructure/terraform/terraform.tfvars
```bash
cd infrastructure/terraform
terraform init && terraform apply
kubectl apply -k infrastructure/kubernetes/overlays/prod

###  REST API

## Health check
```bash
curl http://localhost:8000/health

## Одиночный прогноз

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, ...}'

```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'

## 🐳 Запуск в Docker

```bash
docker build -t credit-default-api .
docker run -p 8000:8000 credit-default-api
# ИЛИ
docker-compose up ml-api

## 🧪 Тестирование

```bash
pytest tests/ -v
pytest --cov=src tests/
make lint    # black + flake8 + mypy

## 📈 Мониторинг

```bash
python src/monitoring/drift_detection.py

Результаты: reports/metrics/drift_metrics.json, reports/figures/drift_analysis/

## 📊 MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Открыть: http://localhost:5000

## 📜 Лицензия
Проект создан в образовательных и демонстрационных целях.
